// Hearth desktop shell — wraps the built frontend and
// manages the backend as a child process. See desktop/src-tauri/README.md
// for the full picture.
//
// Dev builds spawn `python`/`python3 -m app.main` directly (fast iteration,
// assumes a dev Python env). Release builds instead spawn the
// PyInstaller-frozen backend and a bundled llama-server, both shipped as
// Tauri bundle resources (see tauri.conf.json's bundle.resources and
// backend/hearth-backend.spec) — an installed app needs neither Python nor
// a separately-installed llama-server on the target machine.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(unix)]
use std::os::unix::process::CommandExt as UnixCommandExt;

use tauri::Manager;

/// Move the running macOS bundle to the Trash after the frontend has asked
/// the backend to retain profile identity and erase all other local data.
/// macOS has no uninstall event when a user drags a .app to Trash, so this
/// command is the reliable in-app uninstall route.
#[cfg(target_os = "macos")]
#[tauri::command]
fn move_macos_app_to_trash() -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let app_bundle = executable
        .ancestors()
        .find(|path| path.extension().is_some_and(|extension| extension == "app"))
        .ok_or_else(|| "Hearth is not running from a macOS app bundle.".to_string())?;
    let path = app_bundle
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    let script = format!("tell application \"Finder\" to delete POSIX file \"{path}\"");
    let status = Command::new("osascript")
        .args(["-e", &script])
        .status()
        .map_err(|error| format!("couldn't open Finder: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("Finder couldn't move Hearth to the Trash.".to_string())
    }
}

#[cfg(not(target_os = "macos"))]
#[tauri::command]
fn move_macos_app_to_trash() -> Result<(), String> {
    Err("In-app uninstall is only available on macOS.".to_string())
}

// Windows only: prevents the console window hearth-backend.exe would
// otherwise pop up alongside the app. hearth-backend.spec deliberately
// keeps console=True rather than console=False - PyInstaller sets
// sys.stdout/sys.stderr to None for a console=False (windowed) exe on
// Windows, and this backend calls configure_logging() (main.py) and
// uvicorn.run() (both write to stderr/stdout unconditionally), which
// would crash the instant either logs anything. Hiding the window here
// instead — via the CREATE_NO_WINDOW flag on the spawned process — keeps
// those real file descriptors intact while just not displaying the
// window Windows would otherwise show for a console-subsystem exe.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

struct BackendProcess(Mutex<Option<Child>>);

fn dev_backend_dir() -> PathBuf {
    // Dev layout: desktop/src-tauri/../../backend.
    let mut dir = std::env::current_dir().expect("failed to read cwd");
    dir.pop(); // src-tauri -> desktop
    dir.pop(); // desktop -> repo root
    dir.push("backend");
    dir
}

fn exe_name(base: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn apply_spawn_flags(cmd: &mut Command) {
    // Same process-group / no-console flags for dev and release so
    // kill_process_tree can reach grandchildren on Unix, and so Windows
    // doesn't flash a console for `python -m app.main` in tauri:dev.
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    #[cfg(unix)]
    cmd.process_group(0);
}

/// Per-launch shared secret for the local API. The backend binds to
/// loopback, which keeps the network out but not other processes running as
/// the same user — any of them can otherwise read the whole journal over
/// `127.0.0.1:48173`. Generated here because this process is the only one
/// that can hand it to both ends (env var to the backend child, injected
/// global to the webview) without it ever touching disk or a command line
/// other processes can read.
fn generate_api_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn spawn_backend_dev(api_token: &str, app_version: &str) -> std::io::Result<Child> {
    let dir = dev_backend_dir();
    // Windows typically exposes `python`; many Unix setups only have
    // `python3`. Try both so `tauri:dev` works on either.
    let mut last_err: Option<std::io::Error> = None;
    for python in ["python", "python3"] {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "app.main"])
            .current_dir(&dir)
            .env("HEARTH_API_TOKEN", api_token)
            .env("HEARTH_APP_VERSION", app_version);
        apply_spawn_flags(&mut cmd);
        match cmd.spawn() {
            Ok(child) => return Ok(child),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "python/python3 not found")
    }))
}

/// Release builds: spawn the frozen backend from its bundled resource
/// directory, pointing it at the bundled llama-server via the
/// `LLAMA_SERVER_BIN` env var the backend already reads
/// (backend/app/config.py) — no backend code changes needed for that part.
/// `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` are set so llama-server's
/// accompanying shared libraries (it is NOT a standalone binary — see
/// scripts/fetch_llama_cpp.py) resolve; `subprocess.Popen` in
/// backend/app/llm/server_manager.py doesn't override `env=`, so this
/// chains through to the llama-server child process it spawns in turn.
/// Windows needs no equivalent — its DLL search order checks the launched
/// exe's own directory first, and all the DLLs already sit there.
fn spawn_backend_release(app: &tauri::AppHandle, api_token: &str) -> std::io::Result<Child> {
    let resource_dir = app
        .path()
        .resource_dir()
        .expect("failed to resolve bundled resource directory");

    let backend_exe = resource_dir
        .join("backend")
        .join(exe_name("hearth-backend"));
    let llama_dir = resource_dir.join("llama-cpp");
    let llama_server_bin = llama_dir.join(exe_name("llama-server"));

    // Same string the installer was stamped with (from the v* tag via
    // `tauri build --config '{"version":"..."}'`). Keeps crash reports and
    // backend logs aligned with Add/Remove Programs / .deb version.
    let app_version = app.package_info().version.to_string();

    let mut cmd = Command::new(&backend_exe);
    cmd.env("LLAMA_SERVER_BIN", &llama_server_bin);
    cmd.env("HEARTH_API_TOKEN", api_token);
    cmd.env("HEARTH_APP_VERSION", &app_version);

    if cfg!(target_os = "macos") {
        cmd.env("DYLD_LIBRARY_PATH", &llama_dir);
    } else if !cfg!(target_os = "windows") {
        cmd.env("LD_LIBRARY_PATH", &llama_dir);
    }

    apply_spawn_flags(&mut cmd);

    // New process group (pgid == the backend's own pid) so kill_process_tree
    // can signal the whole tree at once — hearth-backend.exe in turn spawns
    // llama-server and, on first memory use, a second llama-server for
    // embeddings (backend/app/llm/server_manager.py, backend/app/memory/
    // embedder.py), both inherited into this same group since neither calls
    // setpgid itself. Without this, killing only the top-level Child left
    // both of those orphaned and still running after the app window closed.
    cmd.spawn()
}

/// Kills the backend process and every descendant it spawned (llama-server
/// for the LLM, and a second llama-server for embeddings — see
/// spawn_backend_release's process_group comment). `Child::kill()` alone
/// only signals the single direct child, leaving those orphaned.
fn kill_process_tree(child: &mut Child) {
    let pid = child.id();

    #[cfg(unix)]
    unsafe {
        // Negative pid targets the whole process group. SIGTERM first so
        // uvicorn/Python get a chance at their own graceful shutdown, then
        // SIGKILL shortly after for anything still alive.
        libc::kill(-(pid as i32), libc::SIGTERM);
        std::thread::sleep(std::time::Duration::from_millis(500));
        libc::kill(-(pid as i32), libc::SIGKILL);
    }

    #[cfg(windows)]
    {
        // taskkill's /T kills the whole process tree rooted at this pid —
        // the standard way to reach grandchild processes on Windows, where
        // there's no process-group equivalent to signal in one call.
        let mut taskkill = Command::new("taskkill");
        taskkill.args(["/PID", &pid.to_string(), "/T", "/F"]);
        taskkill.creation_flags(CREATE_NO_WINDOW);
        let _ = taskkill.status();
    }

    let _ = child.wait();
}

/// Notify the webview that the backend child failed to start. setup() often
/// runs before the window exists, so retry briefly until `main` is ready.
/// Frontend listens for `backend-spawn-failed` (see useSetupStatus).
fn notify_backend_spawn_failed(app: &tauri::AppHandle, message: &str) {
    let handle = app.clone();
    let payload = serde_json::to_string(message)
        .unwrap_or_else(|_| "\"Couldn't start the companion backend.\"".to_string());
    thread::spawn(move || {
        for _ in 0..50 {
            if let Some(window) = handle.get_webview_window("main") {
                let js = format!(
                    "window.dispatchEvent(new CustomEvent('backend-spawn-failed', {{ detail: {payload} }}));"
                );
                let _ = window.eval(&js);
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        eprintln!("backend-spawn-failed: webview never became ready to receive the error");
    });
}

fn main() {
    let api_token = generate_api_token();
    // Runs before any page script, so the first fetch the app makes already
    // has the token. An initialization script rather than an injected
    // `<script>` tag or an `eval` after load: the CSP set in
    // tauri.conf.json blocks inline scripts, and a token that arrives
    // asynchronously would race the boot-time /api/setup/status call.
    let token_script = format!(
        "window.__HEARTH_API_TOKEN__ = {};",
        serde_json::to_string(&api_token).expect("token is a plain string")
    );
    let spawn_token = api_token.clone();

    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![move_macos_app_to_trash])
        .append_invoke_initialization_script(&token_script)
        .setup(move |app| {
            // package_info().version is whatever `tauri build --config` (or
            // tauri.conf.json) stamped — CI derives it from the v* tag.
            let app_version = app.package_info().version.to_string();
            let result = if cfg!(debug_assertions) {
                spawn_backend_dev(&spawn_token, &app_version)
            } else {
                spawn_backend_release(app.handle(), &spawn_token)
            };
            match result {
                Ok(child) => {
                    let state = app.state::<BackendProcess>();
                    *state.0.lock().unwrap() = Some(child);
                }
                Err(err) => {
                    let msg = format!("Couldn't start the companion backend: {err}");
                    eprintln!("{msg}");
                    notify_backend_spawn_failed(app.handle(), &msg);
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<BackendProcess>();
                if let Some(mut child) = state.0.lock().unwrap().take() {
                    kill_process_tree(&mut child);
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Hearth");
}
