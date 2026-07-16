// Hearth desktop shell — wraps the built frontend and
// manages the backend as a child process. See desktop/src-tauri/README.md
// for the full picture.
//
// Dev builds spawn `python3 -m app.main` directly (fast iteration, assumes
// a dev Python env — unchanged from before). Release builds instead spawn
// the PyInstaller-frozen backend and a bundled llama-server, both shipped
// as Tauri bundle resources (see tauri.conf.json's bundle.resources and
// backend/hearth-backend.spec) — an installed app needs neither
// Python nor a separately-installed llama-server on the target machine.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(unix)]
use std::os::unix::process::CommandExt as UnixCommandExt;

use tauri::Manager;

// Windows only: prevents the console window hearth-backend.exe would
// otherwise pop up alongside the app. hearth-backend.spec deliberately
// keeps console=True rather than console=False - PyInstaller sets
// sys.stdout/sys.stderr to None for a console=False (windowed) exe on
// Windows, and this backend calls logging.basicConfig() (main.py) and
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

fn spawn_backend_dev() -> std::io::Result<Child> {
    let dir = dev_backend_dir();
    Command::new("python3")
        .args(["-m", "app.main"])
        .current_dir(&dir)
        .spawn()
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
fn spawn_backend_release(app: &tauri::AppHandle) -> std::io::Result<Child> {
    let resource_dir = app
        .path()
        .resource_dir()
        .expect("failed to resolve bundled resource directory");

    let backend_exe = resource_dir
        .join("backend")
        .join(exe_name("hearth-backend"));
    let llama_dir = resource_dir.join("llama-cpp");
    let llama_server_bin = llama_dir.join(exe_name("llama-server"));

    let mut cmd = Command::new(&backend_exe);
    cmd.env("LLAMA_SERVER_BIN", &llama_server_bin);

    if cfg!(target_os = "macos") {
        cmd.env("DYLD_LIBRARY_PATH", &llama_dir);
    } else if !cfg!(target_os = "windows") {
        cmd.env("LD_LIBRARY_PATH", &llama_dir);
    }

    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    // New process group (pgid == the backend's own pid) so kill_process_tree
    // can signal the whole tree at once — hearth-backend.exe in turn spawns
    // llama-server and, on first memory use, a second llama-server for
    // embeddings (backend/app/llm/server_manager.py, backend/app/memory/
    // embedder.py), both inherited into this same group since neither calls
    // setpgid itself. Without this, killing only the top-level Child left
    // both of those orphaned and still running after the app window closed.
    #[cfg(unix)]
    cmd.process_group(0);

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

fn main() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let result = if cfg!(debug_assertions) {
                spawn_backend_dev()
            } else {
                spawn_backend_release(app.handle())
            };
            match result {
                Ok(child) => {
                    let state = app.state::<BackendProcess>();
                    *state.0.lock().unwrap() = Some(child);
                }
                Err(err) => {
                    eprintln!("failed to start backend: {err}");
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
