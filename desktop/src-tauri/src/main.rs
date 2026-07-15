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

use tauri::Manager;

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

    cmd.spawn()
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
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Hearth");
}
