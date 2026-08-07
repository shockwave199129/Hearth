#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod hardware;
mod installer;
mod ui;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            hardware::detect_system_info,
            hardware::check_disk_space,
            hardware::get_recommended_tier,
            installer::validate_install_path,
            installer::start_installation,
            installer::get_installation_progress,
        ])
        .setup(|app| {
            let main_window = app.get_webview_window("main")
                .expect("no main window");
            
            // Set installer window size based on platform
            #[cfg(target_os = "macos")]
            {
                let _ = main_window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                    width: 700,
                    height: 600,
                }));
            }
            
            #[cfg(target_os = "windows")]
            {
                let _ = main_window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                    width: 720,
                    height: 620,
                }));
            }
            
            #[cfg(target_os = "linux")]
            {
                let _ = main_window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                    width: 700,
                    height: 600,
                }));
            }
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running installer");
}
