use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstallationProgress {
    pub step: String,
    pub progress: u32,
    pub total_steps: u32,
    pub details: String,
    pub is_complete: bool,
    pub error: Option<String>,
}

static PROGRESS: std::sync::OnceLock<Arc<InstallationProgress>> = std::sync::OnceLock::new();

#[tauri::command]
pub fn validate_install_path(path: String) -> Result<bool, String> {
    let p = Path::new(&path);

    // Check if parent exists
    if let Some(parent) = p.parent() {
        if !parent.exists() {
            return Err(format!("Parent directory does not exist: {:?}", parent));
        }
    }

    // Check write permissions
    if p.exists() {
        // Path exists, check if writable
        match fs::metadata(&p) {
            Ok(_) => Ok(true),
            Err(e) => Err(format!("Cannot write to path: {}", e)),
        }
    } else {
        // Path doesn't exist, check parent directory
        if let Some(parent) = p.parent() {
            match fs::metadata(parent) {
                Ok(_) => Ok(true),
                Err(e) => Err(format!("Cannot write to parent directory: {}", e)),
            }
        } else {
            Err("Invalid installation path".to_string())
        }
    }
}

#[tauri::command]
pub async fn start_installation(
    install_path: String,
    hardware_tier: String,
) -> Result<InstallationProgress, String> {
    // Create installation directory if it doesn't exist
    fs::create_dir_all(&install_path)
        .await
        .map_err(|e| format!("Failed to create installation directory: {}", e))?;

    // TODO: Download and install app files based on hardware_tier
    // This would involve:
    // 1. Downloading the appropriate app bundle
    // 2. Extracting files
    // 3. Creating shortcuts/launchers
    // 4. Running backend setup (model downloads, etc.)

    Ok(InstallationProgress {
        step: "Starting installation...".to_string(),
        progress: 0,
        total_steps: 5,
        details: format!("Installing to: {}", install_path),
        is_complete: false,
        error: None,
    })
}

#[tauri::command]
pub fn get_installation_progress() -> Result<InstallationProgress, String> {
    // Return current installation progress
    Ok(InstallationProgress {
        step: "Installation in progress...".to_string(),
        progress: 50,
        total_steps: 5,
        details: "Downloading files...".to_string(),
        is_complete: false,
        error: None,
    })
}

pub fn get_default_install_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        PathBuf::from(format!(
            "{}\\Hearth",
            std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".to_string())
        ))
    }

    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Applications/Hearth.app")
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(format!("{}/.local/opt/hearth", home))
        } else {
            PathBuf::from("/opt/hearth")
        }
    }
}
