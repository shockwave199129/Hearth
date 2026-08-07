use serde::{Deserialize, Serialize};
use sysinfo::{System, Disks, MINIMUM_CPU_LOAD};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub cpu_model: String,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub gpu_info: GpuInfo,
    pub disk_available_gb: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub has_nvidia: bool,
    pub has_amd: bool,
    pub has_metal: bool,
    pub cuda_version: Option<String>,
    pub gpu_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum HardwareTier {
    #[serde(rename = "S")]
    S, // High-end NVIDIA GPU
    #[serde(rename = "A")]
    A, // NVIDIA GPU or Apple Silicon
    #[serde(rename = "B")]
    B, // AMD GPU or integrated GPU
    #[serde(rename = "C")]
    C, // CPU only
}

impl HardwareTier {
    pub fn description(&self) -> &'static str {
        match self {
            HardwareTier::S => "High-end NVIDIA GPU (RTX 3070+, RTX 4060+) - Fastest performance",
            HardwareTier::A => "NVIDIA GPU, Apple Silicon, or strong integrated GPU - Balanced performance",
            HardwareTier::B => "AMD GPU or integrated graphics - Standard performance",
            HardwareTier::C => "CPU only - Slower but still usable",
        }
    }

    pub fn model_size(&self) -> &'static str {
        match self {
            HardwareTier::S | HardwareTier::A => "Full LFM2.5 1.2B model",
            HardwareTier::B | HardwareTier::C => "Optimized LFM2.5 quantized model",
        }
    }

    pub fn tts_engine(&self) -> &'static str {
        match self {
            HardwareTier::S | HardwareTier::A => "Parler-TTS-Tiny-v1 (high quality)",
            HardwareTier::B | HardwareTier::C => "Kokoro (lightweight)",
        }
    }
}

#[tauri::command]
pub fn detect_system_info() -> Result<SystemInfo, String> {
    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.refresh_disks_list();

    let disks = Disks::new_with_refreshed_list();
    let disk_available_gb = disks
        .iter()
        .map(|disk| disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
        .sum();

    let cpu_model = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let gpu_info = detect_gpu_info();

    Ok(SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: sys.cpus().len(),
        cpu_model,
        total_memory_gb: sys.total_memory() as f64 / (1024.0 * 1024.0),
        available_memory_gb: sys.available_memory() as f64 / (1024.0 * 1024.0),
        gpu_info,
        disk_available_gb,
    })
}

fn detect_gpu_info() -> GpuInfo {
    let mut gpu_info = GpuInfo {
        has_nvidia: false,
        has_amd: false,
        has_metal: false,
        cuda_version: None,
        gpu_name: None,
    };

    #[cfg(target_os = "windows")]
    {
        gpu_info = detect_gpu_windows();
    }

    #[cfg(target_os = "macos")]
    {
        gpu_info.has_metal = true;
        gpu_info.gpu_name = Some("Apple Silicon GPU".to_string());
    }

    #[cfg(target_os = "linux")]
    {
        gpu_info = detect_gpu_linux();
    }

    gpu_info
}

#[cfg(target_os = "windows")]
fn detect_gpu_windows() -> GpuInfo {
    use std::process::Command;
    let mut gpu_info = GpuInfo {
        has_nvidia: false,
        has_amd: false,
        has_metal: false,
        cuda_version: None,
        gpu_name: None,
    };

    // Try to detect NVIDIA GPU via nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.trim().split(',').collect();
            if let Some(gpu_name) = lines.first() {
                gpu_info.has_nvidia = true;
                gpu_info.gpu_name = Some(gpu_name.trim().to_string());
                if let Some(version) = lines.get(1) {
                    gpu_info.cuda_version = Some(version.trim().to_string());
                }
            }
        }
    }

    // Check for AMD GPU via rocm-smi
    if let Ok(output) = Command::new("rocm-smi").output() {
        if output.status.success() {
            gpu_info.has_amd = true;
        }
    }

    gpu_info
}

#[cfg(target_os = "linux")]
fn detect_gpu_linux() -> GpuInfo {
    use std::process::Command;
    let mut gpu_info = GpuInfo {
        has_nvidia: false,
        has_amd: false,
        has_metal: false,
        cuda_version: None,
        gpu_name: None,
    };

    // Try to detect NVIDIA GPU
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            gpu_info.has_nvidia = true;
            gpu_info.gpu_name = Some(stdout.trim().to_string());
        }
    }

    // Try to detect AMD GPU via rocm-smi
    if let Ok(output) = Command::new("rocm-smi").output() {
        if output.status.success() {
            gpu_info.has_amd = true;
            gpu_info.gpu_name = Some("AMD GPU (ROCm)".to_string());
        }
    }

    gpu_info
}

#[tauri::command]
pub fn check_disk_space(min_gb: f64) -> Result<bool, String> {
    match detect_system_info() {
        Ok(info) => Ok(info.disk_available_gb >= min_gb),
        Err(e) => Err(e),
    }
}

#[tauri::command]
pub fn get_recommended_tier() -> Result<HardwareTier, String> {
    let info = detect_system_info()?;

    // Tier S: High-end NVIDIA GPU
    if info.gpu_info.has_nvidia && info.total_memory_gb >= 8.0 {
        if let Some(ref name) = info.gpu_info.gpu_name {
            if name.contains("RTX 3070")
                || name.contains("RTX 3080")
                || name.contains("RTX 4060")
                || name.contains("RTX 4070")
                || name.contains("RTX 4080")
            {
                return Ok(HardwareTier::S);
            }
        }
    }

    // Tier A: NVIDIA GPU with 6GB+, Apple Silicon, or good integrated GPU
    if info.gpu_info.has_nvidia && info.total_memory_gb >= 6.0 {
        return Ok(HardwareTier::A);
    }
    if info.gpu_info.has_metal {
        return Ok(HardwareTier::A);
    }
    if info.total_memory_gb >= 8.0 && info.cpu_count >= 4 {
        return Ok(HardwareTier::A);
    }

    // Tier B: AMD GPU or 4GB+ RAM
    if info.gpu_info.has_amd || info.total_memory_gb >= 4.0 {
        return Ok(HardwareTier::B);
    }

    // Tier C: Default (CPU only)
    Ok(HardwareTier::C)
}
