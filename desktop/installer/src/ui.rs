// UI module for shared UI utilities
// This can be extended for native dialogs, notifications, etc.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DialogOptions {
    pub title: String,
    pub message: String,
}

pub fn show_welcome_message() -> String {
    "Welcome to Hearth Installation Wizard

This wizard will guide you through the installation process. It will:
1. Detect your system hardware
2. Recommend the best configuration
3. Choose an installation location
4. Install and configure Hearth

Click Next to continue."
        .to_string()
}

pub fn get_installation_requirements() -> String {
    "Hearth requires:

• 5 GB of free disk space
• Python 3.11+ (for development)
• 4 GB RAM minimum (8 GB recommended)
• Internet connection for initial setup

Please ensure you have sufficient resources before proceeding."
        .to_string()
}
