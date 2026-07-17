import React from 'react';
import '../styles/pages.css';

interface WelcomePageProps {
  onNext: () => void;
}

export function WelcomePage({ onNext }: WelcomePageProps) {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Welcome to Hearth</h1>
        <p className="page-subtitle">Installation Wizard</p>
      </div>

      <div className="page-content">
        <div className="welcome-message">
          <h2>Welcome to Hearth Installation</h2>
          <p>
            Hearth is a privacy-first emotional-support voice companion that runs entirely on your
            machine. This wizard will guide you through the installation process.
          </p>

          <div className="feature-list">
            <h3>This installer will:</h3>
            <ul>
              <li>✓ Detect your system hardware</li>
              <li>✓ Recommend the best configuration for your system</li>
              <li>✓ Help you choose an installation location</li>
              <li>✓ Install and configure Hearth</li>
              <li>✓ Download required models and dependencies</li>
            </ul>
          </div>

          <div className="requirements">
            <h3>System Requirements:</h3>
            <ul>
              <li>5 GB of free disk space</li>
              <li>4 GB RAM minimum (8 GB recommended)</li>
              <li>Internet connection for initial setup</li>
              <li>Windows 10+, macOS 10.15+, or modern Linux</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-primary" onClick={onNext}>
          Next →
        </button>
      </div>
    </div>
  );
}
