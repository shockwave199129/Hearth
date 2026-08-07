import React from 'react';
import '../styles/pages.css';

export function CompletionPage() {
  const handleLaunchApp = () => {
    // Launch the installed Hearth application
    window.location.href = 'hearth://';
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Installation Complete</h1>
      </div>

      <div className="page-content center">
        <div className="success-icon">✓</div>
        <h2>Hearth has been successfully installed!</h2>
        <p>
          Your emotional-support voice companion is ready to use. On the next launch, the app will
          download and configure models for your hardware tier.
        </p>

        <div className="info-box">
          <h3>Next Steps:</h3>
          <ul>
            <li>Launch Hearth from your applications menu or desktop shortcut</li>
            <li>On first run, the app will detect your hardware and download models</li>
            <li>Complete the onboarding process to create your profile</li>
            <li>Start chatting with your companion!</li>
          </ul>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-primary" onClick={() => window.close()}>
          Close Installer
        </button>
      </div>
    </div>
  );
}
