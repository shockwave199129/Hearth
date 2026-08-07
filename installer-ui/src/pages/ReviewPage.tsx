import React from 'react';
import { InstallConfig } from '../App';
import '../styles/pages.css';

interface ReviewPageProps {
  config: InstallConfig;
  onNext: () => void;
  onBack: () => void;
}

export function ReviewPage({ config, onNext, onBack }: ReviewPageProps) {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Review Installation</h1>
        <p className="page-subtitle">Verify your settings before installing</p>
      </div>

      <div className="page-content">
        <div className="review-section">
          <h3>Installation Settings:</h3>
          <div className="review-item">
            <span className="review-label">Hardware Tier:</span>
            <span className="review-value">Tier {config.tier}</span>
          </div>
          <div className="review-item">
            <span className="review-label">Installation Path:</span>
            <span className="review-value">{config.installPath}</span>
          </div>
          {config.systemInfo && (
            <>
              <div className="review-item">
                <span className="review-label">System:</span>
                <span className="review-value">
                  {config.systemInfo.os} ({config.systemInfo.arch})
                </span>
              </div>
              <div className="review-item">
                <span className="review-label">Available Disk:</span>
                <span className="review-value">
                  {config.systemInfo.disk_available_gb.toFixed(1)} GB
                </span>
              </div>
            </>
          )}
        </div>

        <div className="info-box">
          <p>
            <strong>Note:</strong> The installation will download models and dependencies on first
            launch. This may take 10-30 minutes depending on your internet connection and hardware.
          </p>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-secondary" onClick={onBack}>
          ← Back
        </button>
        <button className="btn btn-primary" onClick={onNext}>
          Install →
        </button>
      </div>
    </div>
  );
}
