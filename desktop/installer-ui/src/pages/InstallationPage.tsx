import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { InstallConfig } from '../App';
import '../styles/pages.css';

interface InstallationPageProps {
  config: InstallConfig;
  onComplete: () => void;
  onError: (message: string) => void;
}

export function InstallationPage({ config, onComplete, onError }: InstallationPageProps) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Preparing installation...');

  useEffect(() => {
    const runInstallation = async () => {
      try {
        const result = await invoke('start_installation', {
          install_path: config.installPath,
          hardware_tier: config.tier,
        });

        // Simulate progress updates
        const steps = [
          'Extracting files...',
          'Installing dependencies...',
          'Downloading models...',
          'Configuring system...',
          'Finalizing installation...',
        ];

        for (let i = 0; i < steps.length; i++) {
          setCurrentStep(steps[i]);
          setProgress(((i + 1) / steps.length) * 100);
          await new Promise((resolve) => setTimeout(resolve, 1500));
        }

        onComplete();
      } catch (error) {
        onError(`Installation failed: ${error}`);
      }
    };

    runInstallation();
  }, [config, onComplete, onError]);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Installing Hearth</h1>
      </div>

      <div className="page-content center">
        <div className="progress-section">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
          <p className="progress-text">{Math.round(progress)}%</p>
          <p className="step-text">{currentStep}</p>
        </div>
      </div>
    </div>
  );
}
