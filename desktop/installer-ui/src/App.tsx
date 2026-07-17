import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { WelcomePage } from './pages/WelcomePage';
import { SystemDetectionPage } from './pages/SystemDetectionPage';
import { InstallPathPage } from './pages/InstallPathPage';
import { ReviewPage } from './pages/ReviewPage';
import { InstallationPage } from './pages/InstallationPage';
import { CompletionPage } from './pages/CompletionPage';
import { ErrorPage } from './pages/ErrorPage';
import './App.css';

export enum InstallStep {
  Welcome = 'welcome',
  SystemDetection = 'system-detection',
  InstallPath = 'install-path',
  Review = 'review',
  Installation = 'installation',
  Completion = 'completion',
  Error = 'error',
}

export interface SystemInfo {
  os: string;
  arch: string;
  cpu_count: number;
  cpu_model: string;
  total_memory_gb: number;
  available_memory_gb: number;
  gpu_info: {
    has_nvidia: boolean;
    has_amd: boolean;
    has_metal: boolean;
    cuda_version: string | null;
    gpu_name: string | null;
  };
  disk_available_gb: number;
}

export interface InstallConfig {
  tier: string;
  installPath: string;
  systemInfo: SystemInfo | null;
}

export function App() {
  const [currentStep, setCurrentStep] = useState<InstallStep>(InstallStep.Welcome);
  const [config, setConfig] = useState<InstallConfig>({
    tier: '',
    installPath: '',
    systemInfo: null,
  });
  const [error, setError] = useState<string | null>(null);

  const handleNext = (newStep: InstallStep, newConfig?: Partial<InstallConfig>) => {
    if (newConfig) {
      setConfig((prev) => ({ ...prev, ...newConfig }));
    }
    setCurrentStep(newStep);
  };

  const handleError = (message: string) => {
    setError(message);
    setCurrentStep(InstallStep.Error);
  };

  const handleRetry = () => {
    setError(null);
    setCurrentStep(InstallStep.Welcome);
    setConfig({
      tier: '',
      installPath: '',
      systemInfo: null,
    });
  };

  return (
    <div className="installer-app">
      <div className="installer-container">
        {currentStep === InstallStep.Welcome && (
          <WelcomePage onNext={() => handleNext(InstallStep.SystemDetection)} />
        )}

        {currentStep === InstallStep.SystemDetection && (
          <SystemDetectionPage
            onNext={(tier, systemInfo) =>
              handleNext(InstallStep.InstallPath, { tier, systemInfo })
            }
            onError={handleError}
          />
        )}

        {currentStep === InstallStep.InstallPath && (
          <InstallPathPage
            onNext={(installPath) =>
              handleNext(InstallStep.Review, { installPath })
            }
            onBack={() => handleNext(InstallStep.SystemDetection)}
            onError={handleError}
          />
        )}

        {currentStep === InstallStep.Review && (
          <ReviewPage
            config={config}
            onNext={() => handleNext(InstallStep.Installation)}
            onBack={() => handleNext(InstallStep.InstallPath)}
          />
        )}

        {currentStep === InstallStep.Installation && (
          <InstallationPage
            config={config}
            onComplete={() => handleNext(InstallStep.Completion)}
            onError={handleError}
          />
        )}

        {currentStep === InstallStep.Completion && <CompletionPage />}

        {currentStep === InstallStep.Error && (
          <ErrorPage message={error || 'An unknown error occurred'} onRetry={handleRetry} />
        )}
      </div>
    </div>
  );
}
