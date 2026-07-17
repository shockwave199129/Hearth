import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import '../styles/pages.css';

interface InstallPathPageProps {
  onNext: (installPath: string) => void;
  onBack: () => void;
  onError: (message: string) => void;
}

export function InstallPathPage({ onNext, onBack, onError }: InstallPathPageProps) {
  const [installPath, setInstallPath] = useState<string>(getDefaultPath());
  const [validating, setValidating] = useState(false);

  function getDefaultPath(): string {
    const platform = navigator.platform.toLowerCase();
    if (platform.includes('win')) {
      return 'C:\\Program Files\\Hearth';
    } else if (platform.includes('mac')) {
      return '/Applications/Hearth.app';
    } else {
      return `${process.env.HOME || '~'}/.local/opt/hearth`;
    }
  }

  const handleBrowse = async () => {
    try {
      const selected = await open({
        directory: true,
        title: 'Select Installation Directory',
      });
      if (selected) {
        setInstallPath(selected as string);
      }
    } catch (error) {
      onError(`Failed to open directory picker: ${error}`);
    }
  };

  const handleContinue = async () => {
    setValidating(true);
    try {
      const isValid = (await invoke('validate_install_path', {
        path: installPath,
      })) as boolean;

      if (isValid) {
        onNext(installPath);
      } else {
        onError('Invalid installation path');
      }
    } catch (error) {
      onError(`Path validation failed: ${error}`);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Installation Path</h1>
        <p className="page-subtitle">Choose where to install Hearth</p>
      </div>

      <div className="page-content">
        <div className="form-group">
          <label htmlFor="install-path">Installation Directory:</label>
          <div className="path-input-group">
            <input
              id="install-path"
              type="text"
              value={installPath}
              onChange={(e) => setInstallPath(e.target.value)}
              placeholder="Enter installation path"
            />
            <button className="btn btn-secondary" onClick={handleBrowse}>
              Browse
            </button>
          </div>
          <small className="help-text">Requires at least 5 GB of free space</small>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-secondary" onClick={onBack}>
          ← Back
        </button>
        <button
          className="btn btn-primary"
          onClick={handleContinue}
          disabled={validating || !installPath}
        >
          {validating ? 'Validating...' : 'Continue →'}
        </button>
      </div>
    </div>
  );
}
