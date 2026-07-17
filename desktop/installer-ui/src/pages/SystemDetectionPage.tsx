import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { SystemInfo } from '../App';
import '../styles/pages.css';

interface SystemDetectionPageProps {
  onNext: (tier: string, systemInfo: SystemInfo) => void;
  onError: (message: string) => void;
}

export function SystemDetectionPage({ onNext, onError }: SystemDetectionPageProps) {
  const [loading, setLoading] = useState(true);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [detectedTier, setDetectedTier] = useState<string | null>(null);

  useEffect(() => {
    const detectSystem = async () => {
      try {
        const info = (await invoke('detect_system_info')) as SystemInfo;
        setSystemInfo(info);

        const tier = (await invoke('get_recommended_tier')) as string;
        setDetectedTier(tier);
        setLoading(false);
      } catch (error) {
        onError(`Failed to detect system: ${error}`);
      }
    };

    detectSystem();
  }, [onError]);

  const handleContinue = () => {
    if (systemInfo && detectedTier) {
      onNext(detectedTier, systemInfo);
    }
  };

  if (loading) {
    return (
      <div className="page">
        <div className="page-header">
          <h1 className="page-title">System Detection</h1>
        </div>
        <div className="page-content center">
          <div className="loading-spinner"></div>
          <p>Detecting your system hardware...</p>
        </div>
      </div>
    );
  }

  if (!systemInfo || !detectedTier) {
    return (
      <div className="page">
        <div className="page-header">
          <h1 className="page-title">System Detection Failed</h1>
        </div>
        <div className="page-content center">
          <p>Unable to detect system information. Please try again.</p>
        </div>
      </div>
    );
  }

  const tierDescriptions: Record<string, { title: string; description: string; tts: string }> = {
    S: {
      title: 'Tier S - High Performance',
      description: 'High-end NVIDIA GPU (RTX 3070+, RTX 4060+)',
      tts: 'Parler-TTS-Tiny-v1 (high quality)',
    },
    A: {
      title: 'Tier A - Balanced',
      description: 'NVIDIA GPU, Apple Silicon, or strong integrated GPU',
      tts: 'Parler-TTS-Tiny-v1 (high quality)',
    },
    B: {
      title: 'Tier B - Standard',
      description: 'AMD GPU or integrated graphics',
      tts: 'Kokoro (lightweight)',
    },
    C: {
      title: 'Tier C - CPU Only',
      description: 'CPU-only or low resource systems',
      tts: 'Kokoro (lightweight)',
    },
  };

  const tierInfo = tierDescriptions[detectedTier] || tierDescriptions['C'];

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">System Detection</h1>
        <p className="page-subtitle">Your hardware configuration</p>
      </div>

      <div className="page-content">
        <div className="system-info-section">
          <h2>Detected Hardware:</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">OS:</span>
              <span className="info-value">{systemInfo.os}</span>
            </div>
            <div className="info-item">
              <span className="info-label">CPU:</span>
              <span className="info-value">
                {systemInfo.cpu_count}x {systemInfo.cpu_model}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Memory:</span>
              <span className="info-value">{systemInfo.total_memory_gb.toFixed(1)} GB RAM</span>
            </div>
            <div className="info-item">
              <span className="info-label">Disk:</span>
              <span className="info-value">{systemInfo.disk_available_gb.toFixed(1)} GB available</span>
            </div>
            {systemInfo.gpu_info.gpu_name && (
              <div className="info-item">
                <span className="info-label">GPU:</span>
                <span className="info-value">{systemInfo.gpu_info.gpu_name}</span>
              </div>
            )}
          </div>
        </div>

        <div className="recommendation-section">
          <h2>Recommended Configuration:</h2>
          <div className="tier-card">
            <div className="tier-badge" data-tier={detectedTier}>
              Tier {detectedTier}
            </div>
            <h3>{tierInfo.title}</h3>
            <p>{tierInfo.description}</p>
            <p>
              <strong>TTS Engine:</strong> {tierInfo.tts}
            </p>
          </div>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-primary" onClick={handleContinue}>
          Continue →
        </button>
      </div>
    </div>
  );
}
