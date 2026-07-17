import React from 'react';
import '../styles/pages.css';

interface ErrorPageProps {
  message: string;
  onRetry: () => void;
}

export function ErrorPage({ message, onRetry }: ErrorPageProps) {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Installation Error</h1>
      </div>

      <div className="page-content center">
        <div className="error-icon">⚠</div>
        <h2>Something went wrong</h2>
        <div className="error-message">
          <p>{message}</p>
        </div>
      </div>

      <div className="page-footer">
        <button className="btn btn-primary" onClick={onRetry}>
          Try Again
        </button>
        <button className="btn btn-secondary" onClick={() => window.close()}>
          Close
        </button>
      </div>
    </div>
  );
}
