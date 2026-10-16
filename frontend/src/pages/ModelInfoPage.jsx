/**
 * Model Info Page — FR-006: Model Metadata Card
 * Shows architecture, hyperparameters, training details.
 */

import { useState, useEffect } from 'react';
import { getModelMetadata } from '../services/api';

function InfoRow({ label, value }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '12px 0', borderBottom: '1px solid rgba(255,255,255,0.05)',
    }}>
      <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{label}</span>
      <span style={{ color: 'var(--text-primary)', fontSize: '13px', fontWeight: 500, textAlign: 'right', maxWidth: '60%' }}>{value}</span>
    </div>
  );
}

export default function ModelInfoPage() {
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelMetadata()
      .then(r => setMeta(r.data))
      .catch(() => setError('Model metadata unavailable. Ensure the backend loaded the model successfully.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="page-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
      <div style={{ color: 'var(--text-muted)' }}>Loading model info…</div>
    </div>
  );

  if (error) return (
    <div className="page-body">
      <div className="disclaimer" style={{ borderColor: 'rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.08)' }}>
        <span>⚠️ {error}</span>
      </div>
    </div>
  );

  const hp = meta?.hyperparameters || {};
  const em = meta?.evaluation_metrics || {};

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Model Info</h1>
          <p className="page-subtitle">Architecture, hyperparameters, and training provenance</p>
        </div>
        <div className="status-badge status-badge--healthy">
          <span className="status-dot status-dot--healthy"></span>
          v{meta?.model_version}
        </div>
      </header>

      <div className="page-body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

          {/* Model Card */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Model Overview</div>
            </div>
            <div style={{ padding: '0 20px 20px' }}>
              <InfoRow label="Architecture" value={meta?.model_type} />
              <InfoRow label="Version" value={`v${meta?.model_version}`} />
              <InfoRow label="Training Date" value={meta?.training_date || '—'} />
              <InfoRow label="Dataset" value={meta?.dataset_name} />
              <InfoRow label="Dataset Version" value={meta?.dataset_version} />
              <InfoRow label="Anomaly Threshold" value={meta?.threshold?.toFixed(6)} />
              <InfoRow label="Threshold Method" value={meta?.threshold_method} />
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Hyperparameters</div>
            </div>
            <div style={{ padding: '0 20px 20px' }}>
              <InfoRow label="Window Size" value={`${hp.window_size} timesteps`} />
              <InfoRow label="Hidden Dim" value={hp.hidden_dim} />
              <InfoRow label="LSTM Layers" value={hp.n_layers} />
              <InfoRow label="Dropout" value={hp.dropout} />
              <InfoRow label="Batch Size" value={hp.batch_size} />
              <InfoRow label="Learning Rate" value={hp.learning_rate} />
              <InfoRow label="Optimizer" value={hp.optimizer} />
            </div>
          </div>

          {/* Evaluation Metrics */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Test Set Performance</div>
            </div>
            <div style={{ padding: '0 20px 20px' }}>
              <InfoRow label="F1 Score" value={em.f1 ? `${(em.f1 * 100).toFixed(2)}%` : '—'} />
              <InfoRow label="Precision" value={em.precision ? `${(em.precision * 100).toFixed(2)}%` : '—'} />
              <InfoRow label="Recall" value={em.recall ? `${(em.recall * 100).toFixed(2)}%` : '—'} />
            </div>
          </div>

          {/* Architecture diagram text */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Architecture Overview</div>
            </div>
            <div style={{ padding: '0 20px 20px', fontFamily: 'monospace', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <div style={{ color: '#6366f1', fontWeight: 600, marginBottom: '8px' }}>Encoder</div>
              <div>  Input: (batch, 30, 38)</div>
              <div>  LSTM(in=38, hidden=64, layers=2)</div>
              <div>  Output: hidden state (batch, 64)</div>
              <div style={{ margin: '8px 0', color: '#94a3b8' }}>↓ bottleneck</div>
              <div style={{ color: '#10b981', fontWeight: 600, marginBottom: '8px' }}>Decoder</div>
              <div>  Input: hidden state (batch, 64)</div>
              <div>  Repeat × 30 → (batch, 30, 64)</div>
              <div>  LSTM(in=64, hidden=64, layers=2)</div>
              <div>  Linear(64 → 38)</div>
              <div>  Output: (batch, 30, 38)</div>
              <div style={{ marginTop: '12px', color: '#f59e0b' }}>Loss: MSE Reconstruction Error</div>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
