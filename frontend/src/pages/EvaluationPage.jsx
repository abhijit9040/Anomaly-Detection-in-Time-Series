/**
 * Evaluation Page — FR-005: Model Evaluation Report
 * Shows F1, precision, recall, confusion matrix, and comparison vs baseline.
 */

import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { getEvaluationReport } from '../services/api';

function MetricGauge({ label, value, color }) {
  return (
    <div style={{
      background: 'var(--surface-1)', borderRadius: '16px', padding: '24px',
      border: '1px solid rgba(255,255,255,0.06)', textAlign: 'center',
    }}>
      <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '12px', fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: '42px', fontWeight: 700, color, letterSpacing: '-1px' }}>
        {(value * 100).toFixed(1)}<span style={{ fontSize: '22px' }}>%</span>
      </div>
    </div>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getEvaluationReport()
      .then(r => setData(r.data))
      .catch(() => setError('Could not load evaluation results. Run python -m src.evaluate in the training directory first.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="page-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '16px' }}>Loading evaluation results…</div>
    </div>
  );

  if (error) return (
    <div className="page-body">
      <div className="disclaimer" style={{ borderColor: 'rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.08)' }}>
        <span>⚠️ {error}</span>
      </div>
    </div>
  );

  const lstm = data?.lstm_autoencoder;
  const baseline = data?.baseline;
  const improvement = data?.improvement;

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Model Evaluation</h1>
          <p className="page-subtitle">Test set results — held-out, never seen during training</p>
        </div>
        <div className="status-badge status-badge--healthy">
          <span className="status-dot status-dot--healthy"></span>
          Results Loaded
        </div>
      </header>

      <div className="page-body">
        <div className="disclaimer">
          <span className="disclaimer-icon">⚠️</span>
          <span>
            <strong>Decision-support only</strong> — These metrics are for model assessment.
            Threshold was selected on the validation set. Test set was never used during training.
          </span>
        </div>

        {/* ── LSTM Metrics ── */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="card-header">
            <div>
              <div className="card-title">LSTM Autoencoder — Test Set Performance</div>
              <div className="card-subtitle">
                Machine: {lstm?.machine_id} · Threshold: {lstm?.threshold?.toFixed(4)} ({lstm?.threshold_strategy})
              </div>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', padding: '0 20px 20px' }}>
            <MetricGauge label="F1 Score" value={lstm?.f1 || 0} color="#6366f1" />
            <MetricGauge label="Precision" value={lstm?.precision || 0} color="#10b981" />
            <MetricGauge label="Recall" value={lstm?.recall || 0} color="#f59e0b" />
          </div>
        </div>

        {/* ── Baseline vs LSTM comparison ── */}
        {baseline && (
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header">
              <div>
                <div className="card-title">LSTM vs Statistical Baseline</div>
                <div className="card-subtitle">Moving Average Z-Score Baseline · F1 improvement: +{improvement?.f1_improvement_pct}%</div>
              </div>
              <span style={{
                background: 'rgba(16,185,129,0.15)', color: '#10b981',
                padding: '4px 14px', borderRadius: '20px', fontSize: '13px', fontWeight: 600,
              }}>
                +{(improvement?.f1_delta * 100).toFixed(1)}% F1
              </span>
            </div>
            <div style={{ padding: '0 20px 20px' }}>
              <Plot
                data={[
                  {
                    x: ['F1 Score', 'Precision', 'Recall'],
                    y: [baseline.f1, baseline.precision, baseline.recall],
                    type: 'bar', name: 'Baseline (Z-Score)',
                    marker: { color: 'rgba(100,116,139,0.6)' },
                  },
                  {
                    x: ['F1 Score', 'Precision', 'Recall'],
                    y: [lstm.f1, lstm.precision, lstm.recall],
                    type: 'bar', name: 'LSTM Autoencoder',
                    marker: { color: '#6366f1' },
                  },
                ]}
                layout={{
                  barmode: 'group',
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 12 },
                  xaxis: { showgrid: false },
                  yaxis: { range: [0, 1.1], gridcolor: 'rgba(255,255,255,0.05)', tickformat: '.0%' },
                  margin: { t: 10, b: 40, l: 55, r: 10 },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.2 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '280px' }}
              />
            </div>
          </div>
        )}

        {/* ── Detailed stats table ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Detailed Results Summary</div>
          </div>
          <div style={{ padding: '0 20px 20px', overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                  <th style={{ padding: '12px 16px', textAlign: 'left' }}>Method</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>F1 Score</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Precision</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Recall</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left' }}>Threshold</th>
                </tr>
              </thead>
              <tbody>
                {baseline && (
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', color: 'var(--text-primary)' }}>
                    <td style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Moving Avg Z-Score (Baseline)</td>
                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{(baseline.f1 * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{(baseline.precision * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{(baseline.recall * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Dynamic</td>
                  </tr>
                )}
                <tr style={{ color: 'var(--text-primary)', background: 'rgba(99,102,241,0.05)' }}>
                  <td style={{ padding: '12px 16px', color: '#6366f1', fontWeight: 600 }}>LSTM Autoencoder ★</td>
                  <td style={{ padding: '12px 16px', textAlign: 'right', color: '#6366f1', fontWeight: 700 }}>{(lstm.f1 * 100).toFixed(2)}%</td>
                  <td style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 600 }}>{(lstm.precision * 100).toFixed(2)}%</td>
                  <td style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 600 }}>{(lstm.recall * 100).toFixed(2)}%</td>
                  <td style={{ padding: '12px 16px' }}>{lstm.threshold?.toFixed(4)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}
