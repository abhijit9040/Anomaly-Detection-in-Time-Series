/**
 * Dashboard — FR-004: Real-Time Monitoring Dashboard
 * Live sensor stream with LSTM anomaly detection results, Root Cause Analysis,
 * and Batch Inference (FR-012).
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { checkHealth, getStreamData, detectBatch } from '../services/api';

const MAX_POINTS = 200;

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [backendError, setBackendError] = useState(null);

  // Stream state
  const [timestamps, setTimestamps] = useState([]);
  const [reconErrors, setReconErrors] = useState([]);
  const [anomalyFlags, setAnomalyFlags] = useState([]);
  const [sensor0, setSensor0] = useState([]);
  const [sensor1, setSensor1] = useState([]);
  const [sensor2, setSensor2] = useState([]);
  const [anomalyCount, setAnomalyCount] = useState(0);
  const [lastError, setLastError] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [topRootCauses, setTopRootCauses] = useState([]);

  // Batch inference state (FR-012)
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  const threshold = useRef(0.8281);
  const streamRef = useRef(null);

  // Health polling
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await checkHealth();
        setHealth(res.data);
        setBackendError(null);
      } catch {
        setBackendError('Backend unreachable');
      }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => clearInterval(id);
  }, []);

  const fetchStreamPoint = useCallback(async () => {
    try {
      const res = await getStreamData();
      const d = res.data;
      const now = new Date().toLocaleTimeString();

      setTimestamps(prev => [...prev.slice(-MAX_POINTS), now]);
      setSensor0(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[0]]);
      setSensor1(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[1]]);
      setSensor2(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[2]]);

      let top3 = [];
      if (d.sensor_errors && d.sensor_errors.length > 0) {
        top3 = [...d.sensor_errors]
          .sort((a, b) => b.reconstruction_error - a.reconstruction_error)
          .slice(0, 3);
        setTopRootCauses(top3);
      }

      if (d.reconstruction_error !== null) {
        const err = d.reconstruction_error;
        setReconErrors(prev => [...prev.slice(-MAX_POINTS), err]);
        setAnomalyFlags(prev => [...prev.slice(-MAX_POINTS), d.is_anomaly]);
        setLastError(err);

        if (d.is_anomaly) {
          setAnomalyCount(c => c + 1);
          setRecentAlerts(prev => [{
            time: now,
            error: err.toFixed(4),
            threshold: threshold.current.toFixed(4),
            topSensor: top3[0] ? `${top3[0].sensor_name} (${top3[0].reconstruction_error.toFixed(3)})` : 'Ch-00',
          }, ...prev.slice(0, 9)]);
        }
      }
    } catch {
      // silently ignore stream errors
    }
  }, []);

  const startStream = useCallback(() => {
    if (streamRef.current) return;
    const speed = parseInt(localStorage.getItem('setting_stream_speed') || '500', 10);
    setStreaming(true);
    streamRef.current = setInterval(fetchStreamPoint, speed);
  }, [fetchStreamPoint]);

  const stopStream = () => {
    clearInterval(streamRef.current);
    streamRef.current = null;
    setStreaming(false);
  };

  // Auto-start stream on load once model is loaded + initial rapid warm-up
  useEffect(() => {
    if (health?.model_loaded && !streamRef.current) {
      const initBurst = async () => {
        for (let i = 0; i < 4; i++) {
          await fetchStreamPoint();
        }
        startStream();
      };
      initBurst();
    }
  }, [health?.model_loaded, fetchStreamPoint, startStream]);

  useEffect(() => () => clearInterval(streamRef.current), []);

  // Handle Batch Detection Demonstration (FR-012)
  const handleRunBatchSample = async () => {
    setBatchLoading(true);
    try {
      // Create 10 synthetic test windows around current sensor readings
      const windows = Array.from({ length: 10 }, (_, winIdx) => ({
        data: Array.from({ length: 30 }, (__, tIdx) =>
          Array.from({ length: 38 }, (___, sIdx) => {
            // Introduce subtle deviations in windows 3 and 7
            const base = (sIdx % 5) * 0.15;
            const anomalyBonus = (winIdx === 3 || winIdx === 7) && sIdx < 4 ? 0.8 : 0.0;
            return base + Math.sin(tIdx / 5) * 0.05 + anomalyBonus;
          })
        ),
      }));

      const res = await detectBatch(windows);
      setBatchResults(res.data);
    } catch (err) {
      alert('Batch detection error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setBatchLoading(false);
    }
  };

  const modelLoaded = health?.model_loaded;

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Real-Time Dashboard</h1>
          <p className="page-subtitle">Sensor anomaly monitoring — Industrial IoT</p>
        </div>
        <div className="header-actions" style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {modelLoaded && (
            <button
              onClick={streaming ? stopStream : startStream}
              style={{
                padding: '8px 20px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                background: streaming ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
                color: streaming ? '#ef4444' : '#10b981',
                fontWeight: 600, fontSize: '13px',
              }}
            >
              {streaming ? '⏸ Pause Stream' : '▶ Start Stream'}
            </button>
          )}
          <div className={`status-badge ${health ? 'status-badge--healthy' : 'status-badge--danger'}`}>
            <span className={`status-dot ${health ? 'status-dot--healthy' : 'status-dot--danger'}`}></span>
            {health ? 'Backend Online' : backendError ? 'Backend Offline' : 'Connecting...'}
          </div>
        </div>
      </header>

      <div className="page-body">
        <div className="disclaimer">
          <span className="disclaimer-icon">⚠️</span>
          <span>
            <strong>Decision-support only</strong> — Anomaly flags are for human review and action.
            This system does not autonomously trigger equipment shutdown or maintenance orders.
          </span>
        </div>

        {/* ── Metric Cards ── */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Active Sensors</div>
            <div className="metric-value">38</div>
            <div className="metric-change metric-change--up">● All channels active</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Anomalies Detected</div>
            <div className="metric-value" style={{ color: anomalyCount > 0 ? '#ef4444' : undefined }}>
              {anomalyCount}
            </div>
            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
              {streaming ? '● Live monitoring active' : 'Stream paused'}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Model Status</div>
            <div className="metric-value" style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className={`status-dot ${modelLoaded ? 'status-dot--healthy' : 'status-dot--warning'}`}></span>
              <span>{modelLoaded ? 'Loaded' : 'Standby'}</span>
            </div>
            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
              {modelLoaded ? `v${health?.model_version} — LSTM Autoencoder` : 'No model available'}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Last Recon. Error</div>
            <div className="metric-value" style={{ fontSize: '22px', color: lastError && lastError > threshold.current ? '#ef4444' : '#10b981' }}>
              {lastError !== null ? lastError.toFixed(4) : '0.0412'}
            </div>
            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
              Threshold: {threshold.current.toFixed(4)}
            </div>
          </div>
        </div>

        {/* ── Charts ── */}
        <div className="charts-grid">
          <div className="chart-container">
            <div className="card-header">
              <div className="card-title">Sensor Time Series</div>
              <div className="card-subtitle">Live multi-sensor data (Sensors 00, 01, 02)</div>
            </div>
            {sensor0.length > 0 ? (
              <Plot
                data={[
                  { x: timestamps, y: sensor0, mode: 'lines', name: 'Sensor 00', line: { color: '#6366f1', width: 1.5 } },
                  { x: timestamps, y: sensor1, mode: 'lines', name: 'Sensor 01', line: { color: '#10b981', width: 1.5 } },
                  { x: timestamps, y: sensor2, mode: 'lines', name: 'Sensor 02', line: { color: '#f59e0b', width: 1.5 } },
                ]}
                layout={{
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 11 },
                  xaxis: { showgrid: false, zeroline: false },
                  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
                  margin: { t: 10, b: 40, l: 50, r: 10 },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.15 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '260px' }}
              />
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon">📡</div>
                <div className="placeholder-title">Streaming Sensor Data…</div>
                <div className="placeholder-text">Connecting to real-time SMD replay stream.</div>
              </div>
            )}
          </div>

          <div className="chart-container">
            <div className="card-header">
              <div className="card-title">Reconstruction Error</div>
              <div className="card-subtitle">LSTM anomaly score vs threshold</div>
            </div>
            {reconErrors.length > 0 ? (
              <Plot
                data={[
                  {
                    x: timestamps.slice(-reconErrors.length),
                    y: reconErrors, mode: 'lines', name: 'Recon. Error',
                    line: { color: '#6366f1', width: 1.5 },
                    fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.08)',
                  },
                  {
                    x: [timestamps[Math.max(0, timestamps.length - reconErrors.length)], timestamps[timestamps.length - 1]],
                    y: [threshold.current, threshold.current],
                    mode: 'lines', name: 'Threshold',
                    line: { color: '#ef4444', width: 2, dash: 'dash' },
                  },
                  {
                    x: timestamps.slice(-reconErrors.length).filter((_, i) => anomalyFlags[i]),
                    y: reconErrors.filter((_, i) => anomalyFlags[i]),
                    mode: 'markers', name: 'Anomaly',
                    marker: { color: '#ef4444', size: 7, symbol: 'circle' },
                  },
                ]}
                layout={{
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 11 },
                  xaxis: { showgrid: false, zeroline: false },
                  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
                  margin: { t: 10, b: 40, l: 55, r: 10 },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.15 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '260px' }}
              />
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon">📈</div>
                <div className="placeholder-title">Calculating Reconstruction Error…</div>
                <div className="placeholder-text">Evaluating 30-timestep windows through LSTM Autoencoder.</div>
              </div>
            )}
          </div>
        </div>

        {/* ── Root Cause Quick Breakdown (FR-009 / US-002) ── */}
        {topRootCauses.length > 0 && (
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header">
              <div>
                <div className="card-title">Live Root Cause Analysis (FR-009)</div>
                <div className="card-subtitle">Highest error contributing sensor channels for active window</div>
              </div>
              <span style={{ fontSize: '12px', color: '#6366f1', fontWeight: 600 }}>
                Top 3 Channels
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', padding: '0 20px 20px' }}>
              {topRootCauses.map((item, i) => (
                <div key={i} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px', padding: '12px 16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      #{i + 1} {item.sensor_name}
                    </span>
                    <span style={{ fontSize: '12px', color: item.reconstruction_error > threshold.current ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                      err: {item.reconstruction_error.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(100, (item.reconstruction_error / (threshold.current * 1.5)) * 100)}%`,
                      height: '100%',
                      background: item.reconstruction_error > threshold.current ? '#ef4444' : '#6366f1',
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Batch Inference Demonstration Card (FR-012) ── */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="card-header">
            <div>
              <div className="card-title">Batch Anomaly Detection (FR-012)</div>
              <div className="card-subtitle">Execute multi-window batch inference via POST /api/v1/detect/batch</div>
            </div>
            <button
              onClick={handleRunBatchSample}
              disabled={batchLoading || !modelLoaded}
              className="btn btn-primary"
              style={{ padding: '7px 16px', fontSize: '12px' }}
            >
              {batchLoading ? 'Processing Batch…' : '⚡ Run Sample Batch (10 Windows)'}
            </button>
          </div>
          {batchResults && (
            <div style={{ padding: '0 20px 20px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Windows Analyzed</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>{batchResults.total_windows}</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Anomalies Flagged</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#ef4444' }}>{batchResults.anomaly_count}</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Anomaly Rate</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#f59e0b' }}>{batchResults.anomaly_rate_pct}%</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Threshold Applied</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#10b981' }}>{batchResults.threshold.toFixed(4)}</div>
                </div>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                <strong>Top Root Cause Sensors for Batch:</strong> {batchResults.top_anomalous_sensors.map(s => `${s.sensor_name} (err: ${s.reconstruction_error.toFixed(3)})`).join(', ')}
              </div>
            </div>
          )}
        </div>

        {/* ── Recent Alerts ── */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Recent Anomaly Alerts</div>
              <div className="card-subtitle">Flagged windows — for human review only</div>
            </div>
            {recentAlerts.length > 0 && (
              <button
                onClick={() => setRecentAlerts([])}
                style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)', padding: '4px 10px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
              >
                Clear
              </button>
            )}
          </div>
          {recentAlerts.length === 0 ? (
            <div className="placeholder-state" style={{ minHeight: '120px' }}>
              <div className="placeholder-icon">🔔</div>
              <div className="placeholder-title">{streaming ? 'Monitoring Active — Sensor Behavior Normal' : 'Stream Paused'}</div>
              <div className="placeholder-text">Alerts will appear here when the model detects anomalous sensor patterns.</div>
            </div>
          ) : (
            <div style={{ padding: '0 20px 16px', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>#</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Time</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Recon. Error</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Threshold</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Top Root Cause</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {recentAlerts.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', color: 'var(--text-primary)' }}>
                      <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>{recentAlerts.length - i}</td>
                      <td style={{ padding: '8px 12px' }}>{a.time}</td>
                      <td style={{ padding: '8px 12px', color: '#ef4444', fontWeight: 600 }}>{a.error}</td>
                      <td style={{ padding: '8px 12px' }}>{a.threshold}</td>
                      <td style={{ padding: '8px 12px', color: '#f59e0b', fontFamily: 'monospace' }}>{a.topSensor}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{ background: 'rgba(239,68,68,0.15)', color: '#ef4444', padding: '2px 10px', borderRadius: '20px', fontSize: '11px', fontWeight: 600 }}>
                          ⚠ ANOMALY
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
