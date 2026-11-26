/**
 * Dashboard — FR-004: Real-Time Monitoring Dashboard
 * Phase 1: Status hero strip, header cleanup, chart polish, Lucide icons.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import {
  Activity, Zap, AlertTriangle, Cpu, Radio, ChevronDown,
  Maximize2, Pause, Play, RotateCcw, FlaskConical, Info, X
} from 'lucide-react';
import { checkHealth, getStreamData, detectBatch, getDemoWindows, jumpToAnomaly, injectAnomaly, resetStream, getModelMetadata } from '../services/api';
import { advanceAlertState, initialAlertState } from '../utils/alertEvents';
import { playAnomalyAlert, unlockAudio } from '../utils/audioAlert';
import ChartModal from '../components/ChartModal';
import ChannelDetailModal from '../components/ChannelDetailModal';

const MAX_POINTS = 300;

// ── Helpers ──────────────────────────────────────────────────────────────────

function computeSystemStatus(lastError, threshold) {
  if (!lastError || !threshold) return 'normal';
  if (lastError > threshold) return 'anomaly';
  if (lastError > threshold * 0.9) return 'warning';
  return 'normal';
}

function pctDiff(error, threshold) {
  if (!error || !threshold) return null;
  return ((error - threshold) / threshold) * 100;
}

/** Sparkline using a tiny inline SVG (no extra deps) */
function Sparkline({ values, color }) {
  if (!values || values.length < 2) return null;
  const w = 80, h = 24;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}



/** Slim dismissible disclaimer bar */
function DisclaimerBar({ onDismiss }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.2)',
      borderRadius: '8px', padding: '8px 14px', marginBottom: '16px', gap: '8px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#f59e0b' }}>
        <Info size={14} style={{ flexShrink: 0 }} />
        <span><strong>Decision-support only</strong> — Anomaly flags are for human review. This system does not autonomously trigger equipment actions.</span>
      </div>
      <button
        onClick={onDismiss}
        aria-label="Dismiss notice"
        style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px', borderRadius: '4px', display: 'flex', alignItems: 'center' }}
        onMouseEnter={e => e.currentTarget.style.color = '#f59e0b'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
      >
        <X size={14} />
      </button>
    </div>
  );
}

/** Demo tools dropdown */
function DemoDropdown({ onInject, onReset, loading }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(o => !o)}
        disabled={loading}
        style={{
          display: 'flex', alignItems: 'center', gap: '5px',
          padding: '7px 12px', borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.1)',
          background: 'rgba(255,255,255,0.04)', color: 'var(--text-muted)',
          cursor: 'pointer', fontWeight: 500, fontSize: '12px',
        }}
      >
        <FlaskConical size={13} /> Demo tools <ChevronDown size={12} />
      </button>
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', right: 0, zIndex: 100,
          background: '#1a2332', border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '10px', overflow: 'hidden', minWidth: '180px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
        }}>
          <button
            onClick={() => { onInject(); setOpen(false); }}
            disabled={loading}
            style={{ width: '100%', textAlign: 'left', padding: '10px 14px', background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(239,68,68,0.1)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <Zap size={13} style={{ color: '#ef4444' }} /> Inject Sensor Spike
          </button>
          <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }} />
          <button
            onClick={() => { onReset(); setOpen(false); }}
            disabled={loading}
            style={{ width: '100%', textAlign: 'left', padding: '10px 14px', background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <RotateCcw size={13} style={{ color: 'var(--text-muted)' }} /> Reset Stream (t=0)
          </button>
        </div>
      )}
    </div>
  );
}

// ── Shared chart layout defaults ──────────────────────────────────────────────
const CHART_LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#94a3b8', size: 11, family: 'Inter, sans-serif' },
  xaxis: {
    showgrid: false,
    zeroline: false,
    automargin: true,
    nticks: 5,
    tickangle: 0,
    tickfont: { size: 10 },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.05)',
    zeroline: false,
    automargin: true,
    tickfont: { size: 10 },
  },
  margin: { t: 10, b: 55, l: 52, r: 12 },
  showlegend: true,
  legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.22, yanchor: 'top', font: { size: 10 } },
  hoverlabel: { bgcolor: '#1a2332', bordercolor: 'rgba(255,255,255,0.15)', font: { size: 11 } },
};

// ── Main component ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [backendError, setBackendError] = useState(null);

  // Stream state
  const [timestamps, setTimestamps] = useState([]);
  const [reconErrors, setReconErrors] = useState([]);
  const [smoothedErrors, setSmoothedErrors] = useState([]);
  const [anomalyFlags, setAnomalyFlags] = useState([]);
  const [sensor0, setSensor0] = useState([]);
  const [sensor1, setSensor1] = useState([]);
  const [sensor2, setSensor2] = useState([]);
  const [sensorValues, setSensorValues] = useState({});
  const [channelErrors, setChannelErrors] = useState({});
  const [anomalyCount, setAnomalyCount] = useState(0);
  const [lastError, setLastError] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [topRootCauses, setTopRootCauses] = useState([]);
  const [currentTimestep, setCurrentTimestep] = useState(null);
  const [streamNotice, setStreamNotice] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [lastAnomalyTime, setLastAnomalyTime] = useState(null);
  const [lastAnomalyAgo, setLastAnomalyAgo] = useState(null);

  // UI state
  const [disclaimerDismissed, setDisclaimerDismissed] = useState(
    () => localStorage.getItem('disclaimer_dismissed') === '1'
  );
  const [maximizedChart, setMaximizedChart] = useState(null);
  const [detailChannel, setDetailChannel] = useState(null);

  // Batch inference
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [meta, setMeta] = useState(null);
  const [lastStreamPoint, setLastStreamPoint] = useState(null);

  const threshold = useRef(0.5);
  const streamRef = useRef(null);
  const alertStateRef = useRef(initialAlertState);

  // "Xs ago" updater
  useEffect(() => {
    const id = setInterval(() => {
      if (lastAnomalyTime) {
        setLastAnomalyAgo(Math.round((Date.now() - lastAnomalyTime) / 1000));
      }
    }, 1000);
    return () => clearInterval(id);
  }, [lastAnomalyTime]);

  // Health & model metadata polling
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
    getModelMetadata()
      .then(res => {
        setMeta(res.data);
        if (res.data?.threshold) threshold.current = res.data.threshold;
      })
      .catch(() => {});
    const id = setInterval(poll, 10000);
    return () => clearInterval(id);
  }, []);

  const fetchStreamPoint = useCallback(async () => {
    try {
      const res = await getStreamData();
      const d = res.data;
      setLastStreamPoint(d);
      const now = new Date().toLocaleTimeString();

      setTimestamps(prev => [...prev.slice(-MAX_POINTS), now]);
      setSensor0(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[0]]);
      setSensor1(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[1]]);
      setSensor2(prev => [...prev.slice(-MAX_POINTS), d.sensor_values[2]]);

      if (d.sensor_values) {
        setSensorValues(prev => {
          const next = { ...prev };
          d.sensor_values.forEach((val, idx) => {
            const arr = next[idx] || [];
            next[idx] = [...arr.slice(-MAX_POINTS), val];
          });
          return next;
        });
      }

      if (d.sensor_errors) {
        setChannelErrors(prev => {
          const next = { ...prev };
          d.sensor_errors.forEach(errObj => {
            const arr = next[errObj.sensor_index] || [];
            next[errObj.sensor_index] = [...arr.slice(-MAX_POINTS), errObj.reconstruction_error];
          });
          return next;
        });
      }

      if (d.timestep !== undefined && d.timestep !== null) setCurrentTimestep(d.timestep);

      let top3 = [];
      if (d.sensor_errors?.length > 0) {
        top3 = [...d.sensor_errors].sort((a, b) => b.reconstruction_error - a.reconstruction_error).slice(0, 3);
        setTopRootCauses(top3);
      }

      if (d.reconstruction_error != null) {
        const err = d.reconstruction_error;
        setReconErrors(prev => [...prev.slice(-MAX_POINTS), err]);
        setAnomalyFlags(prev => [...prev.slice(-MAX_POINTS), d.is_anomaly]);
        setLastError(err);

        const smoothed = d.smoothed_error ?? err;
        setSmoothedErrors(prev => [...prev.slice(-MAX_POINTS), smoothed]);

        const step = advanceAlertState(alertStateRef.current, d.is_anomaly);
        alertStateRef.current = step.state;
        const topSensor = top3[0]
          ? `${top3[0].sensor_name} (${top3[0].reconstruction_error.toFixed(3)})`
          : 'Ch-00';

        if (step.isNewEvent) {
          setAnomalyCount(c => c + 1);
          setLastAnomalyTime(Date.now());
          setLastAnomalyAgo(0);
          playAnomalyAlert();
          setRecentAlerts(prev => [{
            time: now, error: err.toFixed(4), errorNum: err,
            threshold: threshold.current.toFixed(4), topSensor,
          }, ...prev.slice(0, 9)]);
        } else if (d.is_anomaly) {
          setLastAnomalyAgo(0);
          setRecentAlerts(prev =>
            prev.length && err > prev[0].errorNum
              ? [{ ...prev[0], error: err.toFixed(4), errorNum: err, topSensor }, ...prev.slice(1)]
              : prev
          );
        }
      }
    } catch { /* silently ignore */ }
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

  useEffect(() => {
    if (health?.model_loaded && !streamRef.current) {
      const initBurst = async () => {
        for (let i = 0; i < 4; i++) await fetchStreamPoint();
        startStream();
      };
      initBurst();
    }
  }, [health?.model_loaded, fetchStreamPoint, startStream]);

  useEffect(() => () => clearInterval(streamRef.current), []);

  // Batch
  const handleRunBatchSample = async () => {
    setBatchLoading(true);
    try {
      const demoRes = await getDemoWindows(10);
      const demoWindows = demoRes.data.windows;
      const payload = demoWindows.map(w => ({ data: w.data }));
      const res = await detectBatch(payload);
      const resultsWithGt = res.data.results.map((r, idx) => ({
        ...r,
        ground_truth: demoWindows[idx]?.ground_truth_label ?? 0,
        start_timestep: demoWindows[idx]?.start_timestep,
        end_timestep: demoWindows[idx]?.end_timestep,
      }));
      const trueAnomalyAgreement = resultsWithGt.filter(r => r.ground_truth === 1 && r.is_anomaly).length;
      const totalTrueAnomalies = resultsWithGt.filter(r => r.ground_truth === 1).length;
      setBatchResults({ ...res.data, results: resultsWithGt, true_anomaly_agreement: trueAnomalyAgreement, total_true_anomalies: totalTrueAnomalies });
    } catch (err) {
      alert('Batch detection error: ' + (err.response?.data?.detail?.message || err.response?.data?.detail || err.message));
    } finally {
      setBatchLoading(false);
    }
  };

  const handleJumpToAnomaly = async () => {
    unlockAudio();
    setActionLoading(true);
    try {
      const res = await jumpToAnomaly();
      alertStateRef.current = initialAlertState;
      setStreamNotice(`Jumped to timestep ${res.data.current_index} — Real SMD anomaly incoming in ~7 seconds!`);
      setTimeout(() => setStreamNotice(null), 6000);
      if (!streaming && health?.model_loaded) startStream();
    } catch (err) {
      alert('Failed to jump to anomaly: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionLoading(false);
    }
  };

  const handleInjectAnomaly = async () => {
    unlockAudio();
    setActionLoading(true);
    try {
      await injectAnomaly(15);
      setStreamNotice('Injected sensor spike across channels 00, 01, 02 — Anomaly incoming!');
      setTimeout(() => setStreamNotice(null), 6000);
      if (!streaming && health?.model_loaded) startStream();
    } catch (err) {
      alert('Failed to inject anomaly: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionLoading(false);
    }
  };

  const handleResetStream = async () => {
    setActionLoading(true);
    try {
      await resetStream();
      alertStateRef.current = initialAlertState;
      setRecentAlerts([]);
      setAnomalyCount(0);
      setStreamNotice('Stream reset to normal baseline (Timestep 0).');
      setTimeout(() => setStreamNotice(null), 4000);
      if (!streaming && health?.model_loaded) startStream();
    } catch (err) {
      alert('Failed to reset stream: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionLoading(false);
    }
  };

  const dismissDisclaimer = () => {
    setDisclaimerDismissed(true);
    localStorage.setItem('disclaimer_dismissed', '1');
  };

  const modelLoaded = health?.model_loaded;
  const systemStatus = computeSystemStatus(lastError, threshold.current);
  const pct = pctDiff(lastError, threshold.current);
  const pctLabel = pct === null ? '' : pct >= 0 ? `+${pct.toFixed(1)}% over threshold` : `${Math.abs(pct).toFixed(1)}% under threshold`;

  // ── Sensor time series plot data ────────────────────────────────────────
  const sensorPlotData = [
    { x: timestamps, y: sensor0, mode: 'lines', name: 'Sensor 00', line: { color: '#6366f1', width: 1.5 } },
    { x: timestamps, y: sensor1, mode: 'lines', name: 'Sensor 01', line: { color: '#10b981', width: 1.5 } },
    { x: timestamps, y: sensor2, mode: 'lines', name: 'Sensor 02', line: { color: '#f59e0b', width: 1.5 } },
  ];

  // ── Recon error plot data ────────────────────────────────────────────────
  const reconXslice = timestamps.slice(-reconErrors.length);
  const reconPlotData = [
    {
      x: reconXslice, y: reconErrors, mode: 'lines', name: 'Raw Error',
      line: { color: 'rgba(99,102,241,0.4)', width: 1.2 },
    },
    {
      x: timestamps.slice(-smoothedErrors.length), y: smoothedErrors, mode: 'lines', name: 'Smoothed (k=5)',
      line: { color: '#6366f1', width: 2.2 },
      fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.08)',
    },
    {
      x: [reconXslice[0] ?? timestamps[0], timestamps[timestamps.length - 1]],
      y: [threshold.current, threshold.current],
      mode: 'lines', name: 'Threshold',
      line: { color: '#ef4444', width: 2, dash: 'dash' },
    },
    {
      x: reconXslice.filter((_, i) => anomalyFlags[i]),
      y: reconErrors.filter((_, i) => anomalyFlags[i]),
      mode: 'markers', name: 'Anomaly',
      marker: { color: '#ef4444', size: 7, symbol: 'circle' },
    },
  ];

  return (
    <>
      {/* ── Header ── */}
      <header className="page-header">
        <div>
          <h1 className="page-title">Real-Time Dashboard</h1>
          <p className="page-subtitle">Sensor anomaly monitoring — Industrial IoT</p>
        </div>
        <div className="header-actions">
          {modelLoaded && (
            <>
              <button
                onClick={handleJumpToAnomaly}
                disabled={actionLoading}
                title="Jump replay to 15 steps before the real anomaly event in SMD test dataset"
                className="hdr-btn hdr-btn--primary"
              >
                <Zap size={13} /> Jump to Anomaly
              </button>
              <button
                onClick={streaming ? stopStream : startStream}
                className={streaming ? 'hdr-btn hdr-btn--pause' : 'hdr-btn hdr-btn--resume'}
              >
                {streaming ? <><Pause size={13} /> Pause</> : <><Play size={13} /> Resume</>}
              </button>
              <DemoDropdown onInject={handleInjectAnomaly} onReset={handleResetStream} loading={actionLoading} />
            </>
          )}
          <div className={`status-badge ${health ? 'status-badge--healthy' : 'status-badge--danger'}`}>
            <span className={`status-dot ${health ? 'status-dot--healthy' : 'status-dot--danger'}`} />
            {health ? 'Backend Online' : backendError ? 'Backend Offline' : 'Connecting…'}
          </div>
        </div>
      </header>

      <div className="page-body">
        {/* ── Slim disclaimer ── */}
        {!disclaimerDismissed && <DisclaimerBar onDismiss={dismissDisclaimer} />}

        {/* ── Stream notice ── */}
        {streamNotice && (
          <div className="stream-notice">
            <span>{streamNotice}</span>
            <button onClick={() => setStreamNotice(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* ── Status Hero Strip (Row 1) ── */}
        <div className="metrics-grid" style={{ marginBottom: '20px' }}>
          {/* Card 1: System Status Hero Badge */}
          <div
            className="metric-card"
            style={{
              background: systemStatus === 'anomaly'
                ? 'rgba(239, 68, 68, 0.08)'
                : systemStatus === 'warning'
                ? 'rgba(245, 158, 11, 0.08)'
                : 'var(--bg-card)',
              borderColor: systemStatus === 'anomaly'
                ? 'rgba(239, 68, 68, 0.4)'
                : systemStatus === 'warning'
                ? 'rgba(245, 158, 11, 0.35)'
                : 'var(--border-color)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '5px', margin: 0 }}>
                <Activity size={12} /> System Status
              </div>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '5px',
                fontSize: '11px', fontWeight: 600,
                color: streaming ? '#10b981' : 'var(--text-muted)',
              }}>
                <span style={{
                  width: '6px', height: '6px', borderRadius: '50%',
                  background: streaming ? '#10b981' : 'var(--text-muted)',
                  boxShadow: streaming ? '0 0 6px #10b981' : 'none',
                  animation: streaming ? 'pulse 2s infinite' : 'none',
                }} />
                {streaming ? 'LIVE' : 'PAUSED'}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
              <div style={{
                fontSize: '22px', fontWeight: 800,
                color: systemStatus === 'anomaly' ? '#ef4444' : systemStatus === 'warning' ? '#f59e0b' : '#10b981',
                display: 'flex', alignItems: 'center', gap: '6px',
              }}>
                {systemStatus === 'anomaly' && <Zap size={18} style={{ animation: 'pulse 1s infinite' }} />}
                {systemStatus === 'warning' && <AlertTriangle size={18} />}
                {systemStatus === 'normal' && <Activity size={18} />}
                <span>{systemStatus === 'anomaly' ? 'Anomaly' : systemStatus === 'warning' ? 'Warning' : 'Normal'}</span>
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                err: {lastError !== null && lastError !== undefined ? lastError.toFixed(4) : '—'}
              </div>
            </div>

            <div className="metric-change" style={{ color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '4px', marginTop: '6px' }}>
              <span>Threshold: <strong style={{ color: '#ef4444' }}>{threshold.current.toFixed(4)}</strong></span>
              {pctLabel && (
                <span style={{ color: systemStatus === 'anomaly' ? '#ef4444' : systemStatus === 'warning' ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                  {pctLabel}
                </span>
              )}
            </div>
          </div>

          {/* Card 2: Anomaly Events */}
          <div className="metric-card">
            <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <AlertTriangle size={12} /> Anomaly Events
            </div>
            <div className="metric-value" style={{
              fontSize: '28px',
              color: anomalyCount > 0 ? '#ef4444' : undefined,
              WebkitTextFillColor: anomalyCount > 0 ? '#ef4444' : undefined,
            }}>
              {anomalyCount}
            </div>
            <div className="metric-change" style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
              {lastAnomalyAgo !== null ? `Last event: ${lastAnomalyAgo}s ago` : streaming ? 'Monitoring active' : 'Stream paused'}
            </div>
          </div>

          {/* Card 3: Active Channels */}
          <div className="metric-card">
            <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Radio size={12} /> Active Channels
            </div>
            <div className="metric-value" style={{ fontSize: '28px' }}>
              {meta?.n_features ?? lastStreamPoint?.sensor_values?.length ?? 38}
            </div>
            <div className="metric-change metric-change--up">● channels monitored</div>
          </div>

          {/* Card 4: Model (compact with sparkline) */}
          <div className="metric-card">
            <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Cpu size={12} /> Model
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '4px 0 2px' }}>
              <span className={`status-dot ${modelLoaded ? 'status-dot--healthy' : 'status-dot--warning'}`} />
              <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
                {modelLoaded ? 'Loaded' : 'Standby'}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                v{health?.model_version ?? '1.0.0'}
              </span>
            </div>
            {reconErrors.length > 2 ? (
              <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkline
                  values={reconErrors.slice(-40)}
                  color={systemStatus === 'anomaly' ? '#ef4444' : systemStatus === 'warning' ? '#f59e0b' : '#10b981'}
                />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                  {lastError?.toFixed(4)}
                </span>
              </div>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                LSTM Autoencoder
              </div>
            )}
          </div>
        </div>

        {/* ── Charts Grid ── */}
        <div className="charts-grid" style={{ marginBottom: '20px' }}>
          {/* Sensor Time Series */}
          <div
            className="chart-container chart-card-clickable"
            onClick={() => setMaximizedChart('sensor_ts')}
            title="Click to maximize"
          >
            <div className="card-header">
              <div>
                <div className="card-title">Sensor Time Series</div>
                <div className="card-subtitle">Live multi-sensor data (Sensors 00, 01, 02)</div>
              </div>
              <button
                type="button"
                className="chart-expand-btn"
                onClick={e => { e.stopPropagation(); setMaximizedChart('sensor_ts'); }}
                title="Maximize"
              >
                <Maximize2 size={12} /> Expand
              </button>
            </div>
            {sensor0.length > 0 ? (
              <Plot
                data={sensorPlotData}
                layout={{ ...CHART_LAYOUT_BASE }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '260px' }}
              />
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon"><Radio size={40} style={{ opacity: 0.4 }} /></div>
                <div className="placeholder-title">Streaming Sensor Data…</div>
                <div className="placeholder-text">Connecting to real-time SMD replay stream.</div>
              </div>
            )}
          </div>

          {/* Reconstruction Error */}
          <div
            className="chart-container chart-card-clickable"
            onClick={() => setMaximizedChart('recon_error')}
            title="Click to maximize"
          >
            <div className="card-header">
              <div>
                <div className="card-title">Reconstruction Error</div>
                <div className="card-subtitle">Raw & smoothed (k=5) vs threshold</div>
              </div>
              <button
                type="button"
                className="chart-expand-btn"
                onClick={e => { e.stopPropagation(); setMaximizedChart('recon_error'); }}
                title="Maximize"
              >
                <Maximize2 size={12} /> Expand
              </button>
            </div>
            {reconErrors.length > 0 ? (
              <Plot
                data={reconPlotData}
                layout={{ ...CHART_LAYOUT_BASE }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '260px' }}
              />
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon"><Activity size={40} style={{ opacity: 0.4 }} /></div>
                <div className="placeholder-title">Calculating Reconstruction Error…</div>
                <div className="placeholder-text">Evaluating 30-timestep windows through LSTM Autoencoder.</div>
              </div>
            )}
          </div>
        </div>

        {/* ── Root Cause Quick Breakdown ── */}
        {topRootCauses.length > 0 && (
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header">
              <div>
                <div className="card-title">Live Root Cause Analysis</div>
                <div className="card-subtitle">Highest error contributing sensor channels for active window</div>
              </div>
              <span style={{ fontSize: '12px', color: '#6366f1', fontWeight: 600 }}>
                Top 3 Channels (Click to inspect)
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', padding: '0 20px 20px' }}>
              {topRootCauses.map((item, i) => (
                <div
                  key={i}
                  onClick={() => setDetailChannel(item.sensor_index)}
                  title={`Click to view deep dive for ${item.sensor_name}`}
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '10px', padding: '12px 16px',
                    cursor: 'pointer', transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(99,102,241,0.5)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'; e.currentTarget.style.transform = 'translateY(0)'; }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      #{i + 1} {item.sensor_name}
                    </span>
                    <span style={{ fontSize: '12px', color: item.reconstruction_error > threshold.current ? '#ef4444' : '#10b981', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      {item.reconstruction_error.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(100, (item.reconstruction_error / (threshold.current * 1.5)) * 100)}%`,
                      height: '100%',
                      background: item.reconstruction_error > threshold.current ? '#ef4444' : '#6366f1',
                      borderRadius: '3px',
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Batch Inference Card ── */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="card-header">
            <div>
              <div className="card-title">Batch Scoring on Real Test Windows</div>
              <div className="card-subtitle">Multi-window batch inference on real SMD test dataset via POST /api/v1/detect/batch</div>
            </div>
            <button
              onClick={handleRunBatchSample}
              disabled={batchLoading || !modelLoaded}
              className="btn btn-primary"
              style={{ padding: '7px 16px', fontSize: '12px' }}
            >
              {batchLoading ? 'Processing…' : 'Run Real Test Batch (10 Windows)'}
            </button>
          </div>
          {batchResults && (
            <div style={{ padding: '0 20px 20px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '12px' }}>
                {[
                  { label: 'Scored Windows', value: batchResults.total_windows, color: 'var(--text-primary)' },
                  { label: 'Flagged Windows', value: batchResults.anomaly_count, color: '#ef4444' },
                  { label: 'True Anomaly Agreement', value: `${batchResults.true_anomaly_agreement ?? '—'} / ${batchResults.total_true_anomalies ?? 5}`, color: '#10b981' },
                  { label: 'Threshold Applied', value: batchResults.threshold.toFixed(4), color: '#38bdf8' },
                ].map(c => (
                  <div key={c.label} style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{c.label}</div>
                    <div style={{ fontSize: '18px', fontWeight: 700, color: c.color, fontVariantNumeric: 'tabular-nums' }}>{c.value}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                <strong>Top Root Cause Sensors:</strong> {batchResults.top_anomalous_sensors.map(s => `${s.sensor_name} (${s.reconstruction_error.toFixed(3)})`).join(', ')}
              </div>
            </div>
          )}
        </div>

        {/* ── Recent Alerts ── */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Anomaly Events Log</div>
              <div className="card-subtitle">One entry per anomaly event — for human review only</div>
            </div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button
                onClick={handleJumpToAnomaly}
                disabled={actionLoading}
                className="hdr-btn hdr-btn--primary"
                style={{ fontSize: '12px', padding: '5px 10px' }}
              >
                <Zap size={12} /> Jump to Anomaly Event
              </button>
              {recentAlerts.length > 0 && (
                <button
                  onClick={() => setRecentAlerts([])}
                  className="hdr-btn"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          {recentAlerts.length === 0 ? (
            <div className="placeholder-state" style={{ minHeight: '160px', padding: '24px' }}>
              <div className="placeholder-icon"><AlertTriangle size={40} style={{ opacity: 0.3 }} /></div>
              <div className="placeholder-title">
                {streaming ? 'Monitoring Active — Sensor Behavior Normal' : 'Stream Paused'}
              </div>
              <div className="placeholder-text" style={{ maxWidth: '540px', margin: '0 auto 16px' }}>
                Currently streaming normal baseline data {currentTimestep !== null ? `(Timestep #${currentTimestep})` : ''}.
                Use <strong>Jump to Anomaly</strong> to demonstrate live anomaly events.
              </div>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
                <button
                  onClick={handleJumpToAnomaly}
                  disabled={actionLoading}
                  style={{ background: 'linear-gradient(135deg, #6366f1, #4f46e5)', color: '#fff', border: 'none', padding: '8px 18px', borderRadius: '8px', fontSize: '13px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  <Zap size={14} /> Jump to Anomaly Window
                </button>
              </div>
            </div>
          ) : (
            <div style={{ padding: '0 20px 16px', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                    {['#', 'Time', 'Peak Error', 'Threshold', 'Top Root Cause', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recentAlerts.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '8px 12px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>{recentAlerts.length - i}</td>
                      <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums' }}>{a.time}</td>
                      <td style={{ padding: '8px 12px', color: '#ef4444', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{a.error}</td>
                      <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums' }}>{a.threshold}</td>
                      <td
                        style={{ padding: '8px 12px', color: '#f59e0b', fontFamily: 'monospace', cursor: 'pointer' }}
                        title="Click to view channel deep dive"
                        onClick={() => {
                          const match = a.topSensor?.match(/sensor_(\d+)/i) || a.topSensor?.match(/ch-?(\d+)/i);
                          if (match) setDetailChannel(parseInt(match[1], 10));
                        }}
                      >
                        {a.topSensor}
                      </td>
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

      {/* ── Maximized Chart Modals ── */}
      <ChartModal
        isOpen={maximizedChart === 'sensor_ts'}
        onClose={() => setMaximizedChart(null)}
        title="Sensor Time Series (Multi-Sensor Telemetry)"
        subtitle="Real-time multi-sensor values (Sensors 00, 01, 02) with live streaming updates"
        plotData={[
          { x: timestamps, y: sensor0, mode: 'lines', name: 'Sensor 00', line: { color: '#6366f1', width: 2 } },
          { x: timestamps, y: sensor1, mode: 'lines', name: 'Sensor 01', line: { color: '#10b981', width: 2 } },
          { x: timestamps, y: sensor2, mode: 'lines', name: 'Sensor 02', line: { color: '#f59e0b', width: 2 } },
        ]}
        plotLayout={{ hovermode: 'x unified', xaxis: { title: 'Timestamp' }, yaxis: { title: 'Normalized Sensor Value' } }}
      />
      <ChartModal
        isOpen={maximizedChart === 'recon_error'}
        onClose={() => setMaximizedChart(null)}
        title="Reconstruction Error & Anomaly Detection"
        subtitle="Raw error, smoothed error (k=5), decision threshold cutoff, and flagged anomaly timestamps"
        plotData={[
          { x: timestamps.slice(-reconErrors.length), y: reconErrors, mode: 'lines', name: 'Raw Error', line: { color: 'rgba(99,102,241,0.45)', width: 1.5 } },
          { x: timestamps.slice(-smoothedErrors.length), y: smoothedErrors, mode: 'lines', name: 'Smoothed Error (k=5)', line: { color: '#6366f1', width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.1)' },
          { x: [timestamps[Math.max(0, timestamps.length - reconErrors.length)], timestamps[timestamps.length - 1]], y: [threshold.current, threshold.current], mode: 'lines', name: 'Threshold', line: { color: '#ef4444', width: 2, dash: 'dash' } },
          { x: timestamps.slice(-reconErrors.length).filter((_, i) => anomalyFlags[i]), y: reconErrors.filter((_, i) => anomalyFlags[i]), mode: 'markers', name: 'Anomaly Flag', marker: { color: '#ef4444', size: 9, symbol: 'circle' } },
        ]}
        plotLayout={{ hovermode: 'x unified', xaxis: { title: 'Timestamp' }, yaxis: { title: 'Reconstruction Error (MSE)' } }}
      />

      {/* ── Channel Detail Modal ── */}
      <ChannelDetailModal
        isOpen={detailChannel !== null}
        onClose={() => setDetailChannel(null)}
        channelIndex={detailChannel ?? 0}
        onSelectChannel={idx => setDetailChannel(idx)}
        timestamps={timestamps}
        sensorValues={sensorValues}
        channelErrors={channelErrors}
        threshold={threshold.current}
        totalChannels={meta?.n_features ?? 38}
      />
    </>
  );
}
