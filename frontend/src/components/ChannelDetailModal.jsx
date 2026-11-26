/**
 * Per-Channel Detail Modal Component
 * 
 * Inspects a single sensor channel in depth:
 * - Header: channel name, current value, current error, alert status badge
 * - Navigation: Previous / Next channel arrows and keyboard arrow keys (← / →)
 * - Time range selector: Last 30s / 1m / 5m / All
 * - Stats row: Min, Max, Mean, Std Dev, Number of anomalies in visible window
 * - Line chart 1: Channel value over time
 * - Line chart 2: Channel reconstruction error (raw + smoothed k=5, threshold line, anomaly markers)
 * - Lazy-rendered: Only mounts when opened, destroyed on close to prevent memory leaks.
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { calculateCausalSmoothedErrors, calculateWindowStats } from '../utils/channelStats';

export default function ChannelDetailModal({
  isOpen,
  onClose,
  channelIndex = 0,
  onSelectChannel,
  timestamps = [],
  sensorValues = {},
  channelErrors = {},
  threshold = 0.5,
  totalChannels = 38,
}) {
  const modalRef = useRef(null);
  const [timeRange, setTimeRange] = useState('1m'); // '30s' | '1m' | '5m' | 'all'

  // Focus trap, Esc key, and Left/Right keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const prevActive = document.activeElement;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    if (modalRef.current) {
      modalRef.current.focus();
    }

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onSelectChannel((channelIndex - 1 + totalChannels) % totalChannels);
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        onSelectChannel((channelIndex + 1) % totalChannels);
        return;
      }
      if (e.key === 'Tab' && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prevOverflow;
      if (prevActive && typeof prevActive.focus === 'function') {
        prevActive.focus();
      }
    };
  }, [isOpen, onClose, channelIndex, onSelectChannel, totalChannels]);

  // Determine window sample count based on selected time range
  const sampleCount = useMemo(() => {
    switch (timeRange) {
      case '30s': return 30;
      case '1m': return 60;
      case '5m': return 150;
      case 'all':
      default:
        return 300;
    }
  }, [timeRange]);

  const rawValues = sensorValues[channelIndex] || [];
  const rawErrors = channelErrors[channelIndex] || [];

  const visibleTimestamps = useMemo(() => timestamps.slice(-sampleCount), [timestamps, sampleCount]);
  const visibleValues = useMemo(() => rawValues.slice(-sampleCount), [rawValues, sampleCount]);
  const visibleErrors = useMemo(() => rawErrors.slice(-sampleCount), [rawErrors, sampleCount]);

  // Causal smoothed errors (k=5 trailing moving average)
  const smoothedErrors = useMemo(() => {
    return calculateCausalSmoothedErrors(visibleErrors, 5);
  }, [visibleErrors]);

  // Statistics across the visible window
  const stats = useMemo(() => {
    return calculateWindowStats(visibleValues, visibleErrors, threshold);
  }, [visibleValues, visibleErrors, threshold]);

  if (!isOpen) return null;

  const currentVal = rawValues.length > 0 ? rawValues[rawValues.length - 1] : null;
  const currentErr = rawErrors.length > 0 ? rawErrors[rawErrors.length - 1] : null;
  const isElevated = currentErr !== null && currentErr > threshold;
  const channelName = `Sensor ${channelIndex < 10 ? `0${channelIndex}` : channelIndex}`;

  const prevChannel = (channelIndex - 1 + totalChannels) % totalChannels;
  const nextChannel = (channelIndex + 1) % totalChannels;

  // Anomaly marker timestamps and values
  const anomalyTimestamps = [];
  const anomalyPoints = [];
  visibleErrors.forEach((err, idx) => {
    if (err > threshold && idx < visibleTimestamps.length) {
      anomalyTimestamps.push(visibleTimestamps[idx]);
      anomalyPoints.push(err);
    }
  });

  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label={`${channelName} Detail View`}
    >
      <div
        className="modal-content channel-modal-content"
        ref={modalRef}
        tabIndex={-1}
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-header-left">
            <div className="channel-nav-controls">
              <button
                onClick={() => onSelectChannel(prevChannel)}
                className="channel-nav-btn"
                title={`Previous channel: Ch ${prevChannel < 10 ? `0${prevChannel}` : prevChannel} (← Key)`}
              >
                ◀ Ch {prevChannel < 10 ? `0${prevChannel}` : prevChannel}
              </button>
              <div className="channel-title-pill">
                <span className="channel-title-icon">📊</span>
                <span className="channel-title-text">{channelName}</span>
                <span className={`channel-badge ${isElevated ? 'channel-badge--alert' : 'channel-badge--normal'}`}>
                  {isElevated ? '⚠️ ELEVATED' : '● NORMAL'}
                </span>
              </div>
              <button
                onClick={() => onSelectChannel(nextChannel)}
                className="channel-nav-btn"
                title={`Next channel: Ch ${nextChannel < 10 ? `0${nextChannel}` : nextChannel} (→ Key)`}
              >
                Ch {nextChannel < 10 ? `0${nextChannel}` : nextChannel} ▶
              </button>
            </div>
          </div>

          <div className="modal-header-actions">
            {/* Time range selector */}
            <div className="time-range-group">
              <span className="time-range-label">Range:</span>
              {['30s', '1m', '5m', 'all'].map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`time-range-btn ${timeRange === range ? 'active' : ''}`}
                >
                  {range === 'all' ? 'All' : range}
                </button>
              ))}
            </div>

            <button
              onClick={onClose}
              className="modal-close-btn"
              aria-label="Close modal"
              title="Close (Esc)"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="modal-body channel-modal-body">
          {/* Channel Key Metrics Row */}
          <div className="channel-metrics-strip">
            <div className="channel-metric-cell">
              <span className="channel-metric-label">Current Reading</span>
              <span className="channel-metric-val">{currentVal !== null ? currentVal.toFixed(4) : '—'}</span>
            </div>
            <div className="channel-metric-cell">
              <span className="channel-metric-label">Reconstruction Error</span>
              <span className="channel-metric-val" style={{ color: isElevated ? '#ef4444' : '#10b981' }}>
                {currentErr !== null ? currentErr.toFixed(4) : '—'}
              </span>
            </div>
            <div className="channel-metric-cell">
              <span className="channel-metric-label">Threshold Cutoff</span>
              <span className="channel-metric-val" style={{ color: '#38bdf8' }}>{threshold.toFixed(4)}</span>
            </div>
            <div className="channel-metric-cell">
              <span className="channel-metric-label">Anomalies in Window</span>
              <span className="channel-metric-val" style={{ color: stats.anomalies > 0 ? '#ef4444' : '#94a3b8' }}>
                {stats.anomalies}
              </span>
            </div>
          </div>

          {/* Statistical Breakdown Row */}
          <div className="channel-stats-row">
            <div className="stat-chip">
              <span className="stat-chip-label">Min:</span>
              <span className="stat-chip-val">{stats.min !== null ? stats.min.toFixed(4) : '—'}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-label">Max:</span>
              <span className="stat-chip-val">{stats.max !== null ? stats.max.toFixed(4) : '—'}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-label">Mean:</span>
              <span className="stat-chip-val">{stats.mean !== null ? stats.mean.toFixed(4) : '—'}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-label">Std Dev:</span>
              <span className="stat-chip-val">{stats.std !== null ? stats.std.toFixed(4) : '—'}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-label">Window Size:</span>
              <span className="stat-chip-val">{visibleValues.length} pts</span>
            </div>
          </div>

          {/* Dual Charts Grid */}
          <div className="channel-charts-grid">
            {/* Chart 1: Sensor Value Over Time */}
            <div className="channel-chart-box">
              <div className="chart-box-header">
                <span className="chart-box-title">Telemetry Signal — {channelName}</span>
                <span className="chart-box-sub">Normalized value trajectory</span>
              </div>
              {visibleValues.length > 0 ? (
                <Plot
                  data={[
                    {
                      x: visibleTimestamps,
                      y: visibleValues,
                      mode: 'lines',
                      name: channelName,
                      line: { color: '#6366f1', width: 2 },
                      fill: 'tozeroy',
                      fillcolor: 'rgba(99, 102, 241, 0.08)',
                    },
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', size: 11 },
                    hovermode: 'x unified',
                    hoverlabel: { bgcolor: '#0f172a', font: { color: '#f8fafc', size: 11 } },
                    xaxis: {
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.05)',
                      zeroline: false,
                      automargin: true,
                      nticks: 6,
                    },
                    yaxis: {
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.05)',
                      zeroline: false,
                      automargin: true,
                    },
                    margin: { t: 20, b: 40, l: 50, r: 15 },
                    showlegend: false,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  useResizeHandler={true}
                  style={{ width: '100%', height: '220px' }}
                />
              ) : (
                <div className="placeholder-state" style={{ minHeight: '200px' }}>
                  <div className="placeholder-icon">📡</div>
                  <div className="placeholder-title">No sensor stream data</div>
                </div>
              )}
            </div>

            {/* Chart 2: Reconstruction Error Over Time */}
            <div className="channel-chart-box">
              <div className="chart-box-header">
                <span className="chart-box-title">Reconstruction Error & Threshold</span>
                <span className="chart-box-sub">Raw error vs smoothed (k=5) & anomaly markers</span>
              </div>
              {visibleErrors.length > 0 ? (
                <Plot
                  data={[
                    {
                      x: visibleTimestamps,
                      y: visibleErrors,
                      mode: 'lines',
                      name: 'Raw Error',
                      line: { color: 'rgba(99,102,241,0.35)', width: 1.2 },
                    },
                    {
                      x: visibleTimestamps,
                      y: smoothedErrors,
                      mode: 'lines',
                      name: 'Smoothed Error (k=5)',
                      line: { color: '#8b5cf6', width: 2.2 },
                      fill: 'tozeroy',
                      fillcolor: 'rgba(139,92,246,0.08)',
                    },
                    {
                      x: [visibleTimestamps[0], visibleTimestamps[visibleTimestamps.length - 1]],
                      y: [threshold, threshold],
                      mode: 'lines',
                      name: 'Threshold',
                      line: { color: '#ef4444', width: 1.8, dash: 'dash' },
                    },
                    {
                      x: anomalyTimestamps,
                      y: anomalyPoints,
                      mode: 'markers',
                      name: 'Anomaly Flag',
                      marker: { color: '#ef4444', size: 7, symbol: 'circle' },
                    },
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', size: 11 },
                    hovermode: 'x unified',
                    hoverlabel: { bgcolor: '#0f172a', font: { color: '#f8fafc', size: 11 } },
                    xaxis: {
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.05)',
                      zeroline: false,
                      automargin: true,
                      nticks: 6,
                    },
                    yaxis: {
                      showgrid: true,
                      gridcolor: 'rgba(255,255,255,0.05)',
                      zeroline: false,
                      automargin: true,
                    },
                    margin: { t: 20, b: 50, l: 50, r: 15 },
                    showlegend: true,
                    legend: {
                      orientation: 'h',
                      x: 0.5,
                      xanchor: 'center',
                      y: -0.22,
                      yanchor: 'top',
                      font: { size: 10, color: '#94a3b8' },
                    },
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  useResizeHandler={true}
                  style={{ width: '100%', height: '220px' }}
                />
              ) : (
                <div className="placeholder-state" style={{ minHeight: '200px' }}>
                  <div className="placeholder-icon">📈</div>
                  <div className="placeholder-title">Awaiting error data</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
