/**
 * SensorsPage — FR-004 & FR-009: Multi-Channel Sensor Monitoring & Root Cause Analysis
 *
 * Provides dedicated inspection of all 38 sensor channels, with
 * per-channel live tracking and reconstruction error breakdown.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { getStreamData, checkHealth, getModelMetadata } from '../services/api';
import { useModelThreshold } from '../hooks/useModelThreshold';
import ChartModal from '../components/ChartModal';
import ChannelDetailModal from '../components/ChannelDetailModal';

const MAX_POINTS = 300;

// Sensor group presets
const SENSOR_GROUPS = {
  core: { name: 'Compute & Memory (Sensors 0–7)', indices: [0, 1, 2, 3, 4, 5, 6, 7] },
  io: { name: 'Storage & I/O (Sensors 8–15)', indices: [8, 9, 10, 11, 12, 13, 14, 15] },
  primary: { name: 'Primary Channels (Ch 00–15)', indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] },
  network: { name: 'Network & Bus (Sensors 16–25)', indices: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25] },
  thermal: { name: 'Power & Thermal (Sensors 26–37)', indices: [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] },
};

const PALETTE = [
  '#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
  '#06b6d4', '#14b8a6', '#f97316', '#3b82f6', '#84cc16'
];

export default function SensorsPage() {
  const [selectedGroup, setSelectedGroup] = useState('core');
  const [activeChannel, setActiveChannel] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [streamData, setStreamData] = useState([]); // timestamps
  const [sensorValues, setSensorValues] = useState({}); // { [idx]: number[] }
  const [channelErrors, setChannelErrors] = useState({}); // { [idx]: number[] }
  const [latestErrors, setLatestErrors] = useState([]); // per-sensor error
  const [latestAnomalies, setLatestAnomalies] = useState(0);
  const [health, setHealth] = useState(null);
  const [meta, setMeta] = useState(null);
  const [lastStreamPoint, setLastStreamPoint] = useState(null);
  const { threshold: sensorThreshold, loaded: thresholdLoaded, updateFromStream } = useModelThreshold();

  // Modal states
  const [maximizedChart, setMaximizedChart] = useState(null); // 'group_stream' | 'root_cause' | null
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  const streamRef = useRef(null);

  useEffect(() => {
    checkHealth().then(r => setHealth(r.data)).catch(() => {});
    // Dynamically derive sensor cutoff from API model metadata.
    // In unnormalized mode, per-sensor MSE contributions are evaluated relative to the calibrated window threshold.
    getModelMetadata()
      .then(res => {
        setMeta(res.data);
      })
      .catch(() => {});
  }, []);

  const fetchPoint = useCallback(async () => {
    try {
      const res = await getStreamData();
      const d = res.data;
      if (d.threshold != null) {
        updateFromStream(d.threshold);
      }
      setLastStreamPoint(d);
      const now = new Date().toLocaleTimeString();

      setStreamData(prev => [...prev.slice(-MAX_POINTS), now]);

      setSensorValues(prev => {
        const next = { ...prev };
        d.sensor_values.forEach((val, idx) => {
          const arr = next[idx] || [];
          next[idx] = [...arr.slice(-MAX_POINTS), val];
        });
        return next;
      });

      if (d.sensor_errors) {
        setLatestErrors(d.sensor_errors);
        setChannelErrors(prev => {
          const next = { ...prev };
          d.sensor_errors.forEach(errObj => {
            const arr = next[errObj.sensor_index] || [];
            next[errObj.sensor_index] = [...arr.slice(-MAX_POINTS), errObj.reconstruction_error];
          });
          return next;
        });
      }
      if (d.is_anomaly) {
        setLatestAnomalies(c => c + 1);
      }
    } catch {
      // ignore
    }
  }, []);

  const startStream = useCallback(() => {
    if (streamRef.current) return;
    const speed = parseInt(localStorage.getItem('setting_stream_speed') || '500', 10);
    setStreaming(true);
    streamRef.current = setInterval(fetchPoint, speed);
  }, [fetchPoint]);

  const stopStream = () => {
    clearInterval(streamRef.current);
    streamRef.current = null;
    setStreaming(false);
  };

  // Auto-start stream on load once health confirms model is loaded
  useEffect(() => {
    if (health?.model_loaded && !streamRef.current) {
      const initBurst = async () => {
        for (let i = 0; i < 4; i++) {
          await fetchPoint();
        }
        startStream();
      };
      initBurst();
    }
  }, [health?.model_loaded, fetchPoint, startStream]);

  useEffect(() => () => clearInterval(streamRef.current), []);

  const groupIndices = SENSOR_GROUPS[selectedGroup]?.indices || [0, 1, 2];

  // Top 5 root cause contributors
  const sortedErrors = [...latestErrors].sort((a, b) => b.reconstruction_error - a.reconstruction_error).slice(0, 8);

  const activeSensorHistory = sensorValues[activeChannel] || [];
  const activeCurrentVal = activeSensorHistory.length > 0 ? activeSensorHistory[activeSensorHistory.length - 1] : null;
  const activeMinVal = activeSensorHistory.length > 0 ? Math.min(...activeSensorHistory) : null;
  const activeMaxVal = activeSensorHistory.length > 0 ? Math.max(...activeSensorHistory) : null;

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Sensor Streams & Root Cause Analysis</h1>
          <p className="page-subtitle">Multi-channel IoT telemetry ({meta?.n_features ?? lastStreamPoint?.sensor_values?.length ?? 'n/a'} SMD sensor channels)</p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={streaming ? stopStream : startStream}
            style={{
              padding: '8px 20px', borderRadius: '8px', border: 'none', cursor: 'pointer',
              background: streaming ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
              color: streaming ? '#ef4444' : '#10b981',
              fontWeight: 600, fontSize: '13px',
            }}
          >
            {streaming ? '⏸ Pause Stream' : '▶ Resume Stream'}
          </button>
          <div className={`status-badge ${health?.model_loaded ? 'status-badge--healthy' : 'status-badge--warning'}`}>
            <span className={`status-dot ${health?.model_loaded ? 'status-dot--healthy' : 'status-dot--warning'}`}></span>
            {health?.model_loaded ? 'Model Online' : 'Model Offline'}
          </div>
        </div>
      </header>

      <div className="page-body">
        {/* ── Preset selector bar ── */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '18px', flexWrap: 'wrap' }}>
          {Object.entries(SENSOR_GROUPS).map(([key, group]) => (
            <button
              key={key}
              onClick={() => {
                setSelectedGroup(key);
                setActiveChannel(group.indices[0]);
              }}
              style={{
                padding: '8px 16px', borderRadius: '8px', cursor: 'pointer',
                background: selectedGroup === key ? 'var(--primary, #6366f1)' : 'var(--surface-1)',
                color: selectedGroup === key ? '#ffffff' : 'var(--text-muted)',
                border: '1px solid rgba(255,255,255,0.08)',
                fontSize: '13px', fontWeight: 500,
                transition: 'all 0.15s ease',
              }}
            >
              {group.name}
            </button>
          ))}
        </div>

        {/* ── Channel quick-cards grid ── */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
          gap: '10px',
          marginBottom: '20px'
        }}>
          {groupIndices.map(idx => {
            const isSelected = activeChannel === idx;
            const values = sensorValues[idx] || [];
            const lastVal = values.length > 0 ? values[values.length - 1] : 0;
            const errObj = latestErrors.find(e => e.sensor_index === idx);
            // Sensor flagged elevated if threshold is loaded and error exceeds dynamically derived threshold
            const isElevated = thresholdLoaded && errObj && errObj.reconstruction_error > sensorThreshold;

            return (
              <div
                key={idx}
                onClick={() => {
                  setActiveChannel(idx);
                  setDetailModalOpen(true);
                }}
                className="chart-card-clickable"
                title={`Channel ${idx < 10 ? `0${idx}` : idx} — Click to open detail modal`}
                style={{
                  background: isSelected ? 'rgba(99,102,241,0.2)' : 'var(--surface-1)',
                  border: isSelected
                    ? '2px solid #6366f1'
                    : isElevated
                    ? '1px solid rgba(239,68,68,0.45)'
                    : '1px solid rgba(255,255,255,0.06)',
                  boxShadow: isSelected ? '0 0 15px rgba(99,102,241,0.3)' : undefined,
                  borderRadius: '10px',
                  padding: '10px 12px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <span style={{ fontSize: '11px', color: isSelected ? '#818cf8' : 'var(--text-muted)', fontWeight: isSelected ? 700 : 500 }}>
                    Ch {idx < 10 ? `0${idx}` : idx}
                  </span>
                  {isElevated ? (
                    <span style={{ color: '#ef4444', fontSize: '10px', fontWeight: 600 }}>⚠️ alert</span>
                  ) : (
                    <span style={{ color: '#10b981', fontSize: '10px' }}>●</span>
                  )}
                </div>
                <div style={{ fontSize: '16px', fontWeight: 600, color: isSelected ? '#ffffff' : 'var(--text-primary)' }}>
                  {lastVal.toFixed(3)}
                </div>
                {errObj && (
                  <div style={{ fontSize: '10px', color: isElevated ? '#ef4444' : 'var(--text-muted)', marginTop: '2px' }}>
                    err: {errObj.reconstruction_error.toFixed(3)}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Main Charts Area: Multi-stream & Root Cause Analysis ── */}
        <div className="charts-grid">
          {/* Multi-Sensor Channels Plot */}
          <div
            className="chart-container chart-card-clickable"
            onClick={() => setMaximizedChart('group_stream')}
            title="Click to maximize multi-sensor plot"
          >
            <div className="card-header">
              <div>
                <div className="card-title">{SENSOR_GROUPS[selectedGroup]?.name}</div>
                <div className="card-subtitle">Real-time normalized values across channels</div>
              </div>
              <div className="chart-header-actions">
                <button
                  type="button"
                  className="chart-expand-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMaximizedChart('group_stream');
                  }}
                  title="Maximize chart"
                >
                  ⛶ Maximize
                </button>
              </div>
            </div>
            {streamData.length > 1 ? (
              <Plot
                data={groupIndices.slice(0, 5).map((idx, i) => ({
                  x: streamData,
                  y: sensorValues[idx] || [],
                  mode: 'lines',
                  name: `Sensor ${idx < 10 ? `0${idx}` : idx}`,
                  line: { color: PALETTE[i % PALETTE.length], width: activeChannel === idx ? 2.5 : 1.2 },
                }))}
                layout={{
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 11 },
                  xaxis: { showgrid: false, automargin: true, nticks: 6, tickangle: 0 },
                  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', automargin: true },
                  margin: { t: 15, b: 65, l: 45, r: 15 },
                  showlegend: true,
                  legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.25, yanchor: 'top', font: { size: 10 } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '280px' }}
              />
            ) : (
              <div className="placeholder-state" style={{ minHeight: '260px' }}>
                <div className="placeholder-icon">📡</div>
                <div className="placeholder-title">{streaming ? 'Receiving sensor stream…' : 'Stream Inactive'}</div>
                <div className="placeholder-text">Press ▶ Start Stream above to stream multi-sensor readings.</div>
              </div>
            )}
          </div>

          {/* Root Cause Analysis (FR-009) Top Contributing Sensors */}
          <div
            className="chart-container chart-card-clickable"
            onClick={() => setMaximizedChart('root_cause')}
            title="Click to maximize Root Cause Analysis"
          >
            <div className="card-header">
              <div>
                <div className="card-title">Root Cause Analysis (Top Channels)</div>
                <div className="card-subtitle">Sensors contributing highest reconstruction error (Click bar for details)</div>
              </div>
              <div className="chart-header-actions">
                <span style={{ fontSize: '11px', background: 'rgba(99,102,241,0.1)', color: '#6366f1', padding: '3px 8px', borderRadius: '4px' }}>
                  FR-009
                </span>
                <button
                  type="button"
                  className="chart-expand-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMaximizedChart('root_cause');
                  }}
                  title="Maximize chart"
                >
                  ⛶ Maximize
                </button>
              </div>
            </div>
            {sortedErrors.length > 0 ? (
              <Plot
                data={[
                  {
                    x: sortedErrors.map(e => e.reconstruction_error),
                    y: sortedErrors.map(e => e.sensor_name),
                    type: 'bar',
                    orientation: 'h',
                    text: sortedErrors.map(e => e.reconstruction_error.toFixed(4)),
                    textposition: 'outside',
                    textfont: { color: '#cbd5e1', size: 10 },
                    cliponaxis: false,
                    marker: {
                      color: sortedErrors.map(e => (thresholdLoaded && e.reconstruction_error > sensorThreshold) ? '#ef4444' : '#6366f1'),
                    },
                  },
                ]}
                layout={{
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 11 },
                  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', title: 'Reconstruction Error', automargin: true },
                  yaxis: { autorange: 'reversed', automargin: true },
                  margin: { t: 30, b: 50, l: 85, r: 65 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                onClick={(event) => {
                  if (event?.points?.length > 0) {
                    const pt = event.points[0];
                    const errItem = sortedErrors[pt.pointIndex];
                    if (errItem) {
                      setActiveChannel(errItem.sensor_index);
                      setDetailModalOpen(true);
                    }
                  }
                }}
                style={{ width: '100%', height: '280px' }}
              />
            ) : (
              <div className="placeholder-state" style={{ minHeight: '260px' }}>
                <div className="placeholder-icon">🔍</div>
                <div className="placeholder-title">Awaiting Inference Data</div>
                <div className="placeholder-text">When streaming, per-sensor error breakdown will rank anomaly root causes.</div>
              </div>
            )}
          </div>
        </div>

        {/* ── Active Channel Deep-Dive ── */}
        <div className="card" style={{ marginTop: '20px' }}>
          <div className="card-header">
            <div>
              <div className="card-title">Channel Deep Dive — Sensor {activeChannel < 10 ? `0${activeChannel}` : activeChannel}</div>
              <div className="card-subtitle">Live statistical behavior and window dispersion</div>
            </div>
            <button
              onClick={() => setDetailModalOpen(true)}
              className="btn btn-primary"
              style={{ padding: '6px 14px', fontSize: '12px' }}
              title="Open full interactive detail modal for this channel"
            >
              🔍 Open Channel Detail Modal
            </button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', padding: '0 20px 20px' }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Current Value</div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)' }}>
                {activeCurrentVal !== null ? activeCurrentVal.toFixed(4) : '—'}
              </div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Window Minimum</div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#10b981' }}>
                {activeMinVal !== null ? activeMinVal.toFixed(4) : '—'}
              </div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Window Maximum</div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#f59e0b' }}>
                {activeMaxVal !== null ? activeMaxVal.toFixed(4) : '—'}
              </div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '10px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Sample Count</div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#6366f1' }}>
                {activeSensorHistory.length}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Maximized Chart Modals ── */}
      <ChartModal
        isOpen={maximizedChart === 'group_stream'}
        onClose={() => setMaximizedChart(null)}
        title={SENSOR_GROUPS[selectedGroup]?.name || 'Multi-Sensor Channels'}
        subtitle="Live real-time normalized sensor telemetry across channel group"
        plotData={groupIndices.slice(0, 10).map((idx, i) => ({
          x: streamData,
          y: sensorValues[idx] || [],
          mode: 'lines',
          name: `Sensor ${idx < 10 ? `0${idx}` : idx}`,
          line: { color: PALETTE[i % PALETTE.length], width: activeChannel === idx ? 2.5 : 1.5 },
        }))}
        plotLayout={{
          hovermode: 'x unified',
          xaxis: { title: 'Timestamp' },
          yaxis: { title: 'Normalized Sensor Value' },
        }}
      />

      <ChartModal
        isOpen={maximizedChart === 'root_cause'}
        onClose={() => setMaximizedChart(null)}
        threshold={sensorThreshold}
        title="Root Cause Analysis (Top Sensor Breakdown)"
        subtitle="Per-channel reconstruction error ranking (Click any bar to open channel deep-dive)"
        plotData={[
          {
            x: sortedErrors.map(e => e.reconstruction_error),
            y: sortedErrors.map(e => e.sensor_name),
            type: 'bar',
            orientation: 'h',
            text: sortedErrors.map(e => e.reconstruction_error.toFixed(4)),
            textposition: 'outside',
            textfont: { color: '#cbd5e1', size: 11 },
            cliponaxis: false,
            marker: {
              color: sortedErrors.map(e => (thresholdLoaded && e.reconstruction_error > sensorThreshold) ? '#ef4444' : '#6366f1'),
            },
          },
        ]}
        plotLayout={{
          hovermode: 'closest',
          xaxis: { title: 'Reconstruction Error (MSE)', automargin: true },
          yaxis: { autorange: 'reversed', automargin: true },
          margin: { t: 30, b: 60, l: 100, r: 80 },
        }}
      />

      {/* ── Per-Channel Detail Modal ── */}
      <ChannelDetailModal
        isOpen={detailModalOpen}
        onClose={() => setDetailModalOpen(false)}
        channelIndex={activeChannel}
        onSelectChannel={(idx) => setActiveChannel(idx)}
        timestamps={streamData}
        sensorValues={sensorValues}
        channelErrors={channelErrors}
        threshold={sensorThreshold}
        totalChannels={meta?.n_features ?? 38}
      />
    </>
  );
}
