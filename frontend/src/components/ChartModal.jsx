/**
 * Reusable Chart Modal Component
 * 
 * Provides click-to-maximize for all dashboard and sensor charts:
 * - Centered overlay (~90vw x 85vh on desktop, full-screen on mobile)
 * - Dimmed, blurred backdrop with smooth scale/fade animation
 * - Live updating from the parent data source
 * - Zoom (scroll/drag-select), pan, crosshair tooltip (x unified)
 * - Legend toggling and "Reset zoom" button
 * - Clean layout with legend placed below axis labels without overlap
 * - Format x-axis ticks with reduced density
 * - Accessible: Esc key, backdrop click, focus trap, and body scroll lock
 */

import { useEffect, useRef, useState, useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function ChartModal({
  isOpen,
  onClose,
  title,
  subtitle,
  plotData,
  plotLayout = {},
  plotConfig = {},
  onResetZoom,
}) {
  const modalRef = useRef(null);
  const [resetKey, setResetKey] = useState(0);

  // Esc key, backdrop click, focus trap, and body scroll management
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
  }, [isOpen, onClose]);

  const handleResetZoomClick = () => {
    setResetKey(k => k + 1);
    if (onResetZoom) onResetZoom();
  };

  // Enhance layout for maximized display
  const mergedLayout = useMemo(() => {
    const isHorizontalBar = plotData?.[0]?.type === 'bar' && plotData?.[0]?.orientation === 'h';

    return {
      ...plotLayout,
      uirevision: resetKey, // forces Plotly to re-autorange on reset
      autosize: true,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94a3b8', size: 13 },
      hovermode: isHorizontalBar ? 'closest' : (plotLayout.hovermode || 'x unified'),
      hoverlabel: {
        bgcolor: '#0f172a',
        bordercolor: '#334155',
        font: { color: '#f8fafc', size: 12 },
        ...plotLayout.hoverlabel,
      },
      showlegend: plotLayout.showlegend ?? true,
      legend: {
        orientation: 'h',
        x: 0.5,
        xanchor: 'center',
        y: -0.20,
        yanchor: 'top',
        font: { size: 12, color: '#cbd5e1' },
        ...plotLayout.legend,
      },
      margin: {
        t: 25,
        b: 90,
        l: Math.max(70, plotLayout.margin?.l || 50),
        r: Math.max(30, plotLayout.margin?.r || 15),
        ...plotLayout.margin,
      },
      xaxis: {
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.06)',
        zeroline: false,
        automargin: true,
        nticks: isHorizontalBar ? 8 : 10,
        tickfont: { size: 11, color: '#94a3b8' },
        ...plotLayout.xaxis,
      },
      yaxis: {
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.06)',
        zeroline: false,
        automargin: true,
        tickfont: { size: 11, color: '#94a3b8' },
        ...plotLayout.yaxis,
      },
    };
  }, [plotLayout, plotData, resetKey]);

  const mergedConfig = useMemo(() => ({
    responsive: true,
    scrollZoom: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    ...plotConfig,
  }), [plotConfig]);

  if (!isOpen) return null;

  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label={title || 'Maximized Chart'}
    >
      <div
        className="modal-content chart-modal-content"
        ref={modalRef}
        tabIndex={-1}
        outline="none"
      >
        <div className="modal-header">
          <div className="modal-header-left">
            <div className="modal-title-row">
              <span className="modal-icon">⛶</span>
              <h2 className="modal-title">{title}</h2>
              <span className="modal-live-tag">● Live Updating</span>
            </div>
            {subtitle && <p className="modal-subtitle">{subtitle}</p>}
          </div>

          <div className="modal-header-actions">
            <button
              onClick={handleResetZoomClick}
              className="modal-action-btn"
              title="Reset chart zoom and pan to full view"
            >
              ⟲ Reset Zoom
            </button>
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

        <div className="modal-body chart-modal-body">
          {plotData && plotData.length > 0 ? (
            <Plot
              data={plotData}
              layout={mergedLayout}
              config={mergedConfig}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%', minHeight: '400px' }}
            />
          ) : (
            <div className="placeholder-state" style={{ minHeight: '400px' }}>
              <div className="placeholder-icon">📈</div>
              <div className="placeholder-title">Waiting for chart data…</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
