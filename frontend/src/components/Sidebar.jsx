/**
 * Sidebar navigation component.
 *
 * Provides the main navigation structure for the dashboard.
 * Uses react-router-dom NavLink for active state management.
 */

import { NavLink } from 'react-router-dom';

export default function Sidebar() {
  return (
    <aside className="sidebar">
      {/* ── Logo ─────────────────────────────── */}
      <div className="sidebar-header">
        <NavLink to="/" className="sidebar-logo">
          <div className="sidebar-logo-icon">⚡</div>
          <div>
            <div className="sidebar-logo-text">AnomalyGuard</div>
            <div className="sidebar-logo-sub">IoT Monitoring</div>
          </div>
        </NavLink>
      </div>

      {/* ── Navigation ───────────────────────── */}
      <nav className="sidebar-nav">
        <div className="nav-section-title">Monitoring</div>

        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <span className="nav-link-icon">📊</span>
          Dashboard
        </NavLink>

        <NavLink
          to="/sensors"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <span className="nav-link-icon">📡</span>
          Sensor Streams
        </NavLink>

        <div className="nav-section-title">Analysis</div>

        <NavLink
          to="/evaluation"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <span className="nav-link-icon">📈</span>
          Model Evaluation
        </NavLink>

        <NavLink
          to="/model-info"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <span className="nav-link-icon">🧠</span>
          Model Info
        </NavLink>

        <div className="nav-section-title">System</div>

        <NavLink
          to="/settings"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <span className="nav-link-icon">⚙️</span>
          Settings
        </NavLink>
      </nav>

      {/* ── Footer ───────────────────────────── */}
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="status-dot status-dot--healthy"></span>
            <span style={{ fontSize: '13px', fontWeight: 500 }}>System Online</span>
          </div>
          <span style={{ fontSize: '11px', padding: '2px 6px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px' }}>
            v1.0.0
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ overflow: 'hidden' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
              👤 {localStorage.getItem('username') || 'Operator'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {localStorage.getItem('user_role') || 'Plant Operations'}
            </div>
          </div>
          <button
            onClick={() => {
              localStorage.removeItem('auth_token');
              localStorage.removeItem('user_role');
              localStorage.removeItem('username');
              window.location.href = '/login';
            }}
            title="Sign out"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              fontSize: '14px',
              padding: '4px',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
            }}
            onMouseOver={(e) => (e.currentTarget.style.color = '#ef4444')}
            onMouseOut={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
          >
            🚪
          </button>
        </div>
      </div>
    </aside>
  );
}
