/**
 * Sidebar navigation component — Phase 1 refresh with Lucide icons.
 */

import { NavLink, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard, Radio, BarChart2, Brain, Settings, LogOut, Zap, User
} from 'lucide-react';

const NAV_ITEMS = [
  { section: 'Monitoring' },
  { to: '/', end: true,        icon: <LayoutDashboard size={16} />, label: 'Dashboard' },
  { to: '/sensors',            icon: <Radio size={16} />,           label: 'Sensor Streams' },
  { section: 'Analysis' },
  { to: '/evaluation',         icon: <BarChart2 size={16} />,       label: 'Model Evaluation' },
  { to: '/model-info',         icon: <Brain size={16} />,           label: 'Model Info' },
  { section: 'System' },
  { to: '/settings',           icon: <Settings size={16} />,        label: 'Settings' },
];

export default function Sidebar() {
  const navigate = useNavigate();

  const handleSignOut = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    navigate('/login');
  };

  return (
    <aside className="sidebar">
      {/* ── Logo ── */}
      <div className="sidebar-header">
        <NavLink to="/" className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <Zap size={18} />
          </div>
          <div>
            <div className="sidebar-logo-text">AnomalyGuard</div>
            <div className="sidebar-logo-sub">IoT Monitoring</div>
          </div>
        </NavLink>
      </div>

      {/* ── Navigation ── */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item, i) =>
          item.section ? (
            <div key={i} className="nav-section-title">{item.section}</div>
          ) : (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <span className="nav-link-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          )
        )}
      </nav>

      {/* ── Footer ── */}
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="status-dot status-dot--healthy" />
            <span style={{ fontSize: '13px', fontWeight: 500 }}>System Online</span>
          </div>
          <span style={{ fontSize: '11px', padding: '2px 6px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px' }}>
            v1.0.0
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
            <User size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
                {localStorage.getItem('username') || 'Operator'}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {localStorage.getItem('user_role') || 'Plant Operations'}
              </div>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            title="Sign out"
            aria-label="Sign out"
            style={{
              background: 'transparent', border: 'none',
              color: 'var(--text-muted)', cursor: 'pointer',
              padding: '4px', borderRadius: '4px',
              display: 'flex', alignItems: 'center',
              transition: 'color 0.15s ease',
            }}
            onMouseOver={e => (e.currentTarget.style.color = '#ef4444')}
            onMouseOut={e => (e.currentTarget.style.color = 'var(--text-muted)')}
          >
            <LogOut size={14} />
          </button>
        </div>
      </div>
    </aside>
  );
}
