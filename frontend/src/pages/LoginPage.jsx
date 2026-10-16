/**
 * Login page — FR-007: Basic Authentication & Session Handling
 *
 * Connects directly to the FastAPI JWT endpoint POST /api/v1/auth/login.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../services/api';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleAuth = async (user, pass) => {
    setLoading(true);
    setError('');

    try {
      const res = await login(user, pass);
      const { access_token } = res.data;
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_role', user === 'admin' ? 'System Admin' : 'Plant Operator');
      localStorage.setItem('username', user);
      navigate('/');
    } catch (err) {
      if (err.response?.status === 401) {
        setError('Invalid username or password. Use demo credentials below.');
      } else {
        setError(err.response?.data?.message || 'Failed to connect to backend service.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleAuth(username, password);
  };

  const handleQuickLogin = (user, pass) => {
    setUsername(user);
    setPassword(pass);
    handleAuth(user, pass);
  };

  return (
    <div className="login-page">
      <div className="login-card">
        {/* ── Logo ─────────────────────────── */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div
            style={{
              width: '56px',
              height: '56px',
              background: 'linear-gradient(135deg, #e94560, #0f3460)',
              borderRadius: '14px',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '28px',
              boxShadow: '0 4px 15px rgba(233, 69, 96, 0.3)',
              marginBottom: '16px',
            }}
          >
            ⚡
          </div>
          <h1 className="login-title">AnomalyGuard</h1>
          <p className="login-subtitle">
            Sign in to access the predictive sensor anomaly monitoring dashboard
          </p>
        </div>

        {/* ── Error Message ────────────────── */}
        {error && (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '13px',
              color: '#ef4444',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── Login Form ───────────────────── */}
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username" className="form-label">Username</label>
            <input
              id="username"
              type="text"
              className="form-input"
              placeholder="e.g. admin or operator"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoFocus
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">Password</label>
            <input
              id="password"
              type="password"
              className="form-input"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: '100%', justifyContent: 'center', marginTop: '8px', padding: '12px' }}
          >
            {loading ? 'Authenticating with Backend...' : 'Sign In'}
          </button>
        </form>

        {/* ── Quick Demo Login ─────────────── */}
        <div style={{ marginTop: '20px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', textAlign: 'center' }}>
            Quick Demo Logins (1-Click)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '12px', padding: '8px' }}
              onClick={() => handleQuickLogin('admin', 'admin')}
              disabled={loading}
            >
              👑 Admin
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '12px', padding: '8px' }}
              onClick={() => handleQuickLogin('operator', 'operator')}
              disabled={loading}
            >
              👷 Operator
            </button>
          </div>
        </div>

        {/* ── Disclaimer ───────────────────── */}
        <div
          style={{
            marginTop: '24px',
            fontSize: '11px',
            color: 'var(--text-muted)',
            textAlign: 'center',
            lineHeight: '1.5',
          }}
        >
          ⚠️ Decision-support system only. Anomaly flags are for human review.
          <br />
          This system does not autonomously trigger actions.
        </div>
      </div>
    </div>
  );
}
