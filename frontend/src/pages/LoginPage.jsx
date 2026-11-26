/**
 * Login & Registration Page — FR-007: Database Authentication & Session Handling
 *
 * Connects directly to FastAPI backend with real Neon PostgreSQL storage,
 * salted Bcrypt password hashing, and signed HMAC-SHA256 JWT tokens.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login, register } from '../services/api';

export default function LoginPage() {
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'register'
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState('operator');
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccessMsg('');

    try {
      const res = await login(username, password);
      const { access_token, user } = res.data;
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_role', user?.role === 'admin' ? 'System Admin' : 'Plant Operator');
      localStorage.setItem('username', user?.username || username);
      navigate('/');
    } catch (err) {
      if (err.response?.status === 401) {
        setError('Invalid username or password. Please verify your credentials.');
      } else {
        setError(err.response?.data?.detail?.message || err.response?.data?.message || 'Authentication failed. Please check backend connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccessMsg('');

    if (password.length < 4) {
      setError('Password must be at least 4 characters long.');
      setLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      setLoading(false);
      return;
    }

    try {
      const res = await register(username, email, password, role);
      const { access_token, user } = res.data;
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_role', user?.role === 'admin' ? 'System Admin' : 'Plant Operator');
      localStorage.setItem('username', user?.username || username);
      setSuccessMsg('Account created successfully! Redirecting...');
      setTimeout(() => navigate('/'), 800);
    } catch (err) {
      if (err.response?.status === 409) {
        setError(err.response?.data?.detail?.message || 'Username or email is already registered.');
      } else {
        setError(err.response?.data?.detail?.message || err.response?.data?.message || 'Registration failed.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card" style={{ maxWidth: '440px' }}>
        {/* ── Logo ─────────────────────────── */}
        <div style={{ textAlign: 'center', marginBottom: '20px' }}>
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
              marginBottom: '14px',
            }}
          >
            ⚡
          </div>
          <h1 className="login-title">AnomalyGuard</h1>
          <p className="login-subtitle">
            Industrial IoT Sensor Anomaly Monitoring Platform
          </p>
        </div>

        {/* ── Auth Tabs ────────────────────── */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '4px',
          background: 'rgba(255,255,255,0.04)',
          padding: '4px',
          borderRadius: '10px',
          marginBottom: '20px',
          border: '1px solid rgba(255,255,255,0.08)'
        }}>
          <button
            type="button"
            onClick={() => { setActiveTab('login'); setError(''); }}
            style={{
              padding: '8px 12px',
              borderRadius: '7px',
              border: 'none',
              background: activeTab === 'login' ? '#6366f1' : 'transparent',
              color: activeTab === 'login' ? '#ffffff' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => { setActiveTab('register'); setError(''); }}
            style={{
              padding: '8px 12px',
              borderRadius: '7px',
              border: 'none',
              background: activeTab === 'register' ? '#6366f1' : 'transparent',
              color: activeTab === 'register' ? '#ffffff' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            Register Account
          </button>
        </div>

        {/* ── Alerts ───────────────────────── */}
        {error && (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.35)',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '13px',
              color: '#ef4444',
              marginBottom: '18px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {successMsg && (
          <div
            style={{
              background: 'rgba(16, 185, 129, 0.12)',
              border: '1px solid rgba(16, 185, 129, 0.35)',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '13px',
              color: '#10b981',
              marginBottom: '18px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span>✅</span>
            <span>{successMsg}</span>
          </div>
        )}

        {/* ── Sign In Form ─────────────────── */}
        {activeTab === 'login' && (
          <form onSubmit={handleLoginSubmit}>
            <div className="form-group">
              <label htmlFor="username" className="form-label">Username or Email</label>
              <input
                id="username"
                type="text"
                className="form-input"
                placeholder="Enter your username or email"
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
                placeholder="Enter your password"
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
              style={{ width: '100%', justifyContent: 'center', marginTop: '10px', padding: '12px' }}
            >
              {loading ? 'Authenticating with Database...' : 'Sign In'}
            </button>
          </form>
        )}

        {/* ── Sign Up Form ─────────────────── */}
        {activeTab === 'register' && (
          <form onSubmit={handleRegisterSubmit}>
            <div className="form-group">
              <label htmlFor="reg-username" className="form-label">Choose Username</label>
              <input
                id="reg-username"
                type="text"
                className="form-input"
                placeholder="e.g. john_doe"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                autoFocus
                autoComplete="username"
              />
            </div>

            <div className="form-group">
              <label htmlFor="reg-email" className="form-label">Work Email</label>
              <input
                id="reg-email"
                type="email"
                className="form-input"
                placeholder="name@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>

            <div className="form-group">
              <label htmlFor="reg-role" className="form-label">Operational Role</label>
              <select
                id="reg-role"
                className="form-input"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                style={{ cursor: 'pointer' }}
              >
                <option value="operator">Plant Operator (Monitor & Alert Review)</option>
                <option value="admin">System Engineer / Administrator (Full Access)</option>
                <option value="viewer">Auditor / Viewer (Read Only)</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="reg-password" className="form-label">Create Password</label>
              <input
                id="reg-password"
                type="password"
                className="form-input"
                placeholder="Minimum 4 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="new-password"
              />
            </div>

            <div className="form-group">
              <label htmlFor="reg-confirm" className="form-label">Confirm Password</label>
              <input
                id="reg-confirm"
                type="password"
                className="form-input"
                placeholder="Repeat password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
              />
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ width: '100%', justifyContent: 'center', marginTop: '10px', padding: '12px' }}
            >
              {loading ? 'Creating Account in Database...' : 'Register Account'}
            </button>
          </form>
        )}

        {/* ── Demo Credentials Provenance Note ── */}
        <div style={{
          marginTop: '20px',
          background: 'rgba(255,255,255,0.02)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '11px',
          color: 'var(--text-muted)'
        }}>
          <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
            🔒 Production PostgreSQL Authentication:
          </div>
          <div>
            Credentials are securely hashed with bcrypt. Pre-seeded database accounts available for immediate evaluation:
            <span style={{ color: '#818cf8', fontFamily: 'monospace' }}> admin / admin</span> or
            <span style={{ color: '#818cf8', fontFamily: 'monospace' }}> operator / operator</span>.
          </div>
        </div>

        {/* ── Disclaimer ───────────────────── */}
        <div
          style={{
            marginTop: '20px',
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
