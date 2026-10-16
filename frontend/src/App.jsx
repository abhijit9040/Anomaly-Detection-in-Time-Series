/**
 * App — Root application component with routing.
 *
 * Layout structure:
 * - Login page: standalone (no sidebar)
 * - Protected pages: sidebar + main content area (requires auth token)
 */

import { BrowserRouter, Routes, Route, Navigate, Outlet, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import LoginPage from './pages/LoginPage';
import SensorsPage from './pages/SensorsPage';
import EvaluationPage from './pages/EvaluationPage';
import ModelInfoPage from './pages/ModelInfoPage';
import SettingsPage from './pages/SettingsPage';

/**
 * Route protection wrapper — FR-007.
 * Enforces session token existence, redirects unauthenticated users to /login.
 */
function ProtectedLayout() {
  const token = localStorage.getItem('auth_token');
  const location = useLocation();

  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Login — standalone page (no sidebar) */}
        <Route path="/login" element={<LoginPage />} />

        {/* Authenticated routes with sidebar layout */}
        <Route element={<ProtectedLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/sensors" element={<SensorsPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/model-info" element={<ModelInfoPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>

        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
