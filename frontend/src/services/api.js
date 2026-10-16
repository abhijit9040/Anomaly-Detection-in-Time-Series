/**
 * Axios API client for the Anomaly Detection backend.
 *
 * Base URL points to the FastAPI backend.
 * All API calls go through this client for consistent
 * error handling and auth token management.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// ── Request Interceptor: attach auth token ──
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response Interceptor: handle errors ──
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid — redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ═══════════════════════════════════════════════
// API Methods
// ═══════════════════════════════════════════════

/**
 * Health check — FR-008
 */
export const checkHealth = () => apiClient.get('/health');

/**
 * Detect anomaly — FR-003
 * @param {Object} sensorWindow - { data: number[][], machine_id?: string }
 */
export const detectAnomaly = (sensorWindow) =>
  apiClient.post('/api/v1/detect', sensorWindow);

/**
 * Batch anomaly detection — FR-012
 * @param {Array} windows - Array of { data: number[][] }
 */
export const detectBatch = (windows) =>
  apiClient.post('/api/v1/detect/batch', { windows });

/**
 * Get model metadata — FR-006
 */
export const getModelMetadata = () => apiClient.get('/api/v1/model');

/**
 * Get stream data point — FR-004
 */
export const getStreamData = () => apiClient.get('/api/v1/stream');

/**
 * Get evaluation report — FR-005
 */
export const getEvaluationReport = () => apiClient.get('/api/v1/evaluation');

/**
 * Login — FR-007
 * @param {string} username
 * @param {string} password
 */
export const login = (username, password) =>
  apiClient.post('/api/v1/auth/login', { username, password });

export default apiClient;
