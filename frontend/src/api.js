import axios from 'axios';

export const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const api = axios.create({ baseURL: API });
api.interceptors.request.use(config => {
  const token = localStorage.getItem('demo-token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const errorText = e => e.response?.data?.detail || e.message;

export const fmt = (value, digits = 3) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
