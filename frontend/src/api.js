// api.js — All backend calls in one place.
// Vite exposes env vars via import.meta.env.VITE_*
// Set VITE_API_URL=http://localhost:8000 in your .env file for local dev.

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export const api = {
  chat:          (question, history = []) =>
                   req('/chat', { method: 'POST', body: JSON.stringify({ question, history }) }),
  services:      () => req('/services'),
  serviceDetail: (id) => req(`/services/${id}`),
  ingestion:     () => req('/about/ingestion'),
  health:        () => req('/health'),

  // Document Q&A — upload your own file, ask questions about just that file.
  // Only reachable when FEATURES.documentQA is enabled in config.js; the backend
  // /documents/* endpoints stay live either way.
  uploadDocument: (file) => uploadDocument(file),
  askDocument:    (docId, question) =>
                    req(`/documents/${docId}/ask`, { method: 'POST', body: JSON.stringify({ question }) }),
  documentMeta:   (docId) => req(`/documents/${docId}`),
};
