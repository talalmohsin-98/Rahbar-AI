# Rahbar AI — Frontend

React 19 + Vite + React Router 7. Five pages — Landing, Services, Recommend
(the deterministic wizard), Assistant (the RAG chat + Pipeline Inspector), and
About — talking to the FastAPI backend through a single wrapper in
[`src/api.js`](src/api.js).

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # tsc -b && vite build
npm run lint         # oxlint
```

The backend URL comes from `VITE_API_URL` and falls back to
`http://localhost:8000`, so `npm run dev` needs no env file when the backend is
running locally on its default port.

`src/pages/DocumentQA.jsx` is present but unrouted — see **Scope** in the
[root README](../README.md) and the `FEATURES` flag in
[`src/config.js`](src/config.js).
