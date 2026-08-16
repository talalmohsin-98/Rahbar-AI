// config.js — Frontend feature flags.
//
// Rahbar AI's scope is assisting citizens with five government service domains:
// NADRA / CNIC, FBR tax filing, Driving License, SECP registration, and
// Passport (DGIP). Surfaces outside that scope are switched off here rather
// than deleted, so nothing has to be rebuilt if the scope ever widens again.
//
// documentQA — the "Your Docs" upload-your-own-file RAG surface. Off: asking
// questions about a citizen's private file is a different product from guiding
// them through a government procedure. The page component
// (pages/DocumentQA.jsx), the API wrappers (api.js), and the whole backend
// (/documents/* in main.py, document_qa.py) are untouched and still work —
// flipping this to true restores the nav link, the footer link, and the
// /documents route exactly as they were.

export const FEATURES = {
  documentQA: false,
};
