import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import './index.css';
import { FEATURES } from './config';

import Nav        from './components/Nav';
import Footer     from './components/Footer';
import Disclaimer from './components/Disclaimer';

import Landing     from './pages/Landing';
import Services    from './pages/Services';
import Recommend   from './pages/Recommend';
import Assistant   from './pages/Assistant';
import DocumentQA  from './pages/DocumentQA';  // routed only when FEATURES.documentQA is on
import About       from './pages/About';

// These are full-height chat interfaces — a marketing footer (and the
// disclaimer banner eating into vertical space) doesn't belong under them.
// Every other page keeps the normal Disclaimer + Nav + content + Footer shell.
const CHAT_ROUTES = ['/assistant', ...(FEATURES.documentQA ? ['/documents'] : [])];

function Chrome({ children }) {
  const { pathname } = useLocation();
  const isChatPage = CHAT_ROUTES.includes(pathname);

  return (
    <>
      {!isChatPage && <Disclaimer />}
      <Nav />
      <main>{children}</main>
      {!isChatPage && <Footer />}
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Chrome>
        <Routes>
          <Route path="/"          element={<Landing />}   />
          <Route path="/services"  element={<Services />}  />
          <Route path="/recommend" element={<Recommend />} />
          <Route path="/assistant" element={<Assistant />} />
          {FEATURES.documentQA && <Route path="/documents" element={<DocumentQA />} />}
          <Route path="/about"     element={<About />}     />
          {/* Unknown paths — including /documents while the flag is off — go home
              rather than rendering an empty shell. */}
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
      </Chrome>
    </BrowserRouter>
  );
}
