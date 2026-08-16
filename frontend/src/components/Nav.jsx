import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FEATURES } from '../config';

// `feature` entries are dropped unless their flag is on in config.js — they keep
// their position in the order, so re-enabling one puts it back where it belongs.
const links = [
  { to: '/',          label: 'Home'       },
  { to: '/services',  label: 'Services'   },
  { to: '/recommend', label: 'Get Help'   },
  { to: '/assistant', label: 'Assistant'  },
  { to: '/documents', label: 'Your Docs', feature: 'documentQA' },
  { to: '/about',     label: 'About'      },
].filter(l => !l.feature || FEATURES[l.feature]);

export default function Nav() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <nav style={{
      background: 'var(--navy)',
      borderBottom: '1px solid var(--navy-light)',
      position: 'sticky', top: 0, zIndex: 50,
    }}>
      <div className="container" style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', height: 60,
      }}>
        {/* Logo */}
        <button
          onClick={() => navigate('/')}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: 10,
          }}
        >
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'var(--green)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'var(--font-head)', fontWeight: 700,
            fontSize: 16, color: 'var(--navy)',
          }}>R</div>
          <span style={{
            fontFamily: 'var(--font-head)', fontWeight: 600,
            fontSize: 17, color: 'var(--white)', letterSpacing: '-.3px',
          }}>Rahbar AI</span>
        </button>

        {/* Desktop links */}
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}
             className="nav-links">
          {links.map(l => (
            <NavLink key={l.to} to={l.to} end={l.to === '/'}
              style={({ isActive }) => ({
                padding: '6px 14px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 14, fontWeight: 500,
                color: isActive ? 'var(--green)' : 'var(--slate-light)',
                background: isActive ? 'rgba(0,200,150,.12)' : 'transparent',
                transition: 'all var(--transition)',
                textDecoration: 'none',
              })}
            >{l.label}</NavLink>
          ))}
          <button
            className="btn btn-primary"
            style={{ marginLeft: 8, padding: '7px 16px' }}
            onClick={() => navigate('/assistant')}
          >Ask a Question</button>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setOpen(o => !o)}
          style={{
            display: 'none', background: 'none', border: 'none',
            color: 'var(--white)', fontSize: 22, padding: 4,
          }}
          className="nav-hamburger"
          aria-label="Menu"
        >{open ? '✕' : '☰'}</button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div style={{
          background: 'var(--navy-mid)',
          borderTop: '1px solid var(--navy-light)',
          padding: '12px 24px 20px',
        }}>
          {links.map(l => (
            <NavLink key={l.to} to={l.to} end={l.to === '/'}
              onClick={() => setOpen(false)}
              style={({ isActive }) => ({
                display: 'block', padding: '10px 0',
                color: isActive ? 'var(--green)' : 'var(--slate-light)',
                fontSize: 15, fontWeight: 500, textDecoration: 'none',
                borderBottom: '1px solid var(--navy-light)',
              })}
            >{l.label}</NavLink>
          ))}
        </div>
      )}

      <style>{`
        @media (max-width: 768px) {
          .nav-links { display: none !important; }
          .nav-hamburger { display: block !important; }
        }
      `}</style>
    </nav>
  );
}
