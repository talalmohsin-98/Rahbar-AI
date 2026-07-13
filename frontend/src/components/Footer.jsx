import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer style={{
      background: 'var(--navy)', color: 'var(--slate-light)',
      padding: '48px 0 32px', marginTop: 80,
    }}>
      <div className="container">
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 40, marginBottom: 48,
        }}>
          <div>
            <div style={{
              fontFamily: 'var(--font-head)', fontWeight: 700,
              fontSize: 18, color: 'var(--white)', marginBottom: 12,
            }}>Rahbar AI</div>
            <p style={{ fontSize: 13, lineHeight: 1.7 }}>
              Grounded, cited answers about Pakistani government services.
              Powered by RAG + MCP architecture.
            </p>
          </div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--white)', fontSize: 13,
                          textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 14 }}>
              Pages
            </div>
            {[['/', 'Home'], ['/services', 'Services'], ['/recommend', 'Get Help'],
              ['/assistant', 'Assistant'], ['/documents', 'Your Docs'], ['/about', 'About']].map(([to, label]) => (
              <Link key={to} to={to} style={{
                display: 'block', color: 'var(--slate-light)', fontSize: 14,
                marginBottom: 8, textDecoration: 'none',
              }}
              onMouseEnter={e => e.target.style.color = 'var(--green)'}
              onMouseLeave={e => e.target.style.color = 'var(--slate-light)'}
              >{label}</Link>
            ))}
          </div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--white)', fontSize: 13,
                          textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 14 }}>
              Services
            </div>
            {['NADRA / CNIC', 'FBR Tax Filing', 'Driving License', 'SECP Registration', 'Passport (DGIP)'].map(s => (
              <div key={s} style={{ fontSize: 14, marginBottom: 8 }}>{s}</div>
            ))}
          </div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--white)', fontSize: 13,
                          textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 14 }}>
              Architecture
            </div>
            {['RAG Pipeline', 'MCP Tools', 'LangGraph Agents', 'pgvector', 'Groq Inference'].map(s => (
              <div key={s} style={{ fontSize: 14, marginBottom: 8 }}>{s}</div>
            ))}
          </div>
        </div>
        <div style={{
          borderTop: '1px solid var(--navy-light)', paddingTop: 24,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexWrap: 'wrap', gap: 12, fontSize: 13,
        }}>
          <span>© 2025 Muhammad Talal Mohsin · Portfolio demonstration project</span>
          <span style={{ color: '#475569' }}>
            Not affiliated with NADRA, FBR, SECP, or any Pakistani government body
          </span>
        </div>
      </div>
    </footer>
  );
}
