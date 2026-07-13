import React, { useState } from 'react';

export default function CitationChip({ source, content, score }) {
  const [expanded, setExpanded] = useState(false);
  const label = source.split('/').pop().replace('.pdf', '');

  return (
    <span style={{ display: 'inline-block', position: 'relative' }}>
      <button
        className="citation-chip"
        onClick={() => setExpanded(e => !e)}
        title={`Source: ${source}`}
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M1 8V3a1 1 0 011-1h4.5L8 3.5V8a1 1 0 01-1 1H2a1 1 0 01-1-1z"
                stroke="currentColor" strokeWidth="1.2"/>
          <path d="M6 2v2h2" stroke="currentColor" strokeWidth="1.2"/>
        </svg>
        {label}
      </button>

      {expanded && (
        <div style={{
          position: 'absolute', bottom: '100%', left: 0,
          width: 300, background: 'var(--white)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
          padding: '14px 16px', boxShadow: 'var(--shadow-lg)',
          zIndex: 200, marginBottom: 6,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between',
                        alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: '#1D4ED8',
                           textTransform: 'uppercase', letterSpacing: '.4px' }}>
              {source}
            </span>
            {score !== undefined && (
              <span className="badge badge--green" style={{ fontSize: 10 }}>
                {score.toFixed(2)}
              </span>
            )}
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-soft)', lineHeight: 1.6, margin: 0 }}>
            {content || 'Source passage not available in preview.'}
          </p>
          <button
            onClick={() => setExpanded(false)}
            style={{
              marginTop: 10, fontSize: 12, color: 'var(--slate)',
              background: 'none', border: 'none', padding: 0,
              cursor: 'pointer', textDecoration: 'underline',
            }}
          >Close</button>
        </div>
      )}
    </span>
  );
}
