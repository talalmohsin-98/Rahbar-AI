import React from 'react';
import { useNavigate } from 'react-router-dom';

const ICONS = {
  nadra:    '🪪',
  fbr:      '📋',
  license:  '🚗',
  secp:     '🏢',
  passport: '📘',
};

export default function ServiceCard({ service, compact = false }) {
  const navigate = useNavigate();

  const handleAsk = () => {
    navigate(`/assistant?q=${encodeURIComponent(`Tell me about ${service.name}`)}`);
  };

  return (
    <div className="card" style={{
      padding: compact ? '20px 24px' : '28px',
      display: 'flex', flexDirection: 'column', gap: 12,
      transition: 'box-shadow var(--transition), transform var(--transition)',
      cursor: 'default',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.boxShadow = 'var(--shadow-md)';
      e.currentTarget.style.transform = 'translateY(-2px)';
    }}
    onMouseLeave={e => {
      e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
      e.currentTarget.style.transform = 'translateY(0)';
    }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
        <div style={{
          fontSize: compact ? 28 : 36,
          lineHeight: 1, flexShrink: 0,
        }}>{ICONS[service.id] || '📄'}</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <h3 style={{ fontSize: compact ? 15 : 17, margin: 0 }}>{service.name}</h3>
            <span className="badge badge--navy" style={{ fontSize: 10 }}>
              {service.department}
            </span>
          </div>
          <p style={{
            fontSize: 13, color: 'var(--text-soft)', marginTop: 6, lineHeight: 1.6,
          }}>{service.description}</p>
        </div>
      </div>

      {!compact && service.tags && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {service.tags.map(t => (
            <span key={t} className="badge badge--slate">{t}</span>
          ))}
        </div>
      )}

      <button
        className="btn btn-outline"
        onClick={handleAsk}
        style={{ alignSelf: 'flex-start', fontSize: 13, padding: '7px 14px' }}
      >
        Ask about this →
      </button>
    </div>
  );
}
