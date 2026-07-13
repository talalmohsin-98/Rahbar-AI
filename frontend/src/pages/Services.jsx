import React, { useEffect, useState } from 'react';
import { api } from '../api';
import ServiceCard from '../components/ServiceCard';

export default function Services() {
  const [services, setServices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.services()
      .then(d => { setServices(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div>
      {/* Page header */}
      <div style={{
        background: 'var(--navy)', padding: '48px 0 52px',
        borderBottom: '1px solid var(--navy-light)',
      }}>
        <div className="container">
          <div className="badge badge--green" style={{ marginBottom: 14 }}>5 domains</div>
          <h1 style={{ fontSize: 36, color: 'var(--white)', marginBottom: 12 }}>
            Government Services
          </h1>
          <p style={{ fontSize: 16, color: 'rgba(255,255,255,.55)', maxWidth: 520 }}>
            Browse the five service domains this assistant covers. Click "Ask about this"
            on any card to open the assistant pre-loaded with that service's context.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container">
          {loading && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
              gap: 24,
            }}>
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="card skeleton" style={{ height: 180 }} />
              ))}
            </div>
          )}

          {error && (
            <div style={{
              padding: '24px', background: '#FEF2F2', border: '1px solid #FECACA',
              borderRadius: 'var(--radius-md)', color: '#B91C1C', fontSize: 14,
            }}>
              Could not load services: {error}. Make sure the backend is running.
            </div>
          )}

          {!loading && !error && (
            <>
              <p style={{ color: 'var(--text-soft)', fontSize: 14, marginBottom: 28 }}>
                {services.length} services available
              </p>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
                gap: 24,
              }}>
                {services.map(s => <ServiceCard key={s.id} service={s} />)}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
