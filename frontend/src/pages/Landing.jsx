import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import ServiceCard from '../components/ServiceCard';

const HOW = [
  { n: 1, title: 'Ask in plain English', body: 'Type your question naturally — "What documents do I need to renew my CNIC?" — no special commands needed.' },
  { n: 2, title: 'Retrieved from official sources', body: 'Our pipeline searches official Pakistani government documents using hybrid semantic + keyword search.' },
  { n: 3, title: 'Grounded, cited answer', body: 'Every factual claim is traced to a real source. The system declines rather than guesses when unsure.' },
];

const FAQS = [
  { q: 'What is this?', a: 'A research demonstration of a Retrieval-Augmented Generation (RAG) system applied to Pakistani government services. Every answer is grounded in official public documents.' },
  { q: 'Is this an official government service?', a: 'No. This is not affiliated with NADRA, FBR, SECP, DGIP, or any Pakistani government body. Always verify critical information directly with the relevant authority.' },
  { q: 'Is my data stored?', a: 'No personally identifiable information is stored. Your question text is sent to our backend for processing and not retained after the response.' },
  { q: 'What if the system doesn\'t know the answer?', a: 'The system will explicitly say so rather than fabricating an answer. A clear decline is a feature, not a failure.' },
];

function FAQ({ q, a }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      borderBottom: '1px solid var(--border)', padding: '16px 0',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', background: 'none', border: 'none',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          textAlign: 'left', cursor: 'pointer', gap: 16,
        }}
      >
        <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text)' }}>{q}</span>
        <span style={{
          fontSize: 18, color: 'var(--green)', flexShrink: 0,
          transition: 'transform var(--transition)',
          transform: open ? 'rotate(45deg)' : 'none',
        }}>+</span>
      </button>
      {open && (
        <p style={{
          marginTop: 12, fontSize: 14, color: 'var(--text-soft)',
          lineHeight: 1.7, paddingRight: 32,
        }}>{a}</p>
      )}
    </div>
  );
}

export default function Landing() {
  const navigate = useNavigate();
  const [services, setServices] = useState([]);
  const [q, setQ] = useState('');

  useEffect(() => {
    api.services().then(d => setServices(d.slice(0, 5))).catch(() => {});
  }, []);

  const handleAsk = (e) => {
    e.preventDefault();
    if (q.trim()) navigate(`/assistant?q=${encodeURIComponent(q.trim())}`);
    else navigate('/assistant');
  };

  return (
    <div>
      {/* ── Hero ── */}
      <div className="bg-grid-dots" style={{
        background: `linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%)`,
        padding: '80px 0 90px', position: 'relative', overflow: 'hidden',
      }}>
        {/* Decorative circle */}
        <div style={{
          position: 'absolute', top: -80, right: -80, width: 400, height: 400,
          borderRadius: '50%', background: 'rgba(0,200,150,.06)', pointerEvents: 'none',
        }} />
        <div className="container" style={{ position: 'relative' }}>
          <div style={{ maxWidth: 640 }}>
            <div className="badge badge--green" style={{ marginBottom: 20 }}>
              Grounded · Cited · Open Source
            </div>
            <h1 style={{
              fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 700,
              color: 'var(--white)', lineHeight: 1.15, marginBottom: 20,
            }}>
              Your guide to Pakistani<br />
              <span style={{ color: 'var(--green)' }}>government services</span>
            </h1>
            <p style={{
              fontSize: 17, color: 'rgba(255,255,255,.65)',
              lineHeight: 1.7, marginBottom: 36, maxWidth: 520,
            }}>
              Ask about CNIC renewal, tax filing, passport applications, and more.
              Every answer is grounded in official documents with source citations.
            </p>

            {/* Inline search */}
            <form onSubmit={handleAsk} style={{
              display: 'flex', gap: 10, maxWidth: 540,
              background: 'rgba(255,255,255,.08)', padding: 6,
              borderRadius: 'var(--radius-md)',
              border: '1px solid rgba(255,255,255,.15)',
            }}>
              <input
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder="e.g. What documents do I need for a CNIC?"
                style={{
                  flex: 1, background: 'none', border: 'none', outline: 'none',
                  color: 'var(--white)', fontSize: 14, padding: '8px 12px',
                }}
              />
              <button type="submit" className="btn btn-primary" style={{ flexShrink: 0 }}>
                Ask →
              </button>
            </form>

            <div style={{
              marginTop: 16, display: 'flex', gap: 24, flexWrap: 'wrap',
            }}>
              {['CNIC renewal', 'Tax filing guide', 'Passport application'].map(s => (
                <button key={s} onClick={() => navigate(`/assistant?q=${encodeURIComponent(s)}`)}
                  style={{
                    background: 'none', border: 'none', color: 'rgba(255,255,255,.5)',
                    fontSize: 13, cursor: 'pointer', textDecoration: 'underline',
                    padding: 0,
                  }}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div style={{
            position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
            display: 'flex', flexDirection: 'column', gap: 20,
          }} className="hero-stats">
            {[['5', 'Services covered'], ['40+', 'Source documents'], ['0%', 'Fabrication tolerance']].map(([n, l]) => (
              <div key={l} style={{
                background: 'rgba(255,255,255,.06)', border: '1px solid rgba(255,255,255,.1)',
                borderRadius: 'var(--radius-md)', padding: '16px 24px', textAlign: 'center',
              }}>
                <div style={{ fontFamily: 'var(--font-head)', fontSize: 28, fontWeight: 700, color: 'var(--green)' }}>{n}</div>
                <div style={{ fontSize: 12, color: 'rgba(255,255,255,.45)', marginTop: 2 }}>{l}</div>
              </div>
            ))}
          </div>
        </div>
        <style>{`@media (max-width: 900px) { .hero-stats { display: none !important; } }`}</style>
      </div>

      {/* ── Services preview ── */}
      <div className="section">
        <div className="container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 32 }}>
            <div>
              <div className="badge badge--navy" style={{ marginBottom: 10 }}>5 service domains</div>
              <h2 style={{ fontSize: 28 }}>What can I help with?</h2>
            </div>
            <button className="btn btn-outline" onClick={() => navigate('/services')}>
              Browse all →
            </button>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: 20,
          }}>
            {services.length > 0
              ? services.map(s => <ServiceCard key={s.id} service={s} compact />)
              : Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="card skeleton" style={{ height: 120 }} />
                ))
            }
          </div>
        </div>
      </div>

      {/* ── How it works ── */}
      <div style={{ background: 'var(--navy)', padding: '72px 0' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div className="badge badge--green" style={{ marginBottom: 12 }}>Transparent by design</div>
            <h2 style={{ fontSize: 28, color: 'var(--white)' }}>How it works</h2>
          </div>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 32,
          }}>
            {HOW.map(({ n, title, body }) => (
              <div key={n} style={{ textAlign: 'center' }}>
                <div style={{
                  width: 48, height: 48, borderRadius: '50%',
                  background: 'rgba(0,200,150,.15)',
                  border: '1px solid var(--green)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 18,
                  color: 'var(--green)', margin: '0 auto 20px',
                }}>{n}</div>
                <h3 style={{ fontSize: 17, color: 'var(--white)', marginBottom: 10 }}>{title}</h3>
                <p style={{ fontSize: 14, color: 'rgba(255,255,255,.55)', lineHeight: 1.7 }}>{body}</p>
              </div>
            ))}
          </div>
          <div style={{ textAlign: 'center', marginTop: 48 }}>
            <button className="btn btn-primary" style={{ padding: '12px 28px', fontSize: 15 }}
              onClick={() => navigate('/assistant')}>
              Try the Assistant →
            </button>
          </div>
        </div>
      </div>

      {/* ── Trust ── */}
      <div className="section--sm">
        <div className="container" style={{ maxWidth: 720 }}>
          <div style={{
            background: 'var(--white)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)', padding: '32px 36px',
            borderLeft: '4px solid var(--green)',
          }}>
            <h3 style={{ fontSize: 18, marginBottom: 12 }}>Grounded answers only</h3>
            <p style={{ fontSize: 14, color: 'var(--text-soft)', lineHeight: 1.7 }}>
              Every factual claim in an answer is traced to a specific retrieved document chunk.
              When the system cannot find reliable information, it explicitly says so rather than
              fabricating a response. Cite-ability is not optional — it is the core design constraint.
            </p>
          </div>
        </div>
      </div>

      {/* ── FAQ ── */}
      <div className="section--sm">
        <div className="container" style={{ maxWidth: 720 }}>
          <h2 style={{ fontSize: 26, marginBottom: 32 }}>Frequently asked questions</h2>
          {FAQS.map(f => <FAQ key={f.q} {...f} />)}
        </div>
      </div>
    </div>
  );
}
