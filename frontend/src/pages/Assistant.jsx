import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../api';
import { formatAnswer } from '../lib/chatFormatting';

function Message({ msg }) {
  const isUser = msg.role === 'user';
  const isDecline = msg.declined;

  if (isUser) return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
      <div style={{
        background: 'var(--navy)', color: 'var(--white)',
        borderRadius: '16px 16px 4px 16px',
        padding: '12px 16px', maxWidth: '75%', fontSize: 14, lineHeight: 1.6,
      }}>
        {msg.content}
      </div>
    </div>
  );

  if (isDecline) return (
    <div style={{ marginBottom: 16 }}>
      <div className="declined-state">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M8 5v3.5M8 11h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: 'var(--text)' }}>
            Insufficient grounding — declined
          </div>
          <p style={{ margin: 0, fontSize: 13 }}>{msg.content}</p>
        </div>
      </div>
    </div>
  );

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <div style={{
          width: 28, height: 28, borderRadius: 8, background: 'var(--green)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 13,
          color: 'var(--navy)', flexShrink: 0,
        }}>R</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, lineHeight: 1.75, color: 'var(--text)' }}>
            {formatAnswer(msg.content)}
          </div>

          {msg.pipelineData && (
            <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <span className="badge badge--green">
                {msg.pipelineData.citation_result?.passed !== false ? '✓ Citations verified' : '⚠ Citations flagged'}
              </span>
              <span className="badge badge--slate">
                Hal rate: {Math.round((msg.pipelineData.hallucination_result?.hallucination_rate ?? 0) * 100)}%
              </span>
              <span className="badge badge--navy">
                {msg.pipelineData.intent}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const PIPELINE_STAGES = [
  'Understanding your question…',
  'Searching official documents…',
  'Reranking the best matches…',
  'Compressing relevant passages…',
  'Writing your answer…',
  'Verifying citations…',
];

function TypingIndicator() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex(i => Math.min(i + 1, PIPELINE_STAGES.length - 1));
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 16 }}>
      <div style={{
        width: 28, height: 28, borderRadius: 8, background: 'var(--green)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 13,
        color: 'var(--navy)', flexShrink: 0,
      }}>R</div>
      <div style={{
        background: 'var(--white)', border: '1px solid var(--border)',
        borderRadius: '4px 16px 16px 16px',
        padding: '12px 16px', display: 'flex', gap: 10, alignItems: 'center',
      }}>
        <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
          {[0,1,2].map(i => (
            <div key={i} style={{
              width: 6, height: 6, borderRadius: '50%', background: 'var(--green-dim)',
              animation: `bounce 1.2s ${i * 0.2}s ease-in-out infinite`,
            }} />
          ))}
        </div>
        <span
          key={stageIndex}
          style={{
            fontSize: 13, color: 'var(--text-soft)',
            animation: 'fadeIn 300ms ease-out',
          }}
        >
          {PIPELINE_STAGES[stageIndex]}
        </span>
      </div>
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0.8); opacity: .5; }
          40% { transform: scale(1.1); opacity: 1; }
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

const SUGGESTIONS = [
  'What documents do I need for CNIC registration?',
  'How do I file my income tax return online?',
  'What is the process for getting a driving license?',
  'How do I register a company with SECP?',
  'What are the passport renewal requirements?',
];

export default function Assistant() {
  const [searchParams] = useSearchParams();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesRef = useRef(null);
  const inputRef = useRef(null);

  // Pre-seed from query param (handoff from Services/Wizard)
  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setInput(q);
      inputRef.current?.focus();
    }
  }, [searchParams]);

  // Scroll only the messages panel itself, not the page. scrollIntoView() on
  // a nested element can walk up and scroll ANY scrollable ancestor (or the
  // whole window) into view — which is what was pulling the page down to the
  // footer. Setting scrollTop directly on the messages container is scoped
  // to just that element.
  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  const send = async (question) => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const data = await api.chat(q, history);

      const isDecline = data.intent === 'out_of_scope' ||
        (!data.final_answer && !data.answer);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.final_answer || data.answer || 'No answer generated.',
        declined: isDecline,
        pipelineData: data,
        chunks: data.compressed_chunks || [],
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.message}. Make sure the backend is running on port 8000.`,
        declined: false,
        pipelineData: null,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 60px)', position: 'relative' }}>
      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Top bar */}
        <div style={{
          background: 'var(--white)', borderBottom: '1px solid var(--border)',
          padding: '12px 20px', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10, background: 'var(--navy)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ fontFamily: 'var(--font-head)', fontWeight: 700,
                             fontSize: 16, color: 'var(--green)' }}>R</span>
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 15 }}>Rahbar AI Assistant</div>
              <div style={{ fontSize: 12, color: 'var(--slate)' }}>
                Grounded answers · Citation verified
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div ref={messagesRef} style={{
          flex: 1, overflowY: 'auto', padding: '24px 20px',
          background: 'var(--surface)',
        }}>
          {messages.length === 0 && (
            <div style={{ maxWidth: 560, margin: '0 auto', textAlign: 'center', paddingTop: 40 }}>
              <div style={{ fontSize: 40, marginBottom: 16 }}>🏛️</div>
              <h2 style={{ fontSize: 22, marginBottom: 8 }}>Ask about Pakistani government services</h2>
              <p style={{ color: 'var(--text-soft)', fontSize: 14, marginBottom: 32 }}>
                Every answer is grounded in official documents with source citations.
                Try one of these:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {SUGGESTIONS.map(s => (
                  <button key={s} onClick={() => send(s)}
                    style={{
                      background: 'var(--white)', border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)', padding: '12px 16px',
                      textAlign: 'left', cursor: 'pointer', fontSize: 14,
                      color: 'var(--text)', transition: 'all var(--transition)',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--green)'; e.currentTarget.style.background = '#F0FDF9'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--white)'; }}
                  >{s}</button>
                ))}
              </div>
            </div>
          )}

          {/* Same 800px cap as the input form below, so messages and the
              composer share one column instead of bubbles stretching across
              the whole viewport on wide screens. */}
          <div style={{ maxWidth: 800, margin: '0 auto' }}>
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && <TypingIndicator />}
          </div>
        </div>

        {/* Input */}
        <div style={{
          background: 'var(--white)', borderTop: '1px solid var(--border)',
          padding: '16px 20px',
        }}>
          <form onSubmit={handleSubmit} style={{
            display: 'flex', gap: 10, maxWidth: 800, margin: '0 auto',
          }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask about CNIC, tax filing, passport, driving license..."
              disabled={loading}
              style={{
                flex: 1, padding: '12px 16px', fontSize: 14,
                border: '1.5px solid var(--border)', borderRadius: 'var(--radius-md)',
                outline: 'none', background: 'var(--surface)',
                transition: 'border-color var(--transition)',
              }}
              onFocus={e => e.target.style.borderColor = 'var(--green)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
            <button type="submit" className="btn btn-primary"
              disabled={loading || !input.trim()}
              style={{ padding: '12px 20px', opacity: loading ? .6 : 1 }}>
              {loading ? '…' : 'Send'}
            </button>
          </form>
          <p style={{
            textAlign: 'center', fontSize: 11, color: 'var(--slate-light)',
            marginTop: 10,
          }}>
            Answers grounded in official Pakistani government documents · Not legal advice
          </p>
        </div>
      </div>
    </div>
  );
}
