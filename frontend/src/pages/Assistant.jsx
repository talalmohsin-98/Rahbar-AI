import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../api';
import { formatAnswer } from '../lib/chatFormatting';
import PipelinePanel from '../components/PipelinePanel';

/**
 * Trust badges under an answer.
 *
 * These must never claim more than the pipeline actually checked, and they
 * have now been wrong in two different ways.
 *
 * First: a green "✓ Citations verified" rendered whenever `passed !== false`,
 * which included the verifier parsing ZERO citations and checking nothing.
 *
 * Second, and worse: the badges only ever consulted two of the three checks.
 * `completeness_result` — the check that asks whether the answer actually used
 * what retrieval found — was computed, logged, shown in the inspector, and
 * ignored here. An answer the pipeline had FLAGGED still wore full green.
 *
 * The fix is not more logic in this file. It is that the backend now emits a
 * single `verification_verdict` and this component renders it. Deriving a
 * verdict in the UI is what allowed the UI to disagree with the pipeline.
 */
function AnswerBadges({ data }) {
  const verdict = data.verification_verdict;
  const cit = data.citation_result;
  const hal = data.hallucination_result;

  // THE GATE. Everything below reads this one field rather than re-deriving a
  // verdict from the individual checks. The bug this replaces: these badges
  // read citation_result and hallucination_result and never looked at
  // completeness_result, so "What is Gamma Family FRC?" — which the pipeline
  // FLAGGED for using almost none of what it retrieved — still rendered as
  // full green "2/2 claims grounded · 2 citations verified".
  //
  // No verdict means an older/partial payload, and the safe reading of "I
  // don't know" is never green.
  const status = verdict?.status ?? 'unverified';
  const isVerified = status === 'verified';

  const verdictBadge =
    status === 'verified' ? { cls: 'badge--green',  text: '✓ Grounded' }
  : status === 'partial'  ? { cls: 'badge--yellow', text: `⚠ ${verdict.headline}` }
  :                         { cls: 'badge--slate',  text: 'Not verified' };

  // Detail badges keep their counts either way — the numbers are still true
  // and still useful. What they lose on a flagged answer is the green: a
  // count is a fact about one check, not a verdict on the answer.
  const checked    = cit?.checked_count ?? 0;
  const unverified = cit?.unverified_claims?.length ?? 0;
  const citStatus  = cit?.status ?? (cit?.passed === false ? 'flagged' : 'unverifiable');

  const citBadge =
    citStatus === 'verified'   ? { cls: isVerified ? 'badge--green' : 'badge--slate',
                                   text: `${checked} citation${checked === 1 ? '' : 's'} checked` }
  : citStatus === 'flagged'    ? { cls: 'badge--yellow', text: `⚠ ${unverified} of ${checked} citations unverified` }
  : citStatus === 'not_applicable' ? null
  :                              { cls: 'badge--slate', text: 'Citations not verified' };

  const evaluated = hal?.evaluated_count ?? 0;
  const halBadge =
    hal?.status !== 'evaluated' || evaluated === 0
      ? { cls: 'badge--slate', text: 'Grounding not checked' }
      : { cls: hal.hallucinated_count > 0 ? 'badge--yellow'
                                          : (isVerified ? 'badge--green' : 'badge--slate'),
          text: `${hal.grounded_count}/${evaluated} claims grounded` };

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span className={`badge ${verdictBadge.cls}`}>{verdictBadge.text}</span>
        {citBadge && <span className={`badge ${citBadge.cls}`}>{citBadge.text}</span>}
        <span className={`badge ${halBadge.cls}`}>{halBadge.text}</span>
        {data.generation_meta?.truncated && (
          <span className="badge badge--yellow">⚠ Answer cut off</span>
        )}
        <span className="badge badge--navy">{data.intent}</span>
      </div>

      {/* Why it isn't green, in the citizen's terms. A warning colour with no
          explanation just makes someone distrust the whole tool; naming the
          gap tells them what to do about it (ask more specifically, or go
          verify this particular line). */}
      {status !== 'verified' && verdict?.reasons?.length > 0 && (
        <ul style={{
          margin: '8px 0 0', paddingLeft: 18, listStyle: 'disc',
          fontSize: 12.5, lineHeight: 1.6, color: 'var(--text-soft)',
        }}>
          {verdict.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
    </div>
  );
}

function Message({ msg, onInspect }) {
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

          {msg.pipelineData && <AnswerBadges data={msg.pipelineData} />}

          {/* The badges above are the summary; this opens the full trace for
              THIS answer, so an older message in the thread can still be
              inspected after newer ones have arrived. */}
          {msg.pipelineData && (
            <button
              onClick={() => onInspect?.(msg.pipelineData)}
              style={{
                marginTop: 10, background: 'none', border: 'none', padding: 0,
                cursor: 'pointer', fontSize: 12, fontWeight: 600,
                color: 'var(--slate)', textDecoration: 'underline',
                textUnderlineOffset: 3,
              }}
            >
              Inspect pipeline →
            </button>
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

  // Pipeline Inspector. `panelData` is the trace being shown, which defaults to
  // the most recent answer but sticks to whichever answer the user inspected.
  const [panelOpen, setPanelOpen] = useState(false);
  const [panelData, setPanelData] = useState(null);

  const inspect = (data) => {
    setPanelData(data);
    setPanelOpen(true);
  };

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
      // Keep the inspector pointed at the newest answer. If it's already open,
      // it refreshes in place rather than showing a stale trace.
      setPanelData(data);
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

          <button
            onClick={() => setPanelOpen(o => !o)}
            className="btn btn-outline"
            style={{ padding: '8px 14px', fontSize: 13 }}
          >
            {panelOpen ? '✕ Hide pipeline' : '⚙ Pipeline Inspector'}
          </button>
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
              <Message key={i} msg={msg} onInspect={inspect} />
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

      {/* Every stage of the run that produced the answer. The panel is fixed to
          the right edge and slides in, so it overlays the chat rather than
          reflowing it mid-conversation. */}
      <PipelinePanel
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        data={panelData}
      />
    </div>
  );
}
