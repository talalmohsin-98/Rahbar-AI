import React, { useState, useRef, useEffect } from 'react';
import { api } from '../api';
import { formatAnswer } from '../lib/chatFormatting';

function UploadZone({ onUploaded }) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const meta = await api.uploadDocument(file);
      onUploaded(meta);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ maxWidth: 560, margin: '0 auto', textAlign: 'center', paddingTop: 40 }}>
      <div style={{ fontSize: 40, marginBottom: 16 }}>📄</div>
      <h2 style={{ fontSize: 22, marginBottom: 8 }}>Ask questions about your own document</h2>
      <p style={{ color: 'var(--text-soft)', fontSize: 14, marginBottom: 28 }}>
        Upload a PDF, TXT, or Markdown file. Every answer will be grounded and cited
        against that document only — nothing is mixed in from the government-service corpora.
      </p>

      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? 'var(--green)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '40px 24px',
          background: dragOver ? '#F0FDF9' : 'var(--white)',
          cursor: 'pointer',
          transition: 'all var(--transition)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files?.[0])}
        />
        {uploading ? (
          <p style={{ fontSize: 14, color: 'var(--text-soft)' }}>Uploading and indexing…</p>
        ) : (
          <>
            <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
              Drop a file here, or click to browse
            </p>
            <p style={{ fontSize: 12, color: 'var(--slate)' }}>.pdf, .txt, or .md — up to 15 MB</p>
          </>
        )}
      </div>

      {error && (
        <p style={{ marginTop: 16, fontSize: 13, color: 'var(--error)' }}>{error}</p>
      )}

      <p style={{ marginTop: 24, fontSize: 12, color: 'var(--slate-light)' }}>
        Uploaded documents are kept in memory only, are not shared with other users,
        and are automatically discarded after a few hours.
      </p>
    </div>
  );
}

function Message({ msg, chunks }) {
  const isUser = msg.role === 'user';

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

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <div style={{
          width: 28, height: 28, borderRadius: 8, background: 'var(--navy)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 14, flexShrink: 0,
        }}>📄</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, lineHeight: 1.75, color: 'var(--text)' }}>
            {formatAnswer(msg.content)}
          </div>
        </div>
      </div>
    </div>
  );
}

const DOC_PIPELINE_STAGES = [
  'Searching your document…',
  'Reranking the best passages…',
  'Writing your answer…',
];

function TypingIndicator() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex(i => Math.min(i + 1, DOC_PIPELINE_STAGES.length - 1));
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 16 }}>
      <div style={{
        width: 28, height: 28, borderRadius: 8, background: 'var(--navy)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
      }}>📄</div>
      <div style={{
        background: 'var(--white)', border: '1px solid var(--border)',
        borderRadius: '4px 16px 16px 16px',
        padding: '12px 16px', display: 'flex', gap: 10, alignItems: 'center',
      }}>
        <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: 6, height: 6, borderRadius: '50%', background: 'var(--slate-light)',
              animation: `docqa-bounce 1.2s ${i * 0.2}s ease-in-out infinite`,
            }} />
          ))}
        </div>
        <span key={stageIndex} style={{ fontSize: 13, color: 'var(--text-soft)', animation: 'docqa-fadeIn 300ms ease-out' }}>
          {DOC_PIPELINE_STAGES[stageIndex]}
        </span>
      </div>
      <style>{`
        @keyframes docqa-bounce {
          0%, 80%, 100% { transform: scale(0.8); opacity: .5; }
          40% { transform: scale(1.1); opacity: 1; }
        }
        @keyframes docqa-fadeIn { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

export default function DocumentQA() {
  const [doc, setDoc] = useState(null);         // { doc_id, filename, chunk_count }
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesRef = useRef(null);

  // Scroll only the messages panel, not the page — scrollIntoView() on a
  // nested element can scroll ANY scrollable ancestor (or the whole window)
  // into view, which is what caused the page to jump down to the footer.
  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  const send = async (question) => {
    if (!question.trim() || loading || !doc) return;
    const q = question.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);

    try {
      const data = await api.askDocument(doc.doc_id, q);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer || 'No answer generated.',
        chunks: data.chunks_used || [],
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.message}`,
        chunks: [],
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => { e.preventDefault(); send(input); };

  const resetDocument = () => {
    setDoc(null);
    setMessages([]);
    setInput('');
  };

  const latestChunks = messages.filter(m => m.role === 'assistant').at(-1)?.chunks || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 60px)' }}>
      {/* Top bar */}
      <div style={{
        background: 'var(--white)', borderBottom: '1px solid var(--border)',
        padding: '12px 20px', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, background: 'var(--navy)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16,
          }}>📄</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>Document Q&A</div>
            <div style={{ fontSize: 12, color: 'var(--slate)' }}>
              {doc ? doc.filename : 'Upload a file to get started'}
            </div>
          </div>
        </div>
        {doc && (
          <button className="btn btn-outline" style={{ fontSize: 13, padding: '7px 14px' }} onClick={resetDocument}>
            Upload a different file
          </button>
        )}
      </div>

      {/* Body */}
      <div ref={messagesRef} style={{ flex: 1, overflowY: 'auto', padding: '24px 20px', background: 'var(--surface)' }}>
        {!doc ? (
          <UploadZone onUploaded={setDoc} />
        ) : (
          <>
            {messages.length === 0 && (
              <div style={{ maxWidth: 480, margin: '0 auto', textAlign: 'center', paddingTop: 40 }}>
                <div className="badge badge--green" style={{ marginBottom: 12 }}>
                  {doc.chunk_count} chunks indexed
                </div>
                <p style={{ color: 'var(--text-soft)', fontSize: 14 }}>
                  Ask anything about <strong>{doc.filename}</strong>.
                </p>
              </div>
            )}
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} chunks={msg.chunks || latestChunks} />
            ))}
            {loading && <TypingIndicator />}
          </>
        )}
      </div>

      {/* Input */}
      {doc && (
        <div style={{ background: 'var(--white)', borderTop: '1px solid var(--border)', padding: '16px 20px' }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 10, maxWidth: 800, margin: '0 auto' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={`Ask about ${doc.filename}...`}
              disabled={loading}
              style={{
                flex: 1, padding: '12px 16px', fontSize: 14,
                border: '1.5px solid var(--border)', borderRadius: 'var(--radius-md)',
                outline: 'none', background: 'var(--surface)',
              }}
            />
            <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}
              style={{ padding: '12px 20px', opacity: loading ? .6 : 1 }}>
              {loading ? '…' : 'Send'}
            </button>
          </form>
          <p style={{ textAlign: 'center', fontSize: 11, color: 'var(--slate-light)', marginTop: 10 }}>
            Answers grounded only in the uploaded document · Not legal advice
          </p>
        </div>
      )}
    </div>
  );
}
