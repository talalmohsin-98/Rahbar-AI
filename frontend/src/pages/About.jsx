import React, { useEffect, useState } from 'react';
import { api } from '../api';

const PIPELINE_STAGES = [
  { icon: '🧠', label: 'Intent Router', desc: 'Classifies the question as factual, comparison, procedural, or out-of-scope. Determines answer format and retrieval strategy.' },
  { icon: '✍️', label: 'Query Rewriter', desc: 'Generates 3–4 search queries per question: the original, a HyDE (Hypothetical Document Embedding) passage, and sub-questions for multi-aspect coverage.' },
  { icon: '🔍', label: 'Hybrid Search', desc: 'Runs dense vector search (pgvector + BAAI/bge-large-en) and BM25 keyword search in parallel. Fuses results with Reciprocal Rank Fusion (RRF).' },
  { icon: '⚖️', label: 'CrossEncoder Reranker', desc: 'Re-scores retrieved chunks using ms-marco-MiniLM-L-6-v2, which reads the query and chunk together. Filters below MIN_RERANK_SCORE = 2.8.' },
  { icon: '✂️', label: 'Context Compressor', desc: 'Scores each sentence in each chunk against the question. Keeps the top-4 most relevant sentences per chunk within a 1,200-token budget.' },
  { icon: '💬', label: 'Generator', desc: 'Calls Groq (llama3-70b-8192) with compressed context and intent-specific formatting instructions. Temperature 0.1 for factual accuracy.' },
  { icon: '📎', label: 'Citation Verifier', desc: 'Checks each [source: file] citation in the answer against its chunk using the CrossEncoder. Flags claims not supported by the cited source.' },
  { icon: '🔬', label: 'Hallucination Evaluator', desc: 'Scores every sentence in the answer against all chunks. Flags sentences with no chunk support. Triggers retry at temperature=0.0 if rate >20%.' },
];

function IngestionTable({ data }) {
  if (!data || !data.services) return (
    <div style={{ color: 'var(--text-soft)', fontSize: 14 }}>
      Loading ingestion metadata…
    </div>
  );
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead>
          <tr style={{ background: 'var(--surface)' }}>
            {['Service', 'Department', 'Documents', 'Chunks', 'Last Ingested'].map(h => (
              <th key={h} style={{
                padding: '10px 14px', textAlign: 'left',
                borderBottom: '2px solid var(--border)',
                fontWeight: 600, fontSize: 12,
                color: 'var(--slate)', textTransform: 'uppercase', letterSpacing: '.5px',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.services.map((s, i) => (
            <tr key={s.id} style={{ background: i % 2 === 0 ? 'var(--white)' : 'var(--surface)' }}>
              <td style={{ padding: '12px 14px', fontWeight: 500 }}>{s.name}</td>
              <td style={{ padding: '12px 14px', color: 'var(--text-soft)' }}>{s.department}</td>
              <td style={{ padding: '12px 14px' }}>{s.doc_count}</td>
              <td style={{ padding: '12px 14px' }}>{s.chunk_count}</td>
              <td style={{ padding: '12px 14px', color: 'var(--text-soft)' }}>{s.last_ingested}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr style={{ background: 'var(--surface)', fontWeight: 600 }}>
            <td colSpan={2} style={{ padding: '10px 14px', borderTop: '2px solid var(--border)' }}>
              Total
            </td>
            <td style={{ padding: '10px 14px', borderTop: '2px solid var(--border)' }}>
              {data.services.reduce((a, s) => a + (s.doc_count || 0), 0)}
            </td>
            <td style={{ padding: '10px 14px', borderTop: '2px solid var(--border)' }}>
              {data.total_chunks}
            </td>
            <td style={{ padding: '10px 14px', borderTop: '2px solid var(--border)',
                         color: 'var(--text-soft)' }}>—</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

export default function About() {
  const [ingestion, setIngestion] = useState(null);

  useEffect(() => {
    api.ingestion().then(setIngestion).catch(() => {});
  }, []);

  return (
    <div>
      <div style={{ background: 'var(--navy)', padding: '48px 0 52px' }}>
        <div className="container">
          <div className="badge badge--green" style={{ marginBottom: 14 }}>Methodology</div>
          <h1 style={{ fontSize: 36, color: 'var(--white)', marginBottom: 12 }}>About this project</h1>
          <p style={{ fontSize: 16, color: 'rgba(255,255,255,.55)', maxWidth: 560 }}>
            What this is, how it works, where the data comes from,
            and what it will and won't do.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container" style={{ maxWidth: 800 }}>

          {/* What this is */}
          <div style={{ marginBottom: 56 }}>
            <h2 style={{ fontSize: 24, marginBottom: 16 }}>What this project is</h2>
            <div className="card" style={{ padding: '28px 32px' }}>
              <p style={{ fontSize: 15, lineHeight: 1.8, color: 'var(--text-soft)', marginBottom: 16 }}>
                <strong style={{ color: 'var(--text)' }}>Citizen Services Assistant</strong> is a
                portfolio demonstration of a Retrieval-Augmented Generation (RAG) system applied to
                Pakistani government services. It was built to demonstrate senior-level engineering
                judgment in RAG pipeline design, agentic orchestration, and MCP tooling — not as a
                production government system.
              </p>
              <p style={{ fontSize: 15, lineHeight: 1.8, color: 'var(--text-soft)', marginBottom: 16 }}>
                Every answer is grounded in official public documents. The system never fabricates
                information — when it cannot find a reliable grounded answer, it explicitly declines.
                Citations are verified post-generation by a separate verification layer.
              </p>
              <div style={{
                background: '#FEF2F2', border: '1px solid #FECACA',
                borderRadius: 'var(--radius-md)', padding: '14px 18px',
                fontSize: 14, color: '#B91C1C', lineHeight: 1.6,
              }}>
                <strong>Not an official government service.</strong> Not affiliated with NADRA, FBR,
                SECP, DGIP, Punjab Excise & Taxation, or any Pakistani government body. Always verify
                information directly with the relevant authority before acting on it.
              </div>
            </div>
          </div>

          {/* Pipeline */}
          <div style={{ marginBottom: 56 }}>
            <h2 style={{ fontSize: 24, marginBottom: 6 }}>How the pipeline works</h2>
            <p style={{ color: 'var(--text-soft)', fontSize: 14, marginBottom: 24 }}>
              Every query passes through 8 stages. Each stage is independently testable.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {PIPELINE_STAGES.map((s, i) => (
                <div key={i} className="card" style={{
                  padding: '16px 20px', display: 'flex', gap: 16, alignItems: 'flex-start',
                }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8, background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 18, flexShrink: 0,
                  }}>{s.icon}</div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 13, color: 'var(--text-soft)', lineHeight: 1.6 }}>{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Groundedness */}
          <div style={{ marginBottom: 56 }}>
            <h2 style={{ fontSize: 24, marginBottom: 16 }}>Groundedness approach</h2>
            <div className="card" style={{ padding: '28px 32px' }}>
              <p style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--text-soft)', marginBottom: 16 }}>
                The pipeline enforces grounding at two layers:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {[
                  ['Pre-generation (retrieval gate)', 'The CrossEncoder reranker scores each retrieved chunk against the question. Chunks below MIN_RERANK_SCORE = 2.8 are filtered out before the LLM sees them. This threshold was calibrated on real domain data.'],
                  ['Post-generation (citation verification)', 'After the LLM generates an answer, each [source: file] citation is checked: does the cited chunk actually support that specific claim? Claims with low entailment scores are flagged.'],
                  ['Post-generation (hallucination evaluation)', 'Every sentence in the answer is scored against all chunks. Sentences with no chunk support are flagged. If more than 20% of sentences are unsupported, the answer is regenerated at temperature=0.'],
                ].map(([title, desc], i) => (
                  <div key={i} style={{
                    display: 'flex', gap: 14, alignItems: 'flex-start',
                  }}>
                    <div style={{
                      width: 6, height: 6, borderRadius: '50%', background: 'var(--green)',
                      flexShrink: 0, marginTop: 6,
                    }} />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 3 }}>{title}</div>
                      <div style={{ fontSize: 13, color: 'var(--text-soft)', lineHeight: 1.6 }}>{desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Ingestion metadata */}
          <div style={{ marginBottom: 56 }}>
            <h2 style={{ fontSize: 24, marginBottom: 6 }}>Data sources & ingestion</h2>
            <p style={{ color: 'var(--text-soft)', fontSize: 14, marginBottom: 24 }}>
              Only publicly available, non-copyrighted government information is ingested.
              Sources are snapshotted at ingestion time — the system does not scrape live at runtime.
            </p>
            <div className="card" style={{ overflow: 'hidden' }}>
              <IngestionTable data={ingestion} />
            </div>
            {ingestion?.total_chunks && (
              <p style={{ fontSize: 13, color: 'var(--slate)', marginTop: 12 }}>
                Total {ingestion.total_chunks} chunks indexed in pgvector.
                Embeddings generated with BAAI/bge-large-en (1024 dimensions).
              </p>
            )}
          </div>

          {/* Tech stack */}
          <div>
            <h2 style={{ fontSize: 24, marginBottom: 16 }}>Technical stack</h2>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12,
            }}>
              {[
                ['Embeddings', 'BAAI/bge-large-en (local, 1024-dim)'],
                ['Vector DB', 'pgvector (PostgreSQL)'],
                ['BM25', 'rank_bm25'],
                ['Reranker', 'ms-marco-MiniLM-L-6-v2'],
                ['LLM inference', 'Groq (llama3-70b + llama3-8b)'],
                ['Orchestration', 'LangGraph StateGraph'],
                ['MCP tools', 'Python mcp SDK (SSE transport)'],
                ['Backend', 'FastAPI (Railway)'],
                ['Frontend', 'React + React Router (Vercel)'],
              ].map(([label, value]) => (
                <div key={label} className="card" style={{ padding: '14px 18px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--slate)',
                                textTransform: 'uppercase', letterSpacing: '.5px', marginBottom: 5 }}>
                    {label}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
