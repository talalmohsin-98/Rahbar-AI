import React from 'react';

function StageRow({ icon, label, value, sub, colour }) {
  return (
    <div style={{
      padding: '12px 0',
      borderBottom: '1px solid rgba(255,255,255,.08)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'rgba(255,255,255,.5)',
                       textTransform: 'uppercase', letterSpacing: '.6px' }}>
          {icon} {label}
        </span>
        <span style={{
          fontSize: 13, fontWeight: 600,
          color: colour || 'var(--green)',
        }}>{value}</span>
      </div>
      {sub && <div style={{ fontSize: 12, color: 'rgba(255,255,255,.4)', marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function ChunkRow({ rank, source, score, content }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,.05)',
      borderRadius: 6, padding: '10px 12px', marginBottom: 6,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'rgba(255,255,255,.4)' }}>
          Rank {rank} · {source?.split('/').pop()}
        </span>
        <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)' }}>
          {typeof score === 'number' ? score.toFixed(2) : score}
        </span>
      </div>
      <p style={{ fontSize: 12, color: 'rgba(255,255,255,.6)', margin: 0, lineHeight: 1.5 }}>
        {content?.slice(0, 90)}{content?.length > 90 ? '…' : ''}
      </p>
    </div>
  );
}

export default function PipelinePanel({ open, onClose, data }) {
  if (!data) return (
    <div className={`pipeline-panel${open ? ' open' : ''}`}>
      <div style={{ padding: '24px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <span style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 15 }}>
            Pipeline Inspector
          </span>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,.1)', border: 'none', color: 'white',
            borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontSize: 13,
          }}>Close</button>
        </div>
        <p style={{ color: 'rgba(255,255,255,.4)', fontSize: 13 }}>
          Send a question to see the pipeline run live.
        </p>
      </div>
    </div>
  );

  const {
    intent, routing_config, search_queries = [],
    raw_chunks = [], reranked_chunks = [], compressed_chunks = [],
    tool_results = [], generation_meta = {}, citation_result = {},
    hallucination_result = {}, retry_count = 0,
  } = data;

  const halRate = hallucination_result?.hallucination_rate ?? 0;
  const citRate = citation_result?.verification_rate ?? 1;

  return (
    <div className={`pipeline-panel${open ? ' open' : ''}`}>
      <div style={{ padding: '20px' }}>
        {/* Header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', marginBottom: 20,
        }}>
          <div>
            <div style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 15 }}>
              Pipeline Inspector
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,.4)', marginTop: 2 }}>
              Live pipeline trace
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,.1)', border: 'none', color: 'white',
            borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontSize: 13,
          }}>✕ Close</button>
        </div>

        {/* Stage 1: Intent */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            🧠 Intent Router
          </div>
          <StageRow icon="" label="Detected Intent" value={intent?.toUpperCase() || '—'} />
          <StageRow icon="" label="Answer Format"
            value={routing_config?.answer_format?.split(' ').slice(0,3).join(' ') || '—'}
            colour="rgba(255,255,255,.7)" />
        </div>

        {/* Stage 2: Queries */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            ✍️ Query Rewriter
          </div>
          <StageRow icon="" label="Queries Generated" value={search_queries.length} />
          <div style={{ marginTop: 6 }}>
            {search_queries.map((q, i) => (
              <div key={i} style={{
                background: 'rgba(255,255,255,.05)', borderRadius: 6,
                padding: '6px 10px', marginBottom: 4,
                fontSize: 11, color: 'rgba(255,255,255,.6)', lineHeight: 1.4,
              }}>
                <span style={{ color: 'var(--green)', marginRight: 6, fontWeight: 700 }}>
                  {i === 0 ? 'OG' : i === 1 ? 'HyDE' : `Q${i-1}`}
                </span>
                {q.slice(0, 60)}{q.length > 60 ? '…' : ''}
              </div>
            ))}
          </div>
        </div>

        {/* Stage 3: Retrieval */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            🔍 Hybrid Search → RRF
          </div>
          <StageRow icon="" label="Chunks After RRF" value={raw_chunks.length} />
          {raw_chunks.slice(0, 3).map(c => (
            <ChunkRow key={c.chunk_id} rank={c.rank}
              source={c.source} score={c.rrf_score} content={c.content} />
          ))}
        </div>

        {/* Stage 4: Rerank */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            ⚖️ CrossEncoder Reranker
          </div>
          <StageRow icon="" label="After Reranking" value={reranked_chunks.length}
            sub="MIN_SCORE = 2.8" />
          {reranked_chunks.slice(0, 3).map(c => (
            <ChunkRow key={c.chunk_id} rank={c.rank}
              source={c.source} score={c.rerank_score} content={c.content} />
          ))}
        </div>

        {/* Stage 5: Compression */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            ✂️ Context Compressor
          </div>
          <StageRow icon="" label="Compressed Chunks" value={compressed_chunks.length} />
        </div>

        {/* MCP Tools */}
        {tool_results.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                          textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
              🔧 MCP Tools
            </div>
            {tool_results.map((t, i) => (
              <StageRow key={i} icon="" label={t.tool}
                value={t.success ? '✓ OK' : '✗ Failed'}
                colour={t.success ? 'var(--green)' : '#EF4444'} />
            ))}
          </div>
        )}

        {/* Stage 6: Generation */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            💬 Generator
          </div>
          <StageRow icon="" label="Model" value={generation_meta.model || 'llama3-70b'} colour="rgba(255,255,255,.7)" />
          <StageRow icon="" label="Tokens (in/out)"
            value={`${generation_meta.prompt_tokens || 0} / ${generation_meta.completion_tokens || 0}`}
            colour="rgba(255,255,255,.7)" />
          {retry_count > 0 && (
            <StageRow icon="" label="Retries" value={retry_count} colour="var(--warning)" />
          )}
        </div>

        {/* Stage 7-8: Verification */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--green)',
                        textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
            📎 Verification
          </div>
          <StageRow icon="" label="Citation Rate"
            value={`${Math.round(citRate * 100)}%`}
            colour={citRate >= 0.8 ? 'var(--green)' : '#EF4444'} />
          <StageRow icon="" label="Hallucination Rate"
            value={`${Math.round(halRate * 100)}%`}
            colour={halRate < 0.2 ? 'var(--green)' : '#EF4444'} />
          <StageRow icon="" label="Overall"
            value={citation_result?.passed && hallucination_result?.passed ? '✓ PASSED' : '⚠ FLAGGED'}
            colour={citation_result?.passed && hallucination_result?.passed ? 'var(--green)' : '#F59E0B'} />
        </div>
      </div>
    </div>
  );
}
