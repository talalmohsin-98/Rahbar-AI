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
    hallucination_result = {}, completeness_result = {}, retry_count = 0,
    verification_verdict = null, input_analysis = null,
    attestation_result = {},
  } = data;

  // Both rates are null when the stage couldn't measure anything (no citations
  // parsed / no evaluable claims). Defaulting those to 0% and 100% respectively
  // is how the inspector used to report a perfect score for an unchecked answer.
  const halRate = hallucination_result?.hallucination_rate;
  const citRate = citation_result?.verification_rate;
  const pct = (r) => (r === null || r === undefined ? 'not measured' : `${Math.round(r * 100)}%`);

  // This panel used to recompute its own "overall" from the three results
  // while the answer badges computed a different one from two of them. One
  // backend verdict now drives both, so the inspector can never disagree with
  // what the citizen was shown.
  const verdictStatus = verification_verdict?.status ?? 'unverified';

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

        {/* Stage 1b: Input analysis — only when it found something worth
            saying. A plain English single-domain question shows nothing. */}
        {input_analysis && (input_analysis.code_switched || input_analysis.compound) && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#F59E0B',
                          textTransform: 'uppercase', letterSpacing: '.8px', marginBottom: 8 }}>
              ⚠ Input Analysis
            </div>
            {input_analysis.code_switched && (
              <StageRow icon="" label="Code-switched"
                value={input_analysis.code_switch_kind}
                sub="English-only embedder — English variant added to the search"
                colour="#F59E0B" />
            )}
            {input_analysis.compound && (
              <StageRow icon="" label="Spans domains"
                value={input_analysis.domains.join(' + ')}
                sub="one extra retrieval query per domain"
                colour="#F59E0B" />
            )}
          </div>
        )}

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
            sub="noise floor = 0.0 (relative)" />
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
          <StageRow icon="" label="Model" value={generation_meta.model || 'openai/gpt-oss-120b'} colour="rgba(255,255,255,.7)" />
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
            value={pct(citRate)}
            sub={citation_result?.checked_count != null
              ? `${citation_result.checked_count} citation(s) checked` : undefined}
            colour={citRate == null ? 'rgba(255,255,255,.5)'
                  : citRate >= 0.8 ? 'var(--green)' : '#EF4444'} />
          <StageRow icon="" label="Hallucination Rate"
            value={pct(halRate)}
            sub={hallucination_result?.evaluated_count != null
              ? `${hallucination_result.evaluated_count} claim(s) evaluated` : undefined}
            colour={halRate == null ? 'rgba(255,255,255,.5)'
                  : halRate < 0.2 ? 'var(--green)' : '#EF4444'} />
          <StageRow icon="" label="Completeness"
            value={completeness_result?.status === 'thin' ? '⚠ context unused'
                 : completeness_result?.status === 'complete' ? '✓ context used'
                 : '—'}
            sub={completeness_result?.claim_lines != null
              ? `${completeness_result.claim_lines} claim(s) in answer` : undefined}
            colour={completeness_result?.status === 'thin' ? '#F59E0B'
                  : completeness_result?.status === 'complete' ? 'var(--green)'
                  : 'rgba(255,255,255,.5)'} />
          <StageRow icon="" label="Term attestation"
            value={attestation_result?.status === 'missing' ? '⚠ not in sources'
                 : attestation_result?.status === 'attested' ? '✓ terms found'
                 : '—'}
            sub={attestation_result?.missing?.length
              ? `never mentioned: ${attestation_result.missing.join(', ')}`
              : undefined}
            colour={attestation_result?.status === 'missing' ? '#F59E0B'
                  : attestation_result?.status === 'attested' ? 'var(--green)'
                  : 'rgba(255,255,255,.5)'} />
          <StageRow icon="" label="Overall"
            value={verdictStatus === 'verified' ? '✓ PASSED'
                 : verdictStatus === 'partial'  ? '⚠ FLAGGED'
                 : '— NOT VERIFIED'}
            sub={verification_verdict?.headline}
            colour={verdictStatus === 'verified' ? 'var(--green)'
                  : verdictStatus === 'partial'  ? '#F59E0B'
                  : 'rgba(255,255,255,.5)'} />
          {verification_verdict?.reasons?.map((r, i) => (
            <div key={i} style={{
              fontSize: 11, color: '#F59E0B', marginTop: 6,
              paddingLeft: 10, borderLeft: '2px solid rgba(245,158,11,.4)',
              lineHeight: 1.5,
            }}>{r}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
