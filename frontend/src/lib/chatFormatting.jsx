import React from 'react';

/**
 * Renders **bold** markdown within a plain-text segment. Without this,
 * literal asterisks show up in the chat (e.g. "**Renewal**").
 */
export function renderInlineMarkdown(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    return part ? <React.Fragment key={`${keyPrefix}-${i}`}>{part}</React.Fragment> : null;
  });
}

/**
 * Turns a raw LLM answer into clean, structured React blocks for display.
 *
 * Users don't want source plumbing in the answer, so we:
 *   1. Strip every citation tag — ASCII `[source: x]` AND the full-width
 *      `【source: x】` variant the model sometimes emits (which the old
 *      renderer left showing literally in the chat).
 *   2. Promote inline bullet markers ("* " / "•") onto their own lines so a
 *      list stacks vertically instead of running together on one line.
 *   3. Group consecutive "- " lines into a real <ul> and render everything
 *      else as paragraphs, so lists and bullet points read cleanly.
 * Citations are still generated internally (for pipeline verification) —
 * they're just never shown here.
 */
export function formatAnswer(text) {
  if (!text) return null;

  let clean = text
    .replace(/[[【]\s*source\s*:[^\]】]*[\]】]/gi, '') // drop citation tags
    .replace(/【[^】]*】/g, '')                          // any stray full-width brackets
    .replace(/[ \t]{2,}/g, ' ');

  // Promote inline bullets ("• x" or " * x") to their own list lines.
  clean = clean
    .replace(/\s*•\s*/g, '\n- ')
    .replace(/(^|[^*\n])\*\s+(?=\S)/g, '$1\n- ');

  const lines = clean.split('\n').map(l => l.trim()).filter(Boolean);

  const blocks = [];
  let list = null;
  const flush = () => { if (list) { blocks.push({ type: 'ul', items: list }); list = null; } };

  lines.forEach(line => {
    if (/^[-*]\s+/.test(line)) {
      (list ||= []).push(line.replace(/^[-*]\s+/, ''));
    } else {
      flush();
      blocks.push({ type: 'p', text: line });
    }
  });
  flush();

  return blocks.map((b, i) =>
    b.type === 'ul'
      ? (
        <ul key={i} style={{ margin: '6px 0', paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {b.items.map((it, j) => (
            <li key={j}>{renderInlineMarkdown(it, `b${i}-${j}`)}</li>
          ))}
        </ul>
      )
      : <p key={i} style={{ margin: '0 0 8px' }}>{renderInlineMarkdown(b.text, `b${i}`)}</p>
  );
}
