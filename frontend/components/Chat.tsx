'use client';
import React from 'react';
import { apiAsk } from '@/lib/api';

function renderInlineBold(text: string) {
  const parts = text.split(/(\*\*.*?\*\*)/g).filter(Boolean);

  return parts.map((part, idx) => {
    const isBold = part.startsWith('**') && part.endsWith('**');
    if (!isBold) return <React.Fragment key={idx}>{part}</React.Fragment>;
    return <strong key={idx}>{part.slice(2, -2)}</strong>;
  });
}

type Message = { role: 'user' | 'assistant', content: string, citations?: { title: string, section?: string }[], chunks?: { title: string, section?: string, text: string, score: number }[] };

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const scrollerRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
    } // auto-scroll to bottom on new message
  }, [messages, loading]);

  const send = async () => {
    if (!q.trim() || loading) return;
    const my = { role: 'user' as const, content: q };
    setMessages(m => [...m, my]);
    setLoading(true);
    try {
      const res = await apiAsk(q);
      console.log('API response:', res);
      const ai: Message = { role: 'assistant', content: res.answer, citations: res.citations, chunks: res.chunks };
      setMessages(m => [...m, ai]);
    } catch (e: any) {
      setMessages(m => [...m, { role: 'assistant', content: 'Error: ' + e.message }]);
    } finally {
      setLoading(false);
      setQ('');
    }
  };

  return (
    <section className="chat-panel">
      <div className="chat-stream" ref={scrollerRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            <h3>Start a conversation</h3>
            <p>Ask about policy or products...</p>
          </div>
        )}

        {messages.map((m, i) => (
          <article key={i} className={m.role === 'user' ? 'msg msg-user' : 'msg msg-assistant'}>
            <div className="msg-meta">{m.role === 'user' ? 'You' : 'Assistant'}</div>

            <div className="msg-content">
              {m.content.split('\n\n').map((para, idx) => (
                <p key={idx}>{renderInlineBold(para)}</p>
              ))}
            </div>

            {m.citations && m.citations.length > 0 && (
              <div className="citation-row">
                {m.citations.map((c, idx) => (
                  <span key={idx} className="citation-pill" title={c.section || ''}>
                    {c.title}
                  </span>
                ))}
              </div>
            )}

            {m.chunks && m.chunks.length > 0 && (
              <details className="chunks">
                <summary>View supporting chunks</summary>
                {m.chunks.map((c, idx) => (
                  <div key={idx} className="chunk-card">
                    <div className="chunk-title">
                       {c.title} 
                       {c.section ? ' | ' + c.section : ''} 
                       {' | score: ' + c.score.toFixed(4)} 
                    </div>
                    <div className="chunk-text">{c.text}</div>
                  </div>
                ))}
              </details>
            )}
          </article>
        ))}

        {loading && (
          <article className="msg msg-assistant">
            <div className="msg-meta">Assistant</div>
            <div className="typing">Thinking...</div>
          </article>
        )}
      </div>

      <div className="composer">
        <input
          placeholder="Ask about policy or products..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') send();
          }}
        />
        <button className="btn btn-primary" onClick={send} disabled={loading}>
          {loading ? 'Loading...' : 'Send'}
        </button>
      </div>
    </section>
  );
}
