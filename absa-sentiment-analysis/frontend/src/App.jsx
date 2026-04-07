import { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:8000/chat";

const SENTIMENT = {
  positive: { bg: "#f0faf4", text: "#1a5c35", border: "#b6dfc8", bar: "#2d9e5f" },
  negative: { bg: "#fdf4f4", text: "#6b1f1f", border: "#e8b4b4", bar: "#c94040" },
  neutral:  { bg: "#f7f7f5", text: "#4a4a4a", border: "#d4d4cc", bar: "#8a8a82" },
  conflict: { bg: "#fdf8ee", text: "#6b4a0f", border: "#e8d4a0", bar: "#c98a20" },
};

const SUGGESTIONS = [
  "The food was amazing but service was terrible.",
  "Great battery life, poor screen quality.",
  "Ambiance was nice | aspects: ambiance, food, price",
];

const ANIM = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body { background: #f0ede8; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.85); }
    50%       { opacity: 1;   transform: scale(1); }
  }
  @keyframes fillBar {
    from { width: 0%; }
    to   { width: var(--w); }
  }
  .msg-in { animation: fadeUp 0.35s cubic-bezier(0.22,1,0.36,1) both; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #d4cfc8; border-radius: 4px; }

  textarea { font-family: 'DM Sans', sans-serif; }
  textarea::placeholder { color: #b0aa9e; }
  textarea:focus { outline: none; border-color: #2c2825 !important; }

  .chip:hover { background: #e8e4de !important; color: #2c2825 !important; }
  .send-btn:hover:not(:disabled) { background: #3d3530 !important; }
  .send-btn:active:not(:disabled) { transform: scale(0.96); }

  .aspect-card { transition: box-shadow 0.2s, transform 0.2s; }
  .aspect-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.07); transform: translateY(-1px); }
`;

function ConfidenceBar({ value, color }) {
  return (
    <div style={{ height: 3, background: "#e8e4de", borderRadius: 99, overflow: "hidden", marginTop: 6 }}>
      <div
        style={{
          height: "100%",
          width: `${(value * 100).toFixed(1)}%`,
          background: color,
          borderRadius: 99,
          animation: "fillBar 0.8s cubic-bezier(0.22,1,0.36,1) both",
          "--w": `${(value * 100).toFixed(1)}%`,
        }}
      />
    </div>
  );
}

function renderMarkdown(text) {
  return text.split("\n").map((line, i) => {
    let parts = line.split(/\*\*(.+?)\*\*/g);
    let rendered = parts.map((p, j) =>
      j % 2 === 1
        ? <strong key={j} style={{ fontWeight: 600, color: "#1a1714" }}>{p}</strong>
        : p
    );
    rendered = rendered.flatMap((node, j) => {
      if (typeof node !== "string") return [node];
      return node.split(/`(.+?)`/g).map((p, k) =>
        k % 2 === 1
          ? <code key={`${j}-${k}`} style={{
              background: "#ede9e2", padding: "1px 6px", borderRadius: 4,
              fontSize: 12.5, fontFamily: "'DM Mono', monospace", color: "#5c4a2a"
            }}>{p}</code>
          : p
      );
    });

    if (line.startsWith("> ")) {
      return (
        <blockquote key={i} style={{
          borderLeft: "2px solid #c4b99a", paddingLeft: 12,
          color: "#6b6358", margin: "4px 0", fontStyle: "italic", fontSize: 13.5
        }}>
          {line.slice(2)}
        </blockquote>
      );
    }
    if (line.startsWith("• ") || line.startsWith("- ")) {
      return (
        <div key={i} style={{ display: "flex", gap: 8, marginBottom: 2, paddingLeft: 2 }}>
          <span style={{ color: "#a09486", marginTop: 1, flexShrink: 0 }}>—</span>
          <span>{rendered.slice(2)}</span>
        </div>
      );
    }
    return <div key={i} style={{ minHeight: line ? undefined : 8 }}>{rendered}</div>;
  });
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => { sendToAPI("hello", []); }, []);

  async function sendToAPI(userText, history) {
    setLoading(true);
    const allMsgs = [...history, { role: "user", content: userText }];
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: allMsgs }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.reply,
        results: data.results || null,
        sentence: data.raw_sentence || null,
        ts: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Could not reach the backend. Make sure FastAPI is running on localhost:8000.",
        ts: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSend(text) {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages(prev => [...prev, {
      role: "user", content: msg,
      ts: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);
    setInput("");
    await sendToAPI(msg, history);
  }

  return (
    <>
      <style>{ANIM}</style>
      <div style={{
        minHeight: "100vh",
        background: "#f0ede8",
        backgroundImage: `
          radial-gradient(ellipse at 20% 10%, rgba(210,195,175,0.4) 0%, transparent 55%),
          radial-gradient(ellipse at 80% 90%, rgba(190,180,165,0.3) 0%, transparent 55%)
        `,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'DM Sans', sans-serif",
        padding: 16,
      }}>
        <div style={{
          width: "100%",
          maxWidth: 800,
          background: "#faf8f5",
          border: "1px solid #e0dbd3",
          borderRadius: 20,
          display: "flex",
          flexDirection: "column",
          height: "90vh",
          maxHeight: 880,
          overflow: "hidden",
          boxShadow: "0 8px 48px rgba(60,45,30,0.10), 0 2px 8px rgba(60,45,30,0.06)",
        }}>

          {/* ── Header ── */}
          <div style={{
            padding: "20px 28px",
            borderBottom: "1px solid #e8e3db",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "#faf8f5",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{
                width: 40, height: 40, borderRadius: 10,
                background: "#2c2825",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0,
              }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f0ede8" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
              </div>
              <div>
                <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 17, fontWeight: 600, color: "#1a1714", letterSpacing: "-0.2px" }}>
                  ABSA Sentiment Analyst
                </div>
                <div style={{ fontSize: 11.5, color: "#9a9088", marginTop: 1, letterSpacing: "0.04em", textTransform: "uppercase", fontWeight: 500 }}>
                  BERT · Aspect-Based Analysis
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#2d9e5f" }} />
              <span style={{ fontSize: 11.5, color: "#9a9088", fontWeight: 500 }}>Online</span>
            </div>
          </div>

          {/* ── Messages ── */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "28px 24px",
            display: "flex", flexDirection: "column", gap: 20,
          }}>
            {messages.map((msg, i) => (
              <div key={i} className="msg-in" style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                alignItems: "flex-start",
                gap: 12,
              }}>
                {/* Avatar */}
                <div style={{
                  width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                  background: msg.role === "user" ? "#2c2825" : "#e8e3db",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  marginTop: 2,
                }}>
                  {msg.role === "user"
                    ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f0ede8" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
                    : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b6358" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="15" x2="8" y2="17"/><line x1="12" y1="15" x2="12" y2="17"/><line x1="16" y1="15" x2="16" y2="17"/></svg>
                  }
                </div>

                <div style={{ maxWidth: "74%", display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}>
                  <div style={{
                    background: msg.role === "user" ? "#2c2825" : "#fff",
                    border: msg.role === "user" ? "none" : "1px solid #e8e3db",
                    borderRadius: msg.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                    padding: "12px 16px",
                    color: msg.role === "user" ? "#f0ede8" : "#2c2825",
                    fontSize: 14,
                    lineHeight: 1.7,
                    wordBreak: "break-word",
                    boxShadow: msg.role === "user" ? "none" : "0 1px 4px rgba(60,45,30,0.06)",
                  }}>
                    {renderMarkdown(msg.content)}

                    {/* Aspect result cards */}
                    {msg.results?.length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 14 }}>
                        {msg.results.map((r, j) => {
                          const s = SENTIMENT[r.sentiment] || SENTIMENT.neutral;
                          return (
                            <div key={j} className="aspect-card" style={{
                              background: s.bg,
                              border: `1px solid ${s.border}`,
                              borderRadius: 10,
                              padding: "10px 14px",
                              minWidth: 130,
                              cursor: "default",
                            }}>
                              <div style={{ fontSize: 10.5, fontWeight: 600, color: "#9a9088", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                                {r.aspect}
                              </div>
                              <div style={{ fontSize: 13.5, fontWeight: 500, color: s.text, marginTop: 3 }}>
                                {r.sentiment.charAt(0).toUpperCase() + r.sentiment.slice(1)}
                              </div>
                              <ConfidenceBar value={r.confidence} color={s.bar} />
                              <div style={{ fontSize: 11, color: "#9a9088", marginTop: 4 }}>
                                {(r.confidence * 100).toFixed(1)}% confidence
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                  {msg.ts && (
                    <div style={{ fontSize: 10.5, color: "#b0aa9e", marginTop: 5, paddingLeft: 2, paddingRight: 2 }}>
                      {msg.ts}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="msg-in" style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                <div style={{
                  width: 30, height: 30, borderRadius: 8, background: "#e8e3db",
                  display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2,
                }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b6358" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/>
                    <path d="M12 7v4"/><line x1="8" y1="15" x2="8" y2="17"/><line x1="12" y1="15" x2="12" y2="17"/><line x1="16" y1="15" x2="16" y2="17"/>
                  </svg>
                </div>
                <div style={{
                  background: "#fff", border: "1px solid #e8e3db",
                  borderRadius: "14px 14px 14px 4px", padding: "14px 18px",
                  boxShadow: "0 1px 4px rgba(60,45,30,0.06)",
                  display: "flex", gap: 5, alignItems: "center",
                }}>
                  {[0, 0.18, 0.36].map((delay, i) => (
                    <div key={i} style={{
                      width: 6, height: 6, borderRadius: "50%", background: "#c4b99a",
                      animation: `pulse 1.2s ease-in-out ${delay}s infinite`,
                    }} />
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Suggestion chips */}
          {messages.length <= 1 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "0 24px 12px" }}>
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="chip" onClick={() => handleSend(s)} style={{
                  background: "#f0ede8", border: "1px solid #ddd8d0",
                  borderRadius: 20, padding: "5px 13px",
                  color: "#6b6358", fontSize: 12, cursor: "pointer",
                  fontFamily: "'DM Sans', sans-serif", fontWeight: 400,
                  transition: "all 0.15s", letterSpacing: "0.01em",
                }}>
                  {s.length > 44 ? s.slice(0, 44) + "…" : s}
                </button>
              ))}
            </div>
          )}

          {/* ── Input area ── */}
          <div style={{
            padding: "14px 20px 18px",
            borderTop: "1px solid #e8e3db",
            background: "#faf8f5",
            display: "flex",
            gap: 10,
            alignItems: "flex-end",
          }}>
            <textarea
              rows={1}
              style={{
                flex: 1,
                background: "#fff",
                border: "1px solid #e0dbd3",
                borderRadius: 12,
                padding: "11px 15px",
                color: "#2c2825",
                fontSize: 14,
                resize: "none",
                lineHeight: 1.55,
                transition: "border-color 0.2s",
                boxShadow: "0 1px 4px rgba(60,45,30,0.05)",
              }}
              placeholder="Type a review sentence… (Shift+Enter for newline)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={() => handleSend()}
              disabled={loading}
              style={{
                width: 44, height: 44, borderRadius: 12, border: "none",
                background: loading ? "#c4b99a" : "#2c2825",
                color: "#f0ede8", cursor: loading ? "not-allowed" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, transition: "background 0.2s, transform 0.1s",
                boxShadow: loading ? "none" : "0 2px 8px rgba(44,40,37,0.25)",
              }}
            >
              {loading
                ? <div style={{ width: 16, height: 16, border: "2px solid rgba(240,237,232,0.3)", borderTop: "2px solid #f0ede8", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                  </svg>
              }
            </button>
          </div>
        </div>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  );
}