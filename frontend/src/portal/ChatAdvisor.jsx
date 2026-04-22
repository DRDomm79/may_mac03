import { useState, useEffect, useRef } from "react";

const API_URL = "https://dac-healthprice-api.onrender.com";

const NAVY   = "#0d2b7a";
const GOLD   = "#f5a623";
const WHITE  = "#ffffff";
const GRAY   = "#94a3b8";
const LTGRAY = "#f1f5f9";
const TXT    = "#111827";
const OK     = "#10b981";
const ERR    = "#ef4444";

const inputStyle = {
  width: "100%", padding: "9px 12px", borderRadius: 8,
  border: "1.5px solid #e5e7eb", fontSize: 13, fontFamily: "inherit",
  outline: "none", boxSizing: "border-box",
};

const GREETING = "Hi! I'm your DAC insurance advisor, powered by Claude. Ask me about coverage options, premium pricing, the application process, or anything else about your health insurance.";

export default function ChatAdvisor({ open, onOpenChange, context = {} }) {
  const [messages, setMessages] = useState([{ role: "assistant", text: GREETING }]);
  const [input, setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    if (open && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setError("");
    const next = [...messages, { role: "user", text: q }];
    setMessages(next);
    setLoading(true);
    try {
      // Skip the initial greeting when building API history
      const history = next.slice(1).map(m => ({ role: m.role, content: m.text }));
      const res = await fetch(`${API_URL}/api/v1/advisor/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history, context }),
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const { reply } = await res.json();
      setMessages(m => [...m, { role: "assistant", text: reply }]);
    } catch {
      setError("Couldn't reach the advisor. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button
        onClick={() => onOpenChange(!open)}
        title={open ? "Close advisor" : "Ask our AI advisor"}
        style={{
          position: "fixed", bottom: 28, right: 28, zIndex: 300,
          width: 56, height: 56, borderRadius: "50%",
          background: GOLD, border: "none", cursor: "pointer",
          boxShadow: "0 4px 20px rgba(245,166,35,0.45)",
          fontSize: 22, display: "flex", alignItems: "center", justifyContent: "center",
          transition: "transform 0.2s", fontFamily: "inherit",
        }}
        onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.08)"; }}
        onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
      >
        {open ? "✕" : "💬"}
      </button>

      {open && (
        <div style={{
          position: "fixed", bottom: 96, right: 28, zIndex: 300,
          width: 340, borderRadius: 18,
          background: WHITE, boxShadow: "0 8px 48px rgba(0,0,0,0.18)",
          display: "flex", flexDirection: "column", overflow: "hidden",
          fontFamily: "'DM Sans', sans-serif",
          animation: "chatSlideUp 0.25s ease",
        }}>
          <style>{`@keyframes chatSlideUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }`}</style>

          {/* Header */}
          <div style={{ background: NAVY, padding: "14px 18px", display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 38, height: 38, borderRadius: "50%", background: GOLD, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>
              💬
            </div>
            <div>
              <p style={{ color: WHITE, fontSize: 14, fontWeight: 700, margin: 0 }}>AI Quote Advisor</p>
              <p style={{ color: OK, fontSize: 11, margin: 0, fontWeight: 600 }}>● Powered by Claude</p>
            </div>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 12px", maxHeight: 340, display: "flex", flexDirection: "column", gap: 10 }}>
            {messages.map((m, i) => (
              <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
                <div style={{
                  maxWidth: "85%", padding: "9px 13px",
                  borderRadius: m.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                  background: m.role === "user" ? NAVY : LTGRAY,
                  color: m.role === "user" ? WHITE : TXT,
                  fontSize: 13, lineHeight: 1.65, whiteSpace: "pre-wrap",
                }}>
                  {m.text}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ display: "flex" }}>
                <div style={{ padding: "9px 13px", borderRadius: "14px 14px 14px 4px", background: LTGRAY, fontSize: 13, color: GRAY }}>
                  Thinking…
                </div>
              </div>
            )}
            {error && <p style={{ fontSize: 12, color: ERR, textAlign: "center", margin: "4px 0 0" }}>{error}</p>}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{ borderTop: "1px solid #e5e7eb", padding: "10px 12px", display: "flex", gap: 8 }}>
            <input
              style={{ ...inputStyle, flex: 1 }}
              placeholder="Ask about coverage, pricing…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && send()}
              autoFocus
              disabled={loading}
            />
            <button
              onClick={send}
              disabled={!input.trim() || loading}
              style={{
                padding: "9px 14px", borderRadius: 8,
                background: input.trim() && !loading ? GOLD : "#e5e7eb",
                border: "none",
                cursor: input.trim() && !loading ? "pointer" : "not-allowed",
                fontFamily: "inherit", fontWeight: 700, fontSize: 13,
                color: input.trim() && !loading ? NAVY : GRAY,
                transition: "all 0.2s",
              }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
