import { useState, useEffect, useRef } from "react";
import { PLATFORM_NODES, PLATFORM_EDGES, NODE_CATEGORIES } from "./platformArchitecture";
import { API_URL } from "./auth";

const NAVY   = "#0d2b7a";
const GOLD   = "#f5a623";
const BGPAGE = "#f0f4ff";
const BGCARD = "#ffffff";
const TXT    = "#111827";
const TXT2   = "#4b5563";
const BORDER = "#e2e8f0";

// Compute node center point
function nodeCenter(n) {
  return { x: n.x + n.w / 2, y: n.y + n.h / 2 };
}

// Build a lookup map id → node
const NODE_MAP = Object.fromEntries(PLATFORM_NODES.map(n => [n.id, n]));

// ─── SVG Diagram ────────────────────────────────────────────────────────────
function Diagram({ selected, onSelect }) {
  return (
    <svg
      viewBox="0 0 960 480"
      style={{ width: "100%", display: "block", background: BGPAGE, borderRadius: 12, border: `1px solid ${BORDER}` }}
      aria-label="DAC HealthPrice platform architecture diagram"
    >
      {/* Title */}
      <text x={480} y={28} textAnchor="middle" fontSize={13} fontWeight={600} fill={TXT2} fontFamily="DM Sans, sans-serif">
        DAC HealthPrice — Platform Architecture
      </text>

      {/* Divider line above backend */}
      <line x1={80} y1={400} x2={880} y2={400} stroke={BORDER} strokeWidth={1} strokeDasharray="4 4" />
      <text x={80} y={395} fontSize={10} fill="#94a3b8" fontFamily="DM Sans, sans-serif">API layer</text>

      {/* Edges */}
      {PLATFORM_EDGES.map((e, i) => {
        const src = NODE_MAP[e.from];
        const dst = NODE_MAP[e.to];
        if (!src || !dst) return null;
        const sx = src.x + src.w / 2;
        const sy = src.y + src.h;
        const dx = dst.x + dst.w / 2;
        const dy = dst.y;
        const mid = (sy + dy) / 2;
        return (
          <path
            key={i}
            d={`M${sx},${sy} C${sx},${mid} ${dx},${mid} ${dx},${dy}`}
            fill="none"
            stroke={e.dashed ? "#cbd5e1" : "#94a3b8"}
            strokeWidth={e.dashed ? 1 : 1.5}
            strokeDasharray={e.dashed ? "4 3" : undefined}
            opacity={e.dashed ? 0.6 : 0.9}
          />
        );
      })}

      {/* Nodes */}
      {PLATFORM_NODES.map(node => {
        const cat = NODE_CATEGORIES[node.category];
        const isSelected = selected?.id === node.id;
        return (
          <g
            key={node.id}
            onClick={() => onSelect(isSelected ? null : node)}
            style={{ cursor: "pointer" }}
            role="button"
            aria-label={node.label}
          >
            {/* Glow on selection */}
            {isSelected && (
              <rect
                x={node.x - 3}
                y={node.y - 3}
                width={node.w + 6}
                height={node.h + 6}
                rx={11}
                fill="none"
                stroke={GOLD}
                strokeWidth={2.5}
                opacity={0.9}
              />
            )}
            <rect
              x={node.x}
              y={node.y}
              width={node.w}
              height={node.h}
              rx={8}
              fill={isSelected ? cat.color : cat.color}
              opacity={isSelected ? 1 : 0.88}
              style={{ filter: isSelected ? `drop-shadow(0 4px 10px ${cat.color}55)` : undefined }}
            />
            <text
              x={node.x + node.w / 2}
              y={node.y + node.h / 2 + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={node.w < 100 ? 10.5 : 12}
              fontWeight={600}
              fill={cat.text}
              fontFamily="DM Sans, sans-serif"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {node.label}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      {Object.entries(NODE_CATEGORIES).map(([key, cat], i) => (
        <g key={key} transform={`translate(${20 + i * 130}, 460)`}>
          <rect width={14} height={14} rx={3} fill={cat.color} opacity={0.9} />
          <text x={20} y={11} fontSize={10} fill={TXT2} fontFamily="DM Sans, sans-serif">{cat.badge}</text>
        </g>
      ))}
    </svg>
  );
}

// ─── Info Panel ──────────────────────────────────────────────────────────────
function InfoPanel({ node, chatMsgs, chatInput, chatLoading, onChatInput, onSend }) {
  const chatEndRef = useRef(null);
  const inputRef   = useRef(null);
  const cat = NODE_CATEGORIES[node.category];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMsgs]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0, height: "100%" }}>
      {/* Node header */}
      <div style={{
        background: cat.color,
        borderRadius: "12px 12px 0 0",
        padding: "20px 20px 16px",
        color: cat.text,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{
            background: "rgba(255,255,255,0.2)",
            borderRadius: 20,
            padding: "2px 10px",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0.4,
          }}>{cat.badge}</span>
          <span style={{
            background: "rgba(255,255,255,0.2)",
            borderRadius: 20,
            padding: "2px 10px",
            fontSize: 11,
            fontWeight: 600,
            color: "#86efac",
          }}>● LIVE</span>
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{node.label}</div>
      </div>

      {/* Description + details */}
      <div style={{ background: BGCARD, padding: "16px 20px", borderLeft: `1px solid ${BORDER}`, borderRight: `1px solid ${BORDER}` }}>
        <p style={{ fontSize: 13.5, color: TXT, lineHeight: 1.65, margin: 0 }}>{node.description}</p>

        {node.endpoints.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: TXT2, textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 6 }}>
              API Endpoints
            </div>
            {node.endpoints.map(ep => (
              <div key={ep} style={{
                fontFamily: "monospace",
                fontSize: 11.5,
                background: "#f1f5f9",
                borderRadius: 4,
                padding: "3px 8px",
                marginBottom: 4,
                color: NAVY,
                display: "inline-block",
                marginRight: 6,
              }}>{ep}</div>
            ))}
          </div>
        )}

        {node.tags.length > 0 && (
          <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 6 }}>
            {node.tags.map(t => (
              <span key={t} style={{
                background: `${cat.color}18`,
                color: cat.color,
                borderRadius: 20,
                padding: "2px 10px",
                fontSize: 11,
                fontWeight: 600,
              }}>{t}</span>
            ))}
          </div>
        )}
      </div>

      {/* AI Chat */}
      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        background: BGCARD,
        border: `1px solid ${BORDER}`,
        borderTop: `1px solid ${BORDER}`,
        borderRadius: "0 0 12px 12px",
        overflow: "hidden",
        minHeight: 0,
      }}>
        <div style={{
          padding: "10px 16px",
          borderBottom: `1px solid ${BORDER}`,
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "#f8fafc",
        }}>
          <span style={{ fontSize: 14 }}>✦</span>
          <span style={{ fontSize: 12, fontWeight: 600, color: TXT2 }}>Ask AI about this feature</span>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px", display: "flex", flexDirection: "column", gap: 10, minHeight: 80, maxHeight: 220 }}>
          {chatMsgs.length === 0 && (
            <p style={{ fontSize: 12, color: "#94a3b8", margin: "auto", textAlign: "center" }}>
              Ask anything about {node.label}…
            </p>
          )}
          {chatMsgs.map((m, i) => (
            <div key={i} style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              background: m.role === "user" ? NAVY : "#f1f5f9",
              color: m.role === "user" ? "#fff" : TXT,
              borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              padding: "8px 12px",
              maxWidth: "90%",
              fontSize: 12.5,
              lineHeight: 1.55,
              whiteSpace: "pre-wrap",
            }}>{m.content}</div>
          ))}
          {chatLoading && (
            <div style={{ alignSelf: "flex-start", background: "#f1f5f9", borderRadius: "16px 16px 16px 4px", padding: "8px 12px", color: TXT2, fontSize: 12 }}>
              Thinking…
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "10px 12px", borderTop: `1px solid ${BORDER}`, display: "flex", gap: 8 }}>
          <input
            ref={inputRef}
            value={chatInput}
            onChange={e => onChatInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); } }}
            placeholder={`Ask about ${node.label}…`}
            style={{
              flex: 1,
              padding: "8px 12px",
              borderRadius: 20,
              border: `1.5px solid ${BORDER}`,
              fontSize: 12.5,
              fontFamily: "DM Sans, sans-serif",
              outline: "none",
              background: "#f8fafc",
            }}
          />
          <button
            onClick={onSend}
            disabled={chatLoading || !chatInput.trim()}
            style={{
              background: chatLoading || !chatInput.trim() ? BORDER : GOLD,
              color: NAVY,
              border: "none",
              borderRadius: 20,
              padding: "8px 16px",
              fontSize: 12,
              fontWeight: 700,
              cursor: chatLoading || !chatInput.trim() ? "default" : "pointer",
              fontFamily: "DM Sans, sans-serif",
              whiteSpace: "nowrap",
            }}
          >Send</button>
        </div>
      </div>
    </div>
  );
}

// ─── Placeholder when nothing selected ───────────────────────────────────────
function Placeholder() {
  return (
    <div style={{
      background: BGCARD,
      border: `1px solid ${BORDER}`,
      borderRadius: 12,
      height: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: 32,
      textAlign: "center",
      minHeight: 300,
    }}>
      <div style={{ fontSize: 36, marginBottom: 16, opacity: 0.3 }}>⬡</div>
      <div style={{ fontSize: 14, fontWeight: 600, color: TXT2, marginBottom: 8 }}>Select a node</div>
      <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
        Click any component in the diagram to see its description, API endpoints, and ask the AI advisor about it.
      </div>
      <div style={{ marginTop: 24, display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
        {[
          { label: "15 components mapped", color: "#0d2b7a" },
          { label: "5 API categories", color: "#0d9488" },
          { label: "30+ endpoints", color: "#7c3aed" },
        ].map(s => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 10, background: "#f8fafc", borderRadius: 8, padding: "8px 14px" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
            <span style={{ fontSize: 12.5, color: TXT2 }}>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Export ─────────────────────────────────────────────────────────────
export default function PlatformExplorer() {
  const [selected,     setSelected]     = useState(null);
  const [chatMsgs,     setChatMsgs]     = useState([]);
  const [chatInput,    setChatInput]    = useState("");
  const [chatLoading,  setChatLoading]  = useState(false);

  // Reset chat when node changes
  useEffect(() => {
    setChatMsgs([]);
    setChatInput("");
  }, [selected?.id]);

  const handleSelect = (node) => setSelected(node);

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading || !selected) return;
    const userText = chatInput.trim();
    setChatInput("");

    // Build messages — first message carries node context
    const isFirstMsg = chatMsgs.length === 0;
    const apiMessages = isFirstMsg
      ? [{
          role: "user",
          content: `[Platform context: I'm a DAC staff member exploring the "${selected.label}" component. Description: ${selected.description}${selected.endpoints.length ? ` API: ${selected.endpoints.join(", ")}` : ""}]\n\nQuestion: ${userText}`,
        }]
      : [
          // seed context in first message, keep real history
          {
            role: "user",
            content: `[Platform context: ${selected.label} — ${selected.description.slice(0, 120)}…]\n\n${chatMsgs[0]?.content || userText}`,
          },
          ...chatMsgs.slice(1).map(m => ({ role: m.role, content: m.content })),
          { role: "user", content: userText },
        ];

    setChatMsgs(prev => [...prev, { role: "user", content: userText }]);
    setChatLoading(true);

    try {
      const r = await fetch(`${API_URL}/api/v1/advisor/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: apiMessages, model: "haiku" }),
      });
      const data = await r.json();
      setChatMsgs(prev => [...prev, { role: "assistant", content: data.reply || "No response received." }]);
    } catch {
      setChatMsgs(prev => [...prev, { role: "assistant", content: "Unable to reach AI advisor. Check server status." }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div style={{ paddingTop: 72, minHeight: "100vh", background: BGPAGE, fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "40px 24px 60px" }}>

        {/* Page header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: NAVY, margin: 0, letterSpacing: -0.5 }}>
            Platform Explorer
          </h1>
          <p style={{ fontSize: 14, color: TXT2, margin: "6px 0 0" }}>
            Interactive map of all live DAC HealthPrice features — click a component to inspect it.
          </p>
        </div>

        {/* 2-column layout */}
        <div style={{ display: "flex", gap: 24, alignItems: "flex-start" }}>

          {/* Left: SVG diagram */}
          <div style={{ flex: "0 0 63%", minWidth: 0 }}>
            <Diagram selected={selected} onSelect={handleSelect} />
          </div>

          {/* Right: Info panel */}
          <div style={{ flex: "0 0 calc(37% - 24px)", minWidth: 280 }}>
            {selected ? (
              <InfoPanel
                node={selected}
                chatMsgs={chatMsgs}
                chatInput={chatInput}
                chatLoading={chatLoading}
                onChatInput={setChatInput}
                onSend={sendChat}
              />
            ) : (
              <Placeholder />
            )}
          </div>
        </div>

        {/* Summary row */}
        <div style={{
          marginTop: 24,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: 12,
        }}>
          {[
            { label: "Staff Pages",    n: 5,  color: "#0d2b7a" },
            { label: "Workbench Tabs", n: 4,  color: "#0d9488" },
            { label: "Admin Panels",   n: 4,  color: "#7c3aed" },
            { label: "Public Portal",  n: 1,  color: "#059669" },
            { label: "API Endpoints",  n: "30+", color: "#1e293b" },
          ].map(s => (
            <div key={s.label} style={{
              background: BGCARD,
              border: `1px solid ${BORDER}`,
              borderRadius: 10,
              padding: "14px 18px",
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}>
              <span style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.n}</span>
              <span style={{ fontSize: 12, color: TXT2, fontWeight: 500 }}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
