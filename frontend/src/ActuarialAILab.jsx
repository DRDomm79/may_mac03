import { useState, useRef, useEffect, useCallback } from "react";

const API = "https://dac-healthprice-api-49dr.onrender.com";

// ── Colors ──────────────────────────────────────────────────────────────────
const C = {
  bg: "#fafbfc", sidebar: "#ffffff", border: "#e8ecf1", borderLight: "#f0f2f5",
  navy: "#0d2b7a", navyLight: "#1a4fba", gold: "#f5a623", goldLight: "#fef3d6",
  text: "#1a1a2e", text2: "#5a6474", text3: "#8b95a5",
  white: "#ffffff", green: "#10b981", greenBg: "#ecfdf5",
  red: "#ef4444", redBg: "#fef2f2", blue: "#3b82f6", blueBg: "#eff6ff",
  codeBg: "#1e1e2e", codeText: "#cdd6f4",
};

// ── Actuarial Quick Actions ─────────────────────────────────────────────────
const SUGGESTIONS = [
  { label: "Clean & prepare data", prompt: "Clean this dataset for actuarial modeling: handle missing values with appropriate imputation, remove duplicates, fix data types, cap outliers at the 99th percentile, and encode categorical variables. Show a before/after summary with data quality metrics.", color: "#8b5cf6" },
  { label: "Exploratory data analysis", prompt: "Perform comprehensive EDA: plot distributions for all numeric columns (use subplots), create a correlation heatmap, show value counts for categorical columns, and generate claim rate analysis by key risk factors. Save all charts to /tmp/ailab_output/. Use a clean actuarial style with the color palette ['#0d2b7a','#f5a623','#3b82f6','#10b981','#8b5cf6','#ef4444'].", color: "#3b82f6" },
  { label: "Frequency model (Poisson GLM)", prompt: "Build a Poisson GLM frequency model to predict claim count/occurrence. Split data 80/20, fit the model, show coefficients with interpretation, calculate AUC-ROC and classification metrics. Plot the ROC curve and feature importance. Save charts to /tmp/ailab_output/. Interpret from an actuarial pricing perspective.", color: "#10b981" },
  { label: "Severity model (Gamma GLM + GBR)", prompt: "Build two severity models on claims with amount > 0: a Gamma GLM (actuarial standard) and a Gradient Boosting Regressor (n_estimators=150, max_depth=4, lr=0.07). Compare MAE, RMSE, R-squared. Plot actual vs predicted and residuals. Save charts to /tmp/ailab_output/. Recommend which model to use for production pricing.", color: "#f59e0b" },
  { label: "Full pricing (Freq x Sev)", prompt: "Build a complete frequency-severity pricing pipeline: Poisson GLM for frequency, best severity model for cost. Calculate Expected Annual Cost = E[Frequency] x E[Severity] for each record. Add expense loading (30%), risk margin (10%), and profit margin (5%). Show a premium rating table by age group and risk class. Save charts to /tmp/ailab_output/.", color: "#ec4899" },
  { label: "Experience study (A vs E)", prompt: "Perform an Actual vs Expected analysis: calculate actual claim rates by age group, gender, and other risk factors. Compare against expected rates. Flag segments where A/E ratio deviates more than 10% from 1.0. Plot A/E ratios with confidence bands. Save charts to /tmp/ailab_output/.", color: "#06b6d4" },
];

function extractCodeBlocks(text) {
  const blocks = [];
  const regex = /```python\n?([\s\S]*?)```/g;
  let m;
  while ((m = regex.exec(text)) !== null) blocks.push(m[1].trim());
  return blocks;
}

function removeCodeBlocks(text) {
  return text.replace(/```python\n?[\s\S]*?```/g, "").trim();
}

function renderMarkdown(text) {
  if (!text) return "";
  let h = text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, `<code style="background:${C.borderLight};padding:2px 7px;border-radius:4px;font-size:12.5px;color:${C.navy}">$1</code>`)
    .replace(/\n/g, "<br/>");
  return h;
}

// ── Parse stdout into structured blocks (tables, numbers, text) ─────────────
function parseOutput(stdout) {
  if (!stdout) return [];
  const lines = stdout.split("\n");
  const blocks = [];
  let tableLines = [];
  let textLines = [];

  const flushText = () => { if (textLines.length) { blocks.push({ type: "text", content: textLines.join("\n") }); textLines = []; } };
  const flushTable = () => { if (tableLines.length) { blocks.push({ type: "table", content: tableLines.join("\n") }); tableLines = []; } };

  for (const line of lines) {
    // Detect table-like lines (has multiple spaces or tabs separating values, or has | separators)
    const isTableLine = (line.includes("  ") && line.trim().split(/\s{2,}/).length >= 3) || line.includes("|");
    if (isTableLine && line.trim()) {
      flushText();
      tableLines.push(line);
    } else {
      flushTable();
      textLines.push(line);
    }
  }
  flushText();
  flushTable();
  return blocks;
}

function OutputTable({ content }) {
  const lines = content.split("\n").filter(l => l.trim());
  if (lines.length < 2) return <pre style={{ fontSize: 13, color: C.text2, margin: 0, fontFamily: "inherit" }}>{content}</pre>;

  // Try to parse space-separated table
  const rows = lines.map(l => l.trim().split(/\s{2,}|\s*\|\s*/));
  const maxCols = Math.max(...rows.map(r => r.length));

  return (
    <div style={{ overflowX: "auto", borderRadius: 10, border: `1px solid ${C.border}`, margin: "8px 0" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: C.borderLight }}>
            {rows[0].map((cell, i) => (
              <th key={i} style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: C.navy, borderBottom: `2px solid ${C.border}`, whiteSpace: "nowrap" }}>{cell}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(1).map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 0 ? C.white : C.bg }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{ padding: "9px 16px", borderBottom: `1px solid ${C.borderLight}`, color: C.text, whiteSpace: "nowrap" }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────
export default function ActuarialAILab() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showCodeFor, setShowCodeFor] = useState({});
  const [dragOver, setDragOver] = useState(false);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── File Upload ─────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (selectedFile) => {
    if (!selectedFile) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const res = await fetch(`${API}/api/v2/ailab/upload`, { method: "POST", body: formData });
      const data = await res.json();
      setFile(selectedFile);
      setFileMeta(data.meta);
      setPreview(data.preview);
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: "system",
        type: "file_uploaded",
        meta: data.meta,
        preview: data.preview,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { id: Date.now(), role: "system", type: "error", content: `Upload failed: ${e.message}` }]);
    }
    setUploading(false);
  }, []);

  // ── Code Execution ──────────────────────────────────────────────────────
  const executeCode = useCallback(async (code, msgId) => {
    try {
      const res = await fetch(`${API}/api/v2/ailab/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code })
      });
      const data = await res.json();
      // Attach results to the parent message
      setMessages(prev => prev.map(m => m.id === msgId ? {
        ...m,
        execution: { stdout: data.stdout, stderr: data.stderr, charts: data.charts || [], success: data.success },
        executed: true
      } : m));
    } catch (e) {
      setMessages(prev => prev.map(m => m.id === msgId ? {
        ...m,
        execution: { stdout: "", stderr: e.message, charts: [], success: false },
        executed: true
      } : m));
    }
  }, []);

  // ── Send Message ────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (customPrompt) => {
    const msg = customPrompt || input.trim();
    if (!msg || loading) return;
    if (!customPrompt) setInput("");

    const userMsgId = Date.now();
    setMessages(prev => [...prev, { id: userMsgId, role: "user", content: msg }]);
    setLoading(true);

    try {
      const history = messages.filter(m => m.role === "user" || m.role === "assistant").slice(-10).map(m => ({ role: m.role, content: m.content }));
      const res = await fetch(`${API}/api/v2/ailab/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, history, filename: fileMeta?.filename || "" })
      });
      const data = await res.json();
      const aiMsgId = Date.now() + 1;
      const codeBlocks = extractCodeBlocks(data.response);
      const explanation = removeCodeBlocks(data.response);

      const aiMsg = {
        id: aiMsgId,
        role: "assistant",
        content: data.response,
        explanation,
        codeBlocks,
        hasCode: codeBlocks.length > 0,
        executed: false,
        execution: null,
      };
      setMessages(prev => [...prev, aiMsg]);

      // Auto-execute code
      if (codeBlocks.length > 0) {
        const allCode = codeBlocks.join("\n\n");
        await executeCode(allCode, aiMsgId);
      }
    } catch (e) {
      setMessages(prev => [...prev, { id: Date.now() + 1, role: "assistant", explanation: `I couldn't connect to the backend. Make sure the API is running at ${API}.\n\nError: ${e.message}`, content: "", codeBlocks: [] }]);
    }
    setLoading(false);
  }, [input, messages, fileMeta, executeCode, loading]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  };

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div
      style={{ minHeight: "100vh", background: C.bg, fontFamily: "'DM Sans', -apple-system, sans-serif", color: C.text, display: "flex", flexDirection: "column" }}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <style>{`
        .ailab * { box-sizing: border-box; }
        .ailab-input:focus { outline: none; border-color: ${C.navy} !important; box-shadow: 0 0 0 3px rgba(13,43,122,0.08); }
        .ailab-suggestion { transition: all 0.2s; cursor: pointer; }
        .ailab-suggestion:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important; }
        .ailab-chart { border-radius: 12px; transition: transform 0.2s; cursor: pointer; }
        .ailab-chart:hover { transform: scale(1.01); }
        .ailab-code-toggle { transition: all 0.15s; }
        .ailab-code-toggle:hover { background: ${C.borderLight} !important; }
        .ailab-fade-in { animation: ailabFadeIn 0.35s ease-out; }
        @keyframes ailabFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .ailab-typing { display: inline-flex; gap: 4px; align-items: center; }
        .ailab-typing span { width: 6px; height: 6px; border-radius: 50%; background: ${C.text3}; animation: typingDot 1.2s infinite; }
        .ailab-typing span:nth-child(2) { animation-delay: 0.2s; }
        .ailab-typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingDot { 0%, 60%, 100% { opacity: 0.3; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-4px); } }
        .ailab-scrollbar::-webkit-scrollbar { width: 5px; }
        .ailab-scrollbar::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
      `}</style>

      {/* Drag overlay */}
      {dragOver && (
        <div style={{ position: "fixed", inset: 0, zIndex: 999, background: "rgba(13,43,122,0.08)", display: "flex", alignItems: "center", justifyContent: "center", backdropFilter: "blur(4px)" }}>
          <div style={{ background: C.white, borderRadius: 20, padding: "48px 64px", border: `2px dashed ${C.navy}`, textAlign: "center", boxShadow: "0 24px 64px rgba(0,0,0,0.1)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>+</div>
            <div style={{ fontSize: 18, fontWeight: 600, color: C.navy }}>Drop your file here</div>
            <div style={{ fontSize: 13, color: C.text3, marginTop: 4 }}>CSV or Excel files</div>
          </div>
        </div>
      )}

      <div className="ailab" style={{ flex: 1, maxWidth: 860, margin: "0 auto", width: "100%", display: "flex", flexDirection: "column", padding: "0 20px" }}>

        {/* Chat Messages */}
        <div style={{ flex: 1, overflowY: "auto", paddingTop: 32, paddingBottom: 16 }} className="ailab-scrollbar">

          {/* Empty State */}
          {messages.length === 0 && (
            <div style={{ textAlign: "center", paddingTop: "12vh" }} className="ailab-fade-in">
              <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 56, height: 56, borderRadius: 16, background: `linear-gradient(135deg, ${C.navy}, ${C.navyLight})`, marginBottom: 20 }}>
                <span style={{ fontSize: 22, color: C.white, fontWeight: 800 }}>AI</span>
              </div>
              <div style={{ fontSize: 26, fontWeight: 700, color: C.text, marginBottom: 6 }}>Actuarial AI Lab</div>
              <div style={{ fontSize: 15, color: C.text3, maxWidth: 480, margin: "0 auto", lineHeight: 1.6 }}>
                Upload your insurance data and I'll help you clean, analyze, model, and price — built for actuaries.
              </div>

              {/* Upload prompt */}
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{ display: "inline-flex", alignItems: "center", gap: 10, margin: "28px auto 0", padding: "14px 28px", borderRadius: 12,
                  border: `2px dashed ${C.border}`, background: C.white, cursor: "pointer", transition: "all 0.2s" }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = C.gold; e.currentTarget.style.background = C.goldLight; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = C.white; }}
              >
                <span style={{ fontSize: 20 }}>+</span>
                <span style={{ fontSize: 14, fontWeight: 600, color: C.text }}>Upload CSV or Excel</span>
              </div>
              <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls" style={{ display: "none" }} onChange={e => uploadFile(e.target.files[0])} />

              {/* Suggestion chips */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", marginTop: 36, maxWidth: 620, margin: "36px auto 0" }}>
                {SUGGESTIONS.map((s, i) => (
                  <button key={i} className="ailab-suggestion"
                    onClick={() => { if (fileMeta) sendMessage(s.prompt); else fileInputRef.current?.click(); }}
                    style={{ padding: "10px 18px", borderRadius: 20, border: `1px solid ${C.border}`, background: C.white,
                      color: C.text, fontSize: 13, fontWeight: 500, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 6,
                      boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg) => (
            <div key={msg.id} className="ailab-fade-in" style={{ marginBottom: 24 }}>

              {/* ─── File Upload Message ─────────────────────────────── */}
              {msg.type === "file_uploaded" && (
                <div style={{ background: C.white, borderRadius: 16, border: `1px solid ${C.border}`, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.04)" }}>
                  <div style={{ padding: "16px 20px", display: "flex", alignItems: "center", gap: 14, borderBottom: `1px solid ${C.borderLight}` }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: C.greenBg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>
                      +
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{msg.meta.filename}</div>
                      <div style={{ fontSize: 12, color: C.text3 }}>{msg.meta.rows?.toLocaleString()} rows x {msg.meta.columns?.length} columns</div>
                    </div>
                    <div style={{ padding: "4px 12px", borderRadius: 20, background: C.greenBg, color: C.green, fontSize: 12, fontWeight: 600 }}>Ready</div>
                  </div>

                  {/* Column chips */}
                  <div style={{ padding: "12px 20px", display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {msg.meta.columns?.map(col => (
                      <span key={col} style={{ padding: "4px 10px", borderRadius: 6, background: C.bg, border: `1px solid ${C.borderLight}`, fontSize: 11, color: C.text2, fontWeight: 500 }}>
                        {col}
                        <span style={{ color: C.text3, marginLeft: 4, fontSize: 10 }}>{msg.meta.dtypes?.[col]?.replace("float64", "float").replace("int64", "int").replace("object", "str")}</span>
                      </span>
                    ))}
                  </div>

                  {/* Mini preview table */}
                  {msg.preview && (
                    <div style={{ overflowX: "auto", maxHeight: 200, borderTop: `1px solid ${C.borderLight}` }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                        <thead>
                          <tr>
                            {msg.meta.columns?.map(col => (
                              <th key={col} style={{ padding: "8px 14px", textAlign: "left", fontWeight: 600, color: C.navy, background: C.bg, borderBottom: `1px solid ${C.border}`, whiteSpace: "nowrap", position: "sticky", top: 0 }}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {msg.preview.slice(0, 5).map((row, i) => (
                            <tr key={i}>
                              {msg.meta.columns?.map(col => (
                                <td key={col} style={{ padding: "6px 14px", borderBottom: `1px solid ${C.borderLight}`, color: C.text2, whiteSpace: "nowrap" }}>
                                  {row[col] === null || row[col] === undefined ? <span style={{ color: C.text3 }}>—</span> : String(row[col]).slice(0, 30)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* ─── Error Message ───────────────────────────────────── */}
              {msg.type === "error" && (
                <div style={{ padding: "12px 18px", borderRadius: 12, background: C.redBg, border: `1px solid #fecaca`, color: C.red, fontSize: 13 }}>{msg.content}</div>
              )}

              {/* ─── User Message ───────────────────────────────────── */}
              {msg.role === "user" && (
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <div style={{ maxWidth: "75%", padding: "12px 18px", borderRadius: "18px 18px 4px 18px", background: C.navy, color: C.white, fontSize: 14, lineHeight: 1.6 }}>
                    {msg.content}
                  </div>
                </div>
              )}

              {/* ─── Assistant Message ──────────────────────────────── */}
              {msg.role === "assistant" && (
                <div style={{ display: "flex", gap: 14 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 10, background: `linear-gradient(135deg, ${C.navy}, ${C.navyLight})`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <span style={{ fontSize: 11, fontWeight: 800, color: C.white }}>AI</span>
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Explanation text */}
                    {msg.explanation && (
                      <div style={{ fontSize: 14, color: C.text, lineHeight: 1.75, marginBottom: msg.hasCode ? 12 : 0 }}
                        dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.explanation) }} />
                    )}

                    {/* Code toggle */}
                    {msg.hasCode && (
                      <button className="ailab-code-toggle"
                        onClick={() => setShowCodeFor(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                        style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 8,
                          border: `1px solid ${C.border}`, background: C.white, color: C.text2, fontSize: 12, fontWeight: 500,
                          cursor: "pointer", fontFamily: "inherit", marginBottom: 12 }}>
                        <span style={{ fontFamily: "monospace", fontSize: 11 }}>&lt;/&gt;</span>
                        {showCodeFor[msg.id] ? "Hide code" : "View code"}
                      </button>
                    )}

                    {/* Code blocks (hidden by default, Julius-style) */}
                    {showCodeFor[msg.id] && msg.codeBlocks?.map((code, ci) => (
                      <div key={ci} style={{ borderRadius: 12, overflow: "hidden", marginBottom: 12, border: `1px solid #2d2d3f` }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 16px", background: "#161622" }}>
                          <span style={{ fontSize: 11, color: "#6b6b8a", fontFamily: "monospace" }}>Python</span>
                          <button onClick={() => navigator.clipboard.writeText(code)}
                            style={{ padding: "3px 10px", borderRadius: 5, background: "transparent", border: `1px solid #2d2d3f`, color: "#8b8baa", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>
                            Copy
                          </button>
                        </div>
                        <pre style={{ padding: "14px 16px", margin: 0, background: C.codeBg, fontSize: 12, color: C.codeText, overflowX: "auto", fontFamily: "'Fira Code', 'SF Mono', 'Consolas', monospace", lineHeight: 1.55 }}>{code}</pre>
                      </div>
                    ))}

                    {/* Execution Results — shown inline like Julius */}
                    {msg.execution && (
                      <div style={{ marginTop: 4 }}>
                        {/* Stdout as structured output */}
                        {msg.execution.stdout && (
                          <div style={{ background: C.white, borderRadius: 12, border: `1px solid ${C.border}`, overflow: "hidden", marginBottom: 12 }}>
                            <div style={{ padding: "14px 18px" }}>
                              {parseOutput(msg.execution.stdout).map((block, bi) => (
                                <div key={bi}>
                                  {block.type === "table" ? (
                                    <OutputTable content={block.content} />
                                  ) : (
                                    <pre style={{ fontSize: 13, color: C.text2, margin: "4px 0", fontFamily: "inherit", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{block.content}</pre>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Error */}
                        {msg.execution.stderr && !msg.execution.success && (
                          <div style={{ padding: "12px 16px", borderRadius: 10, background: C.redBg, border: `1px solid #fecaca`, marginBottom: 12 }}>
                            <pre style={{ fontSize: 12, color: C.red, margin: 0, fontFamily: "monospace", whiteSpace: "pre-wrap" }}>{msg.execution.stderr}</pre>
                          </div>
                        )}

                        {/* Charts — displayed inline like Julius */}
                        {msg.execution.charts?.length > 0 && (
                          <div style={{ display: "grid", gridTemplateColumns: msg.execution.charts.length === 1 ? "1fr" : "1fr 1fr", gap: 12, marginBottom: 8 }}>
                            {msg.execution.charts.map((chart, ci) => (
                              <div key={ci} style={{ background: C.white, borderRadius: 14, border: `1px solid ${C.border}`, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.04)" }}>
                                <img className="ailab-chart" src={chart.data} alt={chart.filename}
                                  style={{ width: "100%", display: "block" }}
                                  onClick={() => {
                                    const w = window.open();
                                    w.document.write(`<img src="${chart.data}" style="max-width:100%;background:#fff" />`);
                                  }}
                                />
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Loading execution */}
                    {msg.hasCode && !msg.executed && msg.role === "assistant" && !loading && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px", borderRadius: 10, background: C.blueBg, border: `1px solid #bfdbfe`, marginTop: 8 }}>
                        <div style={{ width: 14, height: 14, borderRadius: "50%", border: `2px solid ${C.blue}`, borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }} />
                        <span style={{ fontSize: 13, color: C.blue, fontWeight: 500 }}>Running analysis...</span>
                        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div style={{ display: "flex", gap: 14, marginBottom: 24 }} className="ailab-fade-in">
              <div style={{ width: 32, height: 32, borderRadius: 10, background: `linear-gradient(135deg, ${C.navy}, ${C.navyLight})`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <span style={{ fontSize: 11, fontWeight: 800, color: C.white }}>AI</span>
              </div>
              <div style={{ padding: "14px 0" }}>
                <div className="ailab-typing">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* ─── Input Area ─────────────────────────────────────────────── */}
        <div style={{ padding: "12px 0 24px", position: "sticky", bottom: 0, background: C.bg }}>
          {/* File chip */}
          {fileMeta && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
              <div style={{ padding: "5px 14px", borderRadius: 20, background: C.white, border: `1px solid ${C.border}`, fontSize: 12, color: C.text2, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.green }} />
                {fileMeta.filename} ({fileMeta.rows?.toLocaleString()} rows)
                <span onClick={() => { setFile(null); setFileMeta(null); setPreview(null); }}
                  style={{ marginLeft: 4, cursor: "pointer", color: C.text3, fontWeight: 600 }}>x</span>
              </div>
            </div>
          )}

          {/* Suggestion chips after file upload */}
          {fileMeta && messages.length <= 2 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
              {SUGGESTIONS.slice(0, 4).map((s, i) => (
                <button key={i} className="ailab-suggestion"
                  onClick={() => sendMessage(s.prompt)}
                  style={{ padding: "8px 16px", borderRadius: 20, border: `1px solid ${C.border}`, background: C.white,
                    color: C.text, fontSize: 12.5, fontWeight: 500, fontFamily: "inherit", cursor: "pointer",
                    display: "flex", alignItems: "center", gap: 6, boxShadow: "0 1px 2px rgba(0,0,0,0.03)" }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
                  {s.label}
                </button>
              ))}
            </div>
          )}

          {/* Input box */}
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
            <button onClick={() => fileInputRef.current?.click()}
              style={{ width: 44, height: 44, borderRadius: 12, border: `1px solid ${C.border}`, background: C.white,
                color: C.text3, fontSize: 20, cursor: "pointer", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: "inherit", transition: "all 0.15s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.gold; e.currentTarget.style.color = C.gold; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.text3; }}
              title="Upload file">
              +
            </button>
            <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls" style={{ display: "none" }} onChange={e => uploadFile(e.target.files[0])} />

            <div style={{ flex: 1, position: "relative" }}>
              <textarea
                ref={inputRef}
                className="ailab-input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={fileMeta ? "Ask about your data... e.g., 'Build a frequency model'" : "Upload a file to get started, or ask a general actuarial question"}
                rows={1}
                style={{ width: "100%", padding: "12px 50px 12px 16px", borderRadius: 14, border: `1.5px solid ${C.border}`,
                  background: C.white, color: C.text, fontSize: 14, fontFamily: "inherit", resize: "none",
                  minHeight: 44, maxHeight: 120, lineHeight: 1.4 }}
              />
              <button onClick={() => sendMessage()} disabled={loading || !input.trim()}
                style={{ position: "absolute", right: 6, bottom: 6, width: 34, height: 34, borderRadius: 10, border: "none",
                  background: input.trim() && !loading ? C.navy : C.borderLight,
                  color: input.trim() && !loading ? C.white : C.text3,
                  fontSize: 16, fontWeight: 700, cursor: input.trim() && !loading ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s" }}>
                {loading ? (
                  <div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid #fff", borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }} />
                ) : ">"}
              </button>
            </div>
          </div>

          <div style={{ textAlign: "center", marginTop: 10 }}>
            <span style={{ fontSize: 11, color: C.text3 }}>DAC Actuarial AI Lab — powered by Groq + Python. Results are AI-generated and should be reviewed by a qualified actuary.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
