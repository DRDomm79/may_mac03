import { useState, useEffect } from "react";

const API_URL = "https://dac-healthprice-api.onrender.com";
const NAVY   = "#0d2b7a";
const NAVY_D = "#091d5e";
const GOLD   = "#f5a623";
const GOLD_D = "#e67e00";
const WHITE  = "#ffffff";
const GRAY   = "#94a3b8";
const LTGRAY = "#f1f3f5";
const TXT    = "#111827";
const TXT2   = "#4b5563";
const OK     = "#10b981";
const ERR    = "#ef4444";

const TABS = ["System Health", "Model Management", "Vietnam ML", "Rules Management", "Audit Log"];

// ── Mock data (replace with live endpoints when backend is ready) ─────────────

// GET /api/v1/admin/models
const MOCK_MODELS = [
  { id: "health-glm-v2.3", name: "Health GLM",  version: "v2.3", type: "Health", status: "active",   auc: 0.91, accuracy: 0.87, deployed: "2026-04-10", records: 12400 },
  { id: "health-glm-v2.2", name: "Health GLM",  version: "v2.2", type: "Health", status: "archived", auc: 0.88, accuracy: 0.84, deployed: "2026-03-01", records: 9200  },
  { id: "life-v3.0",       name: "Life Pricer", version: "v3.0", type: "Life",   status: "active",   auc: 0.94, accuracy: 0.92, deployed: "2026-04-14", records: 5800  },
  { id: "life-v2.1",       name: "Life Pricer", version: "v2.1", type: "Life",   status: "archived", auc: 0.91, accuracy: 0.89, deployed: "2026-02-20", records: 4100  },
];

// GET /api/v1/admin/rules
const INIT_RULES = [
  { id: "AGE-001",   category: "Age",          name: "Max age limit",            description: "Decline if age > threshold",              severity: "hard", value: 70,  unit: "years",       enabled: true  },
  { id: "AGE-002",   category: "Age",          name: "Min age limit",            description: "Decline if age < threshold",              severity: "hard", value: 18,  unit: "years",       enabled: true  },
  { id: "BMI-001",   category: "BMI",          name: "BMI decline threshold",    description: "Decline if BMI exceeds threshold",        severity: "hard", value: 40,  unit: "kg/m²",       enabled: true  },
  { id: "BMI-002",   category: "BMI",          name: "BMI loading threshold",    description: "+20% loading if BMI over threshold",      severity: "soft", value: 30,  unit: "kg/m²",       enabled: true  },
  { id: "COND-001",  category: "Condition",    name: "Active cancer exclusion",  description: "Decline applicants with active cancer",   severity: "hard", value: null, unit: null,         enabled: true  },
  { id: "COND-002",  category: "Condition",    name: "Multi-condition loading",  description: "+15% per condition beyond threshold",     severity: "soft", value: 2,   unit: "conditions",  enabled: true  },
  { id: "SMOKE-001", category: "Lifestyle",    name: "Smoker loading",           description: "+30% loading for current smokers",        severity: "soft", value: 30,  unit: "%",           enabled: true  },
  { id: "BP-001",    category: "Blood Pressure","name": "Hypertensive crisis",   description: "Decline if systolic exceeds threshold",   severity: "hard", value: 180, unit: "mmHg",        enabled: true  },
  { id: "SCORE-001", category: "Risk Score",   name: "Max risk score",           description: "Decline if composite risk score > value", severity: "hard", value: 8,   unit: "pts",         enabled: true  },
  { id: "SCORE-002", category: "Risk Score",   name: "Manual review threshold",  description: "Flag for UW review if risk score ≥ value",severity: "soft", value: 5,   unit: "pts",         enabled: true  },
];

// GET /api/v1/admin/audit
const MOCK_AUDIT = [
  { id: "AUD-009", user: "admin",   action: "RULE_ENABLED",    target: "SMOKE-001",           ts: new Date(Date.now() - 1 * 3600000).toISOString(),         detail: "Re-enabled smoker loading rule" },
  { id: "AUD-008", user: "radet",   action: "CASE_APPROVED",   target: "DAC-2026-0051",       ts: new Date(Date.now() - 3 * 3600000).toISOString(),         detail: "Approved; borderline BMI, monitoring flagged" },
  { id: "AUD-007", user: "admin",   action: "RULE_DISABLED",   target: "SMOKE-001",           ts: new Date(Date.now() - 12 * 3600000).toISOString(),        detail: "Temporarily disabled smoker loading for pilot" },
  { id: "AUD-006", user: "analyst", action: "DATASET_UPLOAD",  target: "claims_q1_2026.csv",  ts: new Date(Date.now() - 24 * 3600000).toISOString(),        detail: "Uploaded 2,340 IPD claims records" },
  { id: "AUD-005", user: "radet",   action: "CASE_DECLINED",   target: "DAC-2026-0048",       ts: new Date(Date.now() - 2 * 24 * 3600000).toISOString(),   detail: "Declined: composite risk score 9/10" },
  { id: "AUD-004", user: "admin",   action: "MODEL_ACTIVATED", target: "life-v3.0",           ts: new Date(Date.now() - 3 * 24 * 3600000).toISOString(),   detail: "Activated Life Pricer v3.0, replaced v2.1" },
  { id: "AUD-003", user: "radet",   action: "CASE_APPROVED",   target: "DAC-2026-0042",       ts: new Date(Date.now() - 4 * 24 * 3600000).toISOString(),   detail: "Approved case for Sovann Pich, Life Standard" },
  { id: "AUD-002", user: "admin",   action: "RULE_MODIFIED",   target: "BMI-001",             ts: new Date(Date.now() - 6 * 24 * 3600000).toISOString(),   detail: "Changed BMI decline threshold: 38 → 40" },
  { id: "AUD-001", user: "admin",   action: "MODEL_ACTIVATED", target: "health-glm-v2.3",     ts: new Date(Date.now() - 7 * 24 * 3600000).toISOString(),   detail: "Activated Health GLM v2.3, replaced v2.2" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function timeAgo(ts) {
  if (!ts) return "—";
  const mins = Math.floor((Date.now() - new Date(ts)) / 60000);
  if (mins < 60)  return `${mins}m ago`;
  if (mins < 1440) return `${Math.floor(mins / 60)}h ago`;
  return `${Math.floor(mins / 1440)}d ago`;
}

function Badge({ label, color, bg }) {
  return (
    <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 9999, fontSize: 11,
      fontWeight: 700, background: bg, color, textTransform: "uppercase", letterSpacing: 0.5 }}>
      {label}
    </span>
  );
}

const ACTION_STYLE = {
  MODEL_ACTIVATED: { bg: "#dbeafe", color: "#1e40af" },
  RULE_MODIFIED:   { bg: "#fef9c3", color: "#854d0e" },
  RULE_ENABLED:    { bg: "#dcfce7", color: "#166534" },
  RULE_DISABLED:   { bg: "#fee2e2", color: "#991b1b" },
  CASE_APPROVED:   { bg: "#dcfce7", color: "#166534" },
  CASE_DECLINED:   { bg: "#fee2e2", color: "#991b1b" },
  DATASET_UPLOAD:  { bg: "#f3e8ff", color: "#6b21a8" },
};

// ── API-key gate ──────────────────────────────────────────────────────────────

function ApiKeyGate({ onAuth }) {
  const [k, setK] = useState("");
  return (
    <section style={{ paddingTop: 120, paddingBottom: 80, minHeight: "80vh" }}>
      <div style={{ maxWidth: 400, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
        <div style={{ width: 56, height: 56, borderRadius: 16, background: NAVY, display: "flex",
          alignItems: "center", justifyContent: "center", margin: "0 auto 20px" }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C9.24 2 7 4.24 7 7v3H5v12h14V10h-2V7c0-2.76-2.24-5-5-5zm0 2c1.66 0 3 1.34 3 3v3H9V7c0-1.66 1.34-3 3-3z" fill={GOLD}/>
          </svg>
        </div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Admin Console</h2>
        <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>Enter your API key to continue</p>
        <input type="password" placeholder="Admin API key" value={k}
          onChange={e => setK(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && k) onAuth(k); }}
          style={{ width: "100%", padding: "12px 16px", borderRadius: 10, border: "1px solid #d1d5db",
            fontSize: 15, fontFamily: "inherit", marginBottom: 12, outline: "none", boxSizing: "border-box" }} />
        <button onClick={() => { if (k) onAuth(k); }}
          style={{ width: "100%", padding: 14, borderRadius: 8, background: GOLD, color: NAVY,
            border: "none", fontSize: 15, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
          Authenticate
        </button>
      </div>
    </section>
  );
}

// ── Tab: System Health ────────────────────────────────────────────────────────

function SystemHealthTab({ apiKey }) {
  const [health, setHealth]         = useState(null);
  const [uploading, setUploading]   = useState(false);
  const [file, setFile]             = useState(null);
  const [covType, setCovType]       = useState("ipd");
  const [autoRetrain, setAutoRetrain] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [userData, setUserData]     = useState(null);
  const [showTable, setShowTable]   = useState(false);

  const fetchHealth = async () => {
    try {
      const r = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(10000) });
      setHealth(await r.json());
    } catch { setHealth({ error: "Cannot reach API" }); }
  };

  const handleUpload = async () => {
    if (!file || !apiKey) return;
    setUploading(true); setUploadResult(null);
    const fd = new FormData();
    fd.append("file", file); fd.append("coverage_type", covType); fd.append("auto_retrain", autoRetrain);
    try {
      const r = await fetch(`${API_URL}/api/v2/admin/upload-dataset`, {
        method: "POST", headers: { "X-API-Key": apiKey }, body: fd,
      });
      setUploadResult(await r.json());
    } catch (e) { setUploadResult({ status: "error", detail: e.message }); }
    setUploading(false);
  };

  const fetchUsers = async () => {
    try {
      const r = await fetch(`${API_URL}/api/v2/admin/user-behavior`, { headers: { "X-API-Key": apiKey } });
      setUserData(await r.json());
    } catch { setUserData({ status: "error" }); }
  };

  useEffect(() => { fetchHealth(); fetchUsers(); }, []);

  const card = (label, value, ok) => (
    <div style={{ background: LTGRAY, borderRadius: 10, padding: 16 }}>
      <p style={{ fontSize: 12, color: TXT2, marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 16, fontWeight: 600, color: ok === false ? ERR : ok ? OK : TXT }}>{value}</p>
    </div>
  );

  const tierColors = { Bronze: "#92400e", Silver: "#475569", Gold: "#c46800", Platinum: "#1e40af" };
  const tierBg     = { Bronze: "#fffbeb", Silver: "#f1f3f5", Gold: "#fff7ed", Platinum: "#eff6ff" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* API status */}
      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h3 style={h3}>API Status</h3>
          <button onClick={fetchHealth} style={smallBtn}>Refresh</button>
        </div>
        {health ? (
          health.error ? (
            <p style={{ color: ERR, fontSize: 14 }}>{health.error}</p>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
              {card("Status",   health.status || "—",                         health.status === "healthy")}
              {card("Models",   health.models_loaded?.length ?? "—",          (health.models_loaded?.length ?? 0) >= 8)}
              {card("Database", health.database_connected ? "Connected" : "Offline", health.database_connected)}
              {card("Version",  health.model_version || "N/A",                undefined)}
            </div>
          )
        ) : (
          <p style={{ color: TXT2, fontSize: 14 }}>Click Refresh to check status</p>
        )}
      </section>

      {/* Upload claims */}
      <section style={sectionStyle}>
        <h3 style={{ ...h3, marginBottom: 16 }}>Upload Claims Dataset</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <label style={labelStyle}>Coverage type</label>
            <select value={covType} onChange={e => setCovType(e.target.value)} style={inputStyle}>
              {["ipd","opd","dental","maternity"].map(c => <option key={c} value={c}>{c.toUpperCase()}</option>)}
            </select>
          </div>
          <div>
            <label style={labelStyle}>CSV file</label>
            <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} style={inputStyle} />
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 14 }}>
            <input type="checkbox" checked={autoRetrain} onChange={e => setAutoRetrain(e.target.checked)} />
            Auto-retrain model after upload
          </label>
          <button onClick={handleUpload} disabled={!file || uploading}
            style={{ ...goldBtn, opacity: !file || uploading ? 0.5 : 1, cursor: !file || uploading ? "not-allowed" : "pointer" }}>
            {uploading ? "Uploading…" : "Upload dataset"}
          </button>
        </div>
        {uploadResult && (
          <div style={{ marginTop: 14, padding: 14, borderRadius: 10,
            background: uploadResult.status === "accepted" ? "#ecfdf5" : "#fef2f2",
            border: `1px solid ${uploadResult.status === "accepted" ? "#a7f3d0" : "#fecaca"}` }}>
            <pre style={{ fontSize: 12, whiteSpace: "pre-wrap", margin: 0, color: TXT }}>
              {JSON.stringify(uploadResult, null, 2)}
            </pre>
          </div>
        )}
      </section>

      {/* User quote data */}
      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h3 style={h3}>User Quote Data</h3>
          <button onClick={fetchUsers} style={smallBtn}>Refresh</button>
        </div>
        {!userData ? (
          <p style={{ color: TXT2, fontSize: 14 }}>Loading…</p>
        ) : userData.status === "error" ? (
          <p style={{ color: ERR, fontSize: 14 }}>Failed to load user data</p>
        ) : userData.status === "no_db" ? (
          <p style={{ color: TXT2, fontSize: 14 }}>Database not connected</p>
        ) : (() => {
          const { summary = {}, records = [] } = userData;
          if (!records.length) return <p style={{ color: TXT2, fontSize: 14 }}>No quote records yet</p>;
          return (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
                {card("Total quotes", summary.total_quotes || 0, undefined)}
                {card("Avg age",      summary.avg_age || "—",   undefined)}
                {card("OPD rider %",  `${summary.rider_rates?.opd || 0}%`, undefined)}
                {card("Dental %",     `${summary.rider_rates?.dental || 0}%`, undefined)}
              </div>
              {summary.tier_distribution && (
                <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
                  {Object.entries(summary.tier_distribution).map(([tier, count]) => {
                    const pct = Math.round(count / (summary.total_quotes || 1) * 100);
                    return (
                      <div key={tier} style={{ flex: 1, background: tierBg[tier] || LTGRAY, borderRadius: 10, padding: "10px 6px", textAlign: "center" }}>
                        <p style={{ fontSize: 12, fontWeight: 600, color: tierColors[tier] || TXT }}>{tier}</p>
                        <p style={{ fontSize: 18, fontWeight: 700, color: tierColors[tier] || TXT }}>{pct}%</p>
                        <p style={{ fontSize: 10, color: TXT2 }}>{count}</p>
                      </div>
                    );
                  })}
                </div>
              )}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <button onClick={() => setShowTable(t => !t)} style={{ ...goldBtn, padding: "8px 16px", fontSize: 12 }}>
                  {showTable ? "Hide" : "Show"} quotes ({records.length})
                </button>
              </div>
              {showTable && (
                <div style={{ overflowX: "auto", marginTop: 10 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                        {["User","Time","Age","Gender","Region","Smoking","Tier","Riders"].map(h => (
                          <th key={h} style={{ textAlign: "left", padding: "8px 6px", fontSize: 11, color: TXT2, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {records.slice(0, 30).map((r, i) => {
                        const riders = [r.include_opd && "OPD", r.include_dental && "Den", r.include_maternity && "Mat"].filter(Boolean).join(", ") || "—";
                        return (
                          <tr key={i} style={{ borderBottom: "1px solid #f1f3f5" }}>
                            <td style={{ padding: "7px 6px", fontFamily: "monospace", fontSize: 11, color: TXT2 }}>{r.browser_id?.slice(0, 8) || "—"}</td>
                            <td style={{ padding: "7px 6px", fontSize: 11, color: TXT2 }}>{r.created_at ? new Date(r.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                            <td style={{ padding: "7px 6px" }}>{r.age}</td>
                            <td style={{ padding: "7px 6px" }}>{r.gender}</td>
                            <td style={{ padding: "7px 6px", fontSize: 11 }}>{r.region}</td>
                            <td style={{ padding: "7px 6px" }}>{r.smoking}</td>
                            <td style={{ padding: "7px 6px" }}>
                              <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, background: tierBg[r.ipd_tier] || LTGRAY, color: tierColors[r.ipd_tier] || TXT, fontWeight: 600 }}>{r.ipd_tier}</span>
                            </td>
                            <td style={{ padding: "7px 6px", fontSize: 11 }}>{riders}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          );
        })()}
      </section>
    </div>
  );
}

// ── Tab: Model Management ─────────────────────────────────────────────────────

function ModelManagementTab({ vnModels = [] }) {
  const [models, setModels] = useState(MOCK_MODELS);
  const [confirm, setConfirm] = useState(null);
  const [vnStatuses, setVnStatuses] = useState({});
  const [vnConfirm, setVnConfirm] = useState(null);

  const activate = (id) => {
    setModels(prev => prev.map(m => {
      if (m.type !== prev.find(x => x.id === id).type) return m;
      return { ...m, status: m.id === id ? "active" : m.status === "active" ? "archived" : m.status };
    }));
    setConfirm(null);
  };

  const vnActivate = (versionId, modelType) => {
    setVnStatuses(prev => {
      const next = { ...prev };
      vnModels.forEach(m => { if (m.model_type === modelType) next[m.version_id] = "inactive"; });
      next[versionId] = "active";
      return next;
    });
    setVnConfirm(null);
  };

  const vnDeactivate = (versionId) => setVnStatuses(prev => ({ ...prev, [versionId]: "inactive" }));

  const typeGroups = ["Health", "Life"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {typeGroups.map(type => (
        <section key={type} style={sectionStyle}>
          <h3 style={{ ...h3, marginBottom: 16 }}>{type} Models</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Version","Status","AUC","Accuracy","Training Records","Deployed","Action"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "10px 12px", fontSize: 12, color: TXT2, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.filter(m => m.type === type).map(m => (
                  <tr key={m.id} style={{ borderBottom: "1px solid #f1f3f5", background: m.status === "active" ? "#f0fdf4" : WHITE }}>
                    <td style={{ padding: "12px", fontWeight: 600, color: TXT }}>{m.version}</td>
                    <td style={{ padding: "12px" }}>
                      <Badge label={m.status} color={m.status === "active" ? "#166534" : "#6b7280"} bg={m.status === "active" ? "#dcfce7" : "#f3f4f6"} />
                    </td>
                    <td style={{ padding: "12px", fontFamily: "monospace" }}>{m.auc.toFixed(2)}</td>
                    <td style={{ padding: "12px", fontFamily: "monospace" }}>{(m.accuracy * 100).toFixed(1)}%</td>
                    <td style={{ padding: "12px", color: TXT2 }}>{m.records.toLocaleString()}</td>
                    <td style={{ padding: "12px", color: TXT2, fontSize: 12 }}>{m.deployed}</td>
                    <td style={{ padding: "12px" }}>
                      {m.status === "active" ? (
                        <span style={{ fontSize: 12, color: OK, fontWeight: 600 }}>Active</span>
                      ) : confirm === m.id ? (
                        <span style={{ display: "flex", gap: 6 }}>
                          <button onClick={() => activate(m.id)} style={{ padding: "4px 10px", borderRadius: 6, background: NAVY, color: WHITE, border: "none", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>Confirm</button>
                          <button onClick={() => setConfirm(null)} style={{ padding: "4px 10px", borderRadius: 6, background: LTGRAY, color: TXT2, border: "none", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>Cancel</button>
                        </span>
                      ) : (
                        <button onClick={() => setConfirm(m.id)} style={{ padding: "4px 12px", borderRadius: 6, background: GOLD, color: NAVY, border: "none", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Activate</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 11, color: GRAY, marginTop: 10 }}>Source: <code>GET /api/v1/admin/models</code> (mock — wire when backend ready)</p>
        </section>
      ))}

      {/* Vietnam ML models — populated when retraining is triggered in the Vietnam ML tab */}
      {vnModels.length > 0 && (
        <section style={sectionStyle}>
          <h3 style={{ ...h3, marginBottom: 4 }}>Vietnam ML Models</h3>
          <p style={{ fontSize: 12, color: GRAY, marginBottom: 16 }}>Retrained via the Vietnam ML tab. Activate a version to make it live for pricing.</p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Version ID","Model","R²","RMSE","Records","Status","Action"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "10px 12px", fontSize: 12, color: TXT2, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {vnModels.map(v => {
                  const st = vnStatuses[v.version_id] ?? "inactive";
                  return (
                    <tr key={v.version_id} style={{ borderBottom: "1px solid #f1f3f5", background: st === "active" ? "#f0fdf4" : WHITE }}>
                      <td style={{ padding: "12px", fontFamily: "monospace", fontSize: 12, fontWeight: 600, color: NAVY }}>{v.version_id}</td>
                      <td style={{ padding: "12px" }}>
                        <Badge label={v.model_type} color={v.model_type === "health" ? "#1e40af" : "#6b21a8"} bg={v.model_type === "health" ? "#dbeafe" : "#f3e8ff"} />
                      </td>
                      <td style={{ padding: "12px", fontFamily: "monospace" }}>{v.r2?.toFixed(3) ?? "—"}</td>
                      <td style={{ padding: "12px", fontFamily: "monospace" }}>{v.rmse?.toFixed(4) ?? "—"}</td>
                      <td style={{ padding: "12px", color: TXT2 }}>{v.training_records?.toLocaleString() ?? "—"}</td>
                      <td style={{ padding: "12px" }}>
                        <Badge label={st} color={st === "active" ? "#166534" : "#6b7280"} bg={st === "active" ? "#dcfce7" : "#f3f4f6"} />
                      </td>
                      <td style={{ padding: "12px" }}>
                        {st === "active" ? (
                          <button onClick={() => vnDeactivate(v.version_id)} style={{ padding: "4px 10px", borderRadius: 6, background: "#fee2e2", color: "#991b1b", border: "none", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>Deactivate</button>
                        ) : vnConfirm === v.version_id ? (
                          <span style={{ display: "flex", gap: 6 }}>
                            <button onClick={() => vnActivate(v.version_id, v.model_type)} style={{ padding: "4px 10px", borderRadius: 6, background: NAVY, color: WHITE, border: "none", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>Confirm</button>
                            <button onClick={() => setVnConfirm(null)} style={{ padding: "4px 10px", borderRadius: 6, background: LTGRAY, color: TXT2, border: "none", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>Cancel</button>
                          </span>
                        ) : (
                          <button onClick={() => setVnConfirm(v.version_id)} style={{ padding: "4px 12px", borderRadius: 6, background: GOLD, color: NAVY, border: "none", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Activate</button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

// ── Tab: Rules Management ─────────────────────────────────────────────────────

function RulesManagementTab() {
  const [rules, setRules] = useState(INIT_RULES);
  const [editing, setEditing] = useState(null); // rule id
  const [editVal, setEditVal] = useState("");

  const toggle = (id) => setRules(prev => prev.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r));

  const saveEdit = (id) => {
    const num = parseFloat(editVal);
    if (!isNaN(num)) setRules(prev => prev.map(r => r.id === id ? { ...r, value: num } : r));
    setEditing(null);
  };

  const categories = [...new Set(rules.map(r => r.category))];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {[
          { label: "Hard stops", val: rules.filter(r => r.severity === "hard" && r.enabled).length, color: ERR, bg: "#fef2f2" },
          { label: "Soft loadings", val: rules.filter(r => r.severity === "soft" && r.enabled).length, color: GOLD_D, bg: "#fffbeb" },
          { label: "Disabled", val: rules.filter(r => !r.enabled).length, color: GRAY, bg: LTGRAY },
        ].map(s => (
          <div key={s.label} style={{ background: s.bg, borderRadius: 10, padding: "12px 20px", minWidth: 120 }}>
            <p style={{ fontSize: 11, color: TXT2, marginBottom: 2 }}>{s.label}</p>
            <p style={{ fontSize: 24, fontWeight: 700, color: s.color }}>{s.val}</p>
          </div>
        ))}
      </div>

      {categories.map(cat => (
        <section key={cat} style={sectionStyle}>
          <h3 style={{ ...h3, marginBottom: 14 }}>{cat}</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {rules.filter(r => r.category === cat).map(rule => (
              <div key={rule.id} style={{ display: "flex", alignItems: "center", gap: 14, padding: "12px 14px",
                borderRadius: 10, background: rule.enabled ? WHITE : "#f9fafb", border: `1px solid ${rule.enabled ? "#e5e7eb" : "#f3f4f6"}`,
                opacity: rule.enabled ? 1 : 0.6 }}>
                {/* Toggle */}
                <div onClick={() => toggle(rule.id)} style={{ width: 40, height: 22, borderRadius: 11,
                  background: rule.enabled ? OK : "#d1d5db", cursor: "pointer", position: "relative", flexShrink: 0, transition: "background 0.2s" }}>
                  <div style={{ width: 18, height: 18, borderRadius: "50%", background: WHITE, position: "absolute",
                    top: 2, left: rule.enabled ? 20 : 2, transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.2)" }} />
                </div>
                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: TXT }}>{rule.name}</span>
                    <Badge label={rule.severity} color={rule.severity === "hard" ? "#991b1b" : "#854d0e"} bg={rule.severity === "hard" ? "#fee2e2" : "#fef9c3"} />
                    <span style={{ fontSize: 11, color: GRAY, fontFamily: "monospace" }}>{rule.id}</span>
                  </div>
                  <p style={{ fontSize: 12, color: TXT2 }}>{rule.description}</p>
                </div>
                {/* Value editor */}
                {rule.value !== null && (
                  <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                    {editing === rule.id ? (
                      <>
                        <input type="number" value={editVal} onChange={e => setEditVal(e.target.value)}
                          style={{ width: 64, padding: "4px 8px", borderRadius: 6, border: `1px solid ${GOLD}`, fontSize: 13, fontFamily: "inherit", outline: "none" }} />
                        <span style={{ fontSize: 11, color: TXT2 }}>{rule.unit}</span>
                        <button onClick={() => saveEdit(rule.id)} style={{ padding: "3px 8px", borderRadius: 6, background: NAVY, color: WHITE, border: "none", fontSize: 12, cursor: "pointer" }}>Save</button>
                        <button onClick={() => setEditing(null)} style={{ padding: "3px 8px", borderRadius: 6, background: LTGRAY, color: TXT2, border: "none", fontSize: 12, cursor: "pointer" }}>✕</button>
                      </>
                    ) : (
                      <>
                        <span style={{ fontSize: 14, fontWeight: 600, color: NAVY, fontFamily: "monospace" }}>{rule.value}</span>
                        <span style={{ fontSize: 11, color: TXT2 }}>{rule.unit}</span>
                        <button onClick={() => { setEditing(rule.id); setEditVal(String(rule.value)); }}
                          style={{ padding: "3px 8px", borderRadius: 6, background: LTGRAY, color: TXT2, border: "none", fontSize: 11, cursor: "pointer" }}>Edit</button>
                      </>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      ))}
      <p style={{ fontSize: 11, color: GRAY }}>Source: <code>GET /api/v1/admin/rules</code> · <code>PUT /api/v1/admin/rules/:id</code> (mock — wire when backend ready)</p>
    </div>
  );
}

// ── Tab: Audit Log ────────────────────────────────────────────────────────────

function AuditLogTab() {
  const [filter, setFilter] = useState("All");
  const [userFilter, setUserFilter] = useState("All");

  const actionTypes = ["All", ...new Set(MOCK_AUDIT.map(a => a.action))];
  const users       = ["All", ...new Set(MOCK_AUDIT.map(a => a.user))];

  const filtered = MOCK_AUDIT.filter(a =>
    (filter === "All" || a.action === filter) &&
    (userFilter === "All" || a.user === userFilter)
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Filters */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <div>
          <label style={labelStyle}>Action</label>
          <select value={filter} onChange={e => setFilter(e.target.value)} style={{ ...inputStyle, width: "auto", minWidth: 160 }}>
            {actionTypes.map(a => <option key={a} value={a}>{a.replace(/_/g, " ")}</option>)}
          </select>
        </div>
        <div>
          <label style={labelStyle}>User</label>
          <select value={userFilter} onChange={e => setUserFilter(e.target.value)} style={{ ...inputStyle, width: "auto", minWidth: 120 }}>
            {users.map(u => <option key={u} value={u}>{u}</option>)}
          </select>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <span style={{ fontSize: 13, color: TXT2, paddingBottom: 2 }}>{filtered.length} entries</span>
        </div>
      </div>

      {/* Log entries */}
      <section style={sectionStyle}>
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {filtered.length === 0 ? (
            <p style={{ color: TXT2, fontSize: 14, textAlign: "center", padding: 24 }}>No entries match</p>
          ) : filtered.map((entry, i) => {
            const style = ACTION_STYLE[entry.action] ?? { bg: "#f3f4f6", color: "#374151" };
            return (
              <div key={entry.id} style={{ display: "flex", gap: 14, padding: "14px 0",
                borderBottom: i < filtered.length - 1 ? "1px solid #f1f3f5" : "none", alignItems: "flex-start" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: style.color, marginTop: 6, flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
                    <Badge label={entry.action.replace(/_/g, " ")} color={style.color} bg={style.bg} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: NAVY }}>{entry.target}</span>
                    <span style={{ fontSize: 11, color: GRAY }}>by {entry.user}</span>
                  </div>
                  <p style={{ fontSize: 13, color: TXT2 }}>{entry.detail}</p>
                </div>
                <div style={{ flexShrink: 0, textAlign: "right" }}>
                  <p style={{ fontSize: 12, color: GRAY, whiteSpace: "nowrap" }}>{timeAgo(entry.ts)}</p>
                  <p style={{ fontSize: 10, color: GRAY, marginTop: 2 }}>{new Date(entry.ts).toLocaleDateString()}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>
      <p style={{ fontSize: 11, color: GRAY }}>Source: <code>GET /api/v1/admin/audit</code> (mock — wire when backend ready)</p>
    </div>
  );
}

// ── Tab: Vietnam ML ───────────────────────────────────────────────────────────

function VietnamMLTab({ onNewModels = () => {} }) {
  const [versions, setVersions]         = useState([]);
  const [loadingVersions, setLoadingV]  = useState(true);
  const [retraining, setRetraining]     = useState(false);
  const [retrainType, setRetrainType]   = useState("both");
  const [retrainResult, setRetrainResult] = useState(null);

  const fetchVersions = async () => {
    setLoadingV(true);
    try {
      const r = await fetch(`${API_URL}/api/vietnam/model-versions`);
      const data = await r.json();
      setVersions(data.versions || []);
    } catch { setVersions([]); }
    setLoadingV(false);
  };

  useEffect(() => { fetchVersions(); }, []);

  const triggerRetrain = async () => {
    setRetraining(true);
    setRetrainResult(null);
    try {
      const r = await fetch(`${API_URL}/api/vietnam/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: retrainType }),
      });
      const data = await r.json();
      setRetrainResult(data);
      if (data.status === "complete" && data.new_versions?.length) {
        onNewModels(data.new_versions);
      }
      await fetchVersions();
    } catch (e) {
      setRetrainResult({ status: "error", message: e.message });
    }
    setRetraining(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Retrain trigger */}
      <section style={{ background: WHITE, borderRadius: 16, padding: 24, border: "1px solid #e5e7eb" }}>
        <h3 style={{ fontSize: 17, fontWeight: 700, color: TXT, margin: "0 0 6px" }}>Trigger Model Retraining</h3>
        <p style={{ color: TXT2, fontSize: 13, marginBottom: 18 }}>
          Re-run XGBoost training on the latest Vietnam dataset. Results are logged in the version history below.
        </p>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
          <div>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 6 }}>Model</label>
            <select value={retrainType} onChange={e => setRetrainType(e.target.value)}
              style={{ padding: "10px 14px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 14, fontFamily: "inherit", outline: "none", minWidth: 160, background: WHITE }}>
              <option value="both">Both Models</option>
              <option value="health">Health Score only</option>
              <option value="life">Mortality Model only</option>
            </select>
          </div>
          <button onClick={triggerRetrain} disabled={retraining}
            style={{ padding: "11px 24px", borderRadius: 8, background: retraining ? GRAY : GOLD, color: NAVY, border: "none",
              fontSize: 14, fontWeight: 700, cursor: retraining ? "not-allowed" : "pointer", fontFamily: "inherit" }}>
            {retraining ? "Training…" : "Retrain Now"}
          </button>
          <button onClick={fetchVersions} style={{ padding: "11px 16px", borderRadius: 8, background: "#f1f3f5", color: TXT2,
            border: "none", fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}>
            Refresh History
          </button>
        </div>

        {retrainResult && (
          <div style={{ marginTop: 18, padding: 16, borderRadius: 10,
            background: retrainResult.status === "complete" ? "#f0fdf4" : "#fef2f2",
            border: `1px solid ${retrainResult.status === "complete" ? "#a7f3d0" : "#fecaca"}` }}>
            <p style={{ fontSize: 13, fontWeight: 700, color: retrainResult.status === "complete" ? OK : ERR, marginBottom: 6 }}>
              {retrainResult.status === "complete" ? "Retraining complete" : "Retraining failed"}
            </p>
            <p style={{ fontSize: 12, color: TXT2, marginBottom: 10 }}>{retrainResult.message}</p>

            {/* Data-aware cohort summary */}
            {retrainResult.data_summary && (() => {
              const ds = retrainResult.data_summary;
              const excess = ds.smoker_excess_vs_baseline;
              const direction = excess > 0.02 ? "up" : excess < -0.02 ? "down" : "stable";
              const dirColor = direction === "up" ? ERR : direction === "down" ? OK : TXT2;
              const dirLabel = direction === "up"
                ? `↑ +${(excess * 100).toFixed(1)}% vs baseline — elevated RMSE expected`
                : direction === "down"
                ? `↓ ${(excess * 100).toFixed(1)}% vs baseline — tighter fit expected`
                : "≈ stable vs baseline";
              return (
                <div style={{ padding: "8px 12px", background: "rgba(0,0,0,0.04)", borderRadius: 8, fontSize: 12, marginBottom: 10,
                  display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center" }}>
                  <span style={{ color: TXT2 }}>Dataset: <strong>{ds.records?.toLocaleString()} records</strong></span>
                  <span style={{ color: TXT2 }}>Smoker prevalence: <strong>{(ds.smoker_ratio * 100).toFixed(1)}%</strong></span>
                  <span style={{ color: dirColor, fontWeight: 600 }}>{dirLabel}</span>
                </div>
              );
            })()}

            {retrainResult.new_versions?.map(v => (
              <div key={v.version_id} style={{ padding: "8px 12px", background: WHITE, borderRadius: 8, fontSize: 12, marginTop: 6,
                display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontFamily: "monospace", fontWeight: 600, color: NAVY }}>{v.version_id}</span>
                <span style={{ color: TXT2 }}>R² = <strong>{v.r2}</strong></span>
                <span style={{ color: TXT2 }}>RMSE = <strong>{v.rmse}</strong></span>
                <span style={{ color: TXT2 }}>{v.training_records?.toLocaleString()} records</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Version history table */}
      <section style={{ background: WHITE, borderRadius: 16, padding: 24, border: "1px solid #e5e7eb" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h3 style={{ fontSize: 17, fontWeight: 700, color: TXT, margin: 0 }}>Model Version History</h3>
          <span style={{ fontSize: 12, color: GRAY }}>{versions.length} version{versions.length !== 1 ? "s" : ""}</span>
        </div>

        {loadingVersions ? (
          <p style={{ color: TXT2, fontSize: 14 }}>Loading…</p>
        ) : versions.length === 0 ? (
          <p style={{ color: TXT2, fontSize: 14 }}>No versions found. Trigger a retrain or check the backend connection.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Version ID", "Model", "R²", "RMSE", "Records", "Status", "Trained At"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "10px 12px", fontSize: 12, color: TXT2, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.map(v => (
                  <tr key={v.version_id} style={{ borderBottom: "1px solid #f1f3f5", background: v.status === "active" ? "#f0fdf4" : WHITE }}>
                    <td style={{ padding: "12px", fontFamily: "monospace", fontSize: 12, fontWeight: 600, color: NAVY }}>{v.version_id}</td>
                    <td style={{ padding: "12px" }}>
                      <Badge label={v.model_type} color={v.model_type === "health" ? "#1e40af" : "#6b21a8"} bg={v.model_type === "health" ? "#dbeafe" : "#f3e8ff"} />
                    </td>
                    <td style={{ padding: "12px", fontFamily: "monospace" }}>{v.r2?.toFixed(3)}</td>
                    <td style={{ padding: "12px", fontFamily: "monospace" }}>{v.rmse}</td>
                    <td style={{ padding: "12px", color: TXT2 }}>{v.training_records?.toLocaleString()}</td>
                    <td style={{ padding: "12px" }}>
                      <Badge label={v.status} color={v.status === "active" ? "#166534" : "#6b7280"} bg={v.status === "active" ? "#dcfce7" : "#f3f4f6"} />
                    </td>
                    <td style={{ padding: "12px", color: TXT2, fontSize: 12 }}>
                      {v.trained_at ? new Date(v.trained_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p style={{ fontSize: 11, color: GRAY, marginTop: 12 }}>
          Source: <code>GET /api/vietnam/model-versions</code> · history persisted to <code>models/vietnam/version_history.json</code>
        </p>
      </section>
    </div>
  );
}

// ── Shared style tokens ───────────────────────────────────────────────────────

const sectionStyle = {
  background: WHITE, borderRadius: 16, padding: 24, border: "1px solid #e5e7eb",
};
const h3 = { fontSize: 17, fontWeight: 700, color: TXT, margin: 0 };
const labelStyle = { display: "block", fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 6 };
const inputStyle = {
  width: "100%", padding: "10px 14px", borderRadius: 8, border: "1px solid #d1d5db",
  fontSize: 14, fontFamily: "inherit", outline: "none", boxSizing: "border-box",
};
const goldBtn = {
  padding: "11px 24px", borderRadius: 8, background: GOLD, color: NAVY,
  border: "none", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
};
const smallBtn = {
  padding: "7px 16px", borderRadius: 8, background: LTGRAY, color: TXT2,
  border: "none", fontSize: 13, cursor: "pointer", fontFamily: "inherit",
};

// ── Root export ───────────────────────────────────────────────────────────────

export default function AdminConsole() {
  const [key, setKey]   = useState("");
  const [authed, setAuthed] = useState(false);
  const [tab, setTab]   = useState("System Health");
  const [vnModels, setVnModels] = useState([]);

  const addVnModels = (newVersions) => {
    setVnModels(prev => {
      const ids = new Set(prev.map(m => m.version_id));
      return [...prev, ...newVersions.filter(m => !ids.has(m.version_id))];
    });
  };

  if (!authed) return <ApiKeyGate onAuth={k => { setKey(k); setAuthed(true); }} />;

  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "80vh" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px" }}>

        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ color: GOLD_D, fontSize: 13, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Internal</span>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 34, fontWeight: 700, color: TXT, marginTop: 6 }}>Admin Console</h2>
          <p style={{ color: TXT2, fontSize: 14, marginTop: 4 }}>Manage models, underwriting rules, and system health</p>
        </div>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "2px solid #e5e7eb", paddingBottom: 0 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "10px 18px", borderRadius: "8px 8px 0 0", border: "none", fontSize: 14, fontWeight: tab === t ? 700 : 500,
              color: tab === t ? NAVY : TXT2, background: tab === t ? WHITE : "transparent",
              cursor: "pointer", fontFamily: "inherit",
              borderBottom: tab === t ? `2px solid ${NAVY}` : "2px solid transparent",
              marginBottom: -2,
            }}>
              {t}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "System Health"     && <SystemHealthTab apiKey={key} />}
        {tab === "Model Management"  && <ModelManagementTab vnModels={vnModels} />}
        {tab === "Vietnam ML"        && <VietnamMLTab onNewModels={addVnModels} />}
        {tab === "Rules Management"  && <RulesManagementTab />}
        {tab === "Audit Log"         && <AuditLogTab />}
      </div>
    </section>
  );
}
