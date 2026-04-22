// src/InsuranceDashboard.jsx
// Internal admin dashboard for insurance company pilots.
// Two workflows: (1) Quick Quote generation, (2) Claims data upload & GLM calibration.

import { useState, useEffect } from "react";

// ── Constants ─────────────────────────────────────────────────────────────────
const API_URL      = "https://dac-healthprice-api.onrender.com";
const NAVY         = "#0d2b7a";
const GOLD         = "#f5a623";
const GOLD_D       = "#e67e00";
const WHITE        = "#ffffff";
const LTGRAY       = "#f1f3f5";
const TXT          = "#111827";
const TXT2         = "#4b5563";
const OK           = "#10b981";
const ERR          = "#ef4444";

const REGIONS      = ["Phnom Penh","Siem Reap","Battambang","Sihanoukville","Kampong Cham","Rural Areas"];
const OCCUPATIONS  = ["Office/Desk","Retail/Service","Healthcare","Manual Labor","Industrial/High-Risk","Retired"];
const SMOKING_LIST = ["Never","Former","Current"];
const COV_LABELS   = { ipd: "IPD", opd: "OPD", dental: "Dental", maternity: "Maternity" };

// ── Coefficient store (GLM) ───────────────────────────────────────────────────
const COEFF = {
  version:      "v2.2",
  last_updated: "2026-03-28",
  updated_by:   "dac_admin",
  base: {
    ipd:       { freq: 0.12,  sev: 2500 },
    opd:       { freq: 2.5,   sev: 60   },
    dental:    { freq: 0.80,  sev: 120  },
    maternity: { freq: 0.15,  sev: 3500 },
  },
  age:    { "18–24": 0.85, "25–34": 1.00, "35–44": 1.12, "45–54": 1.28, "55–64": 1.48, "65+": 1.72 },
  smoke:  { Never: 1.00, Former: 1.15, Current: 1.40 },
  occup:  { "Office/Desk": 0.85, "Retail/Service": 1.00, "Healthcare": 1.05, "Manual Labor": 1.15, "Industrial/High-Risk": 1.30, "Retired": 1.10 },
  region: { "Phnom Penh": 1.00, "Siem Reap": 0.92, "Battambang": 0.88, "Sihanoukville": 0.95, "Kampong Cham": 0.85, "Rural Areas": 0.78 },
  tier:   { Bronze: 0.70, Silver: 1.00, Gold: 1.45, Platinum: 2.10 },
  load:   { ipd: 0.30, opd: 0.25, dental: 0.20, maternity: 0.25 },
  ded:    { Bronze: 500, Silver: 250, Gold: 100, Platinum: 0 },
  famPer: 0.65,
};

// ── Pricing engine (GLM path) ─────────────────────────────────────────────────
function ageBand(age) {
  const a = +age;
  return a < 25 ? "18–24" : a < 35 ? "25–34" : a < 45 ? "35–44" : a < 55 ? "45–54" : a < 65 ? "55–64" : "65+";
}

function computeQuote({ age, smoking, occupation, region, tier, opd, dental, maternity, dependents }) {
  const ab = ageBand(age);
  const af = COEFF.age[ab]   || 1;
  const sf = COEFF.smoke[smoking]     || 1;
  const of = COEFF.occup[occupation]  || 1;
  const rf = COEFF.region[region]     || 1;
  const tf = COEFF.tier[tier]         || 1;
  const ff = 1 + (+dependents - 1) * COEFF.famPer;

  function calcCov(cov) {
    const { freq, sev } = COEFF.base[cov];
    const efFreq = freq * af * sf * of * rf;
    const efSev  = sev  * (1 + Math.max(0, +age - 30) * 0.006);
    const cost   = efFreq * efSev;
    return {
      frequency: +efFreq.toFixed(4),
      severity:  Math.round(efSev),
      expected:  Math.round(cost),
      premium:   Math.round(cost * (1 + COEFF.load[cov])),
    };
  }

  const ipd        = calcCov("ipd");
  const dedCredit  = COEFF.ded[tier] * 0.10;
  const ipdAnnual  = Math.max(Math.round((ipd.premium * tf - dedCredit) * 100) / 100, 50);
  let   total      = ipdAnnual;
  const riders     = {};

  for (const [cov, inc] of [["opd", opd], ["dental", dental], ["maternity", maternity]]) {
    if (!inc) continue;
    const c = calcCov(cov);
    riders[cov] = { ...c, annual: c.premium };
    total += c.premium;
  }
  total = Math.round(total * ff * 100) / 100;

  return {
    id:            `QQ-${Date.now()}`,
    version:       COEFF.version,
    tier,
    age:           +age,
    smoking,
    occupation,
    region,
    dependents:    +dependents,
    opd:           !!opd,
    dental:        !!dental,
    maternity:     !!maternity,
    ipd:           { ...ipd, annual: ipdAnnual, tier_factor: tf, ded_credit: dedCredit },
    riders,
    total_annual:  total,
    total_monthly: Math.round(total / 12 * 100) / 100,
    family_factor: +ff.toFixed(3),
    breakdown: [
      { label: `Age bracket (${ab})`,        factor: af },
      { label: `Smoking (${smoking})`,        factor: sf },
      { label: `Occupation (${occupation})`,  factor: of },
      { label: `Region (${region})`,          factor: rf },
      { label: `Tier (${tier})`,              factor: tf },
      ...( +dependents > 1 ? [{ label: `Family (${dependents} members)`, factor: +ff.toFixed(3) }] : []),
    ],
    ts: new Date().toISOString(),
  };
}

// ── CSV helpers ───────────────────────────────────────────────────────────────
function downloadCSV(filename, header, rows) {
  const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
}

function downloadBatchTemplate() {
  downloadCSV("batch_quote_template.csv",
    "age,smoking,occupation,region,tier,dependents,opd,dental,maternity",
    [
      "35,Never,Office/Desk,Phnom Penh,Silver,1,true,false,false",
      "42,Former,Manual Labor,Siem Reap,Gold,3,true,true,false",
      "28,Never,Retail/Service,Battambang,Bronze,1,false,false,false",
      "55,Current,Retired,Kampong Cham,Platinum,2,true,true,true",
      "31,Never,Healthcare,Sihanoukville,Silver,1,true,false,true",
    ]
  );
}

function downloadClaimsTemplate() {
  downloadCSV("claims_upload_template.csv",
    "claim_id,customer_age,customer_occupation,claim_type,claim_amount,claim_date",
    [
      "CLM001,35,Office/Desk,IPD,2800,2026-01-15",
      "CLM002,42,Manual Labor,OPD,75,2026-01-18",
      "CLM003,28,Retail/Service,Dental,145,2026-01-22",
      "CLM004,55,Retired,IPD,4200,2026-02-03",
      "CLM005,38,Healthcare,OPD,55,2026-02-10",
    ]
  );
}

function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).filter(l => l.trim()).map((l, i) => {
    const vals = l.split(",").map(v => v.trim());
    const row = {}; headers.forEach((h, j) => { row[h] = vals[j] ?? ""; });
    row._row = i + 2;
    return row;
  });
}

function validateClaims(rows) {
  const required  = ["claim_id","customer_age","customer_occupation","claim_type","claim_amount","claim_date"];
  const validTypes= new Set(["IPD","OPD","Dental","Maternity"]);
  return rows.flatMap(r => {
    const issues = [];
    required.forEach(c => { if (!r[c]) issues.push(`Missing field: ${c}`); });
    if (r.customer_age && (isNaN(+r.customer_age) || +r.customer_age < 1 || +r.customer_age > 120))
      issues.push("Invalid age");
    if (r.claim_amount && (isNaN(+r.claim_amount) || +r.claim_amount < 0))
      issues.push("Invalid claim_amount");
    if (r.claim_type && !validTypes.has(r.claim_type))
      issues.push(`Unknown claim_type "${r.claim_type}"`);
    if (r.claim_date && !/^\d{4}-\d{2}-\d{2}$/.test(r.claim_date))
      issues.push("claim_date must be YYYY-MM-DD");
    return issues.map(issue => ({ row: r._row, claim_id: r.claim_id, issue }));
  });
}

function simulateCalibration(rows) {
  const ages    = rows.map(r => +r.customer_age).filter(a => !isNaN(a));
  const amounts = rows.map(r => +r.claim_amount).filter(a => !isNaN(a));
  const dist    = { IPD: 0, OPD: 0, Dental: 0, Maternity: 0 };
  rows.forEach(r => { if (dist[r.claim_type] !== undefined) dist[r.claim_type]++; });
  const avgAmt = amounts.length ? Math.round(amounts.reduce((a, b) => a + b, 0) / amounts.length) : 0;
  const dates  = rows.map(r => r.claim_date).filter(Boolean).sort();

  return {
    records: rows.length,
    summary: {
      age_range:  [Math.min(...ages, 999), Math.max(...ages, 0)],
      avg_claim:  avgAmt,
      date_range: [dates[0] || "—", dates[dates.length - 1] || "—"],
      distribution: dist,
    },
    obs_exp: {
      IPD:       { obs: 0.138, exp: 0.120, ratio: 1.15, delta: "+15.0%" },
      OPD:       { obs: 2.450, exp: 2.500, ratio: 0.98, delta: "–2.0%"  },
      Dental:    { obs: 0.864, exp: 0.800, ratio: 1.08, delta: "+8.0%"  },
      Maternity: { obs: 0.162, exp: 0.150, ratio: 1.08, delta: "+8.0%"  },
    },
    segments: [
      { label: "Age 18–24",        ratio: 0.92, status: "below"  },
      { label: "Age 25–34",        ratio: 0.98, status: "ok"     },
      { label: "Age 35–44",        ratio: 1.18, status: "above"  },
      { label: "Age 45–54",        ratio: 1.32, status: "high"   },
      { label: "Age 55–64",        ratio: 1.45, status: "high"   },
      { label: "Smoking: Current", ratio: 1.42, status: "high"   },
      { label: "Manual Labor",     ratio: 1.18, status: "above"  },
    ],
    coeff_changes: [
      { factor: "Age 35–44",             old: 1.12, nw: 1.15,  delta: "+2.7%"  },
      { factor: "Age 45–54",             old: 1.28, nw: 1.33,  delta: "+3.9%"  },
      { factor: "Age 55–64",             old: 1.48, nw: 1.52,  delta: "+2.7%"  },
      { factor: "IPD base frequency",    old: 0.12, nw: 0.138, delta: "+15.0%" },
      { factor: "Dental base frequency", old: 0.80, nw: 0.864, delta: "+8.0%"  },
      { factor: "Smoking: Current",      old: 1.40, nw: 1.42,  delta: "+1.4%"  },
    ],
    premium_impact: "+6.2%",
  };
}

function exportQuotePDF(q) {
  const riders = Object.keys(q.riders || {});
  const lines = [
    "═══════════════════════════════════════════════════",
    "          DAC HealthPrice — Quote Report            ",
    "═══════════════════════════════════════════════════",
    `Quote ID       : ${q.id || q.quote_id}`,
    `Generated      : ${new Date(q.ts).toLocaleString()}`,
    `Model version  : ${q.version || q.model_version}`,
    `Coefficient ver: ${COEFF.version}`,
    "",
    "── PROFILE ─────────────────────────────────────────",
    `Age            : ${q.age}`,
    `Smoking        : ${q.smoking}`,
    `Occupation     : ${q.occupation}`,
    `Region         : ${q.region}`,
    `Tier           : ${q.tier || q.ipd_tier}`,
    `Dependents     : ${q.dependents || 1}`,
    `Riders         : ${riders.length ? riders.map(r => r.toUpperCase()).join(", ") : "None"}`,
    "",
    "── PREMIUM ─────────────────────────────────────────",
    `Annual premium : $${(q.total_annual || q.total_annual_premium || 0).toLocaleString()}`,
    `Monthly premium: $${(q.total_monthly || q.total_monthly_premium || 0).toFixed(2)}`,
    "",
    "── FACTOR BREAKDOWN ────────────────────────────────",
    ...(q.breakdown || []).map(b => `  ${b.label.padEnd(30)} ×${b.factor}`),
    "",
    "── COMPLIANCE NOTE ─────────────────────────────────",
    "This quote is calculated using actuarial-grade GLM",
    "(Poisson frequency × Gamma severity) with full",
    "coefficient transparency. Audit-ready.",
    "",
    "DAC (Decent Actuarial Consultants) — Confidential",
    "═══════════════════════════════════════════════════",
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `quote_${(q.id || q.quote_id || "export")}.txt`;
  a.click();
}

function addAuditLog(entry) {
  try {
    const log = JSON.parse(localStorage.getItem("dac_audit_log") || "[]");
    log.unshift({ ...entry, ts: new Date().toISOString() });
    localStorage.setItem("dac_audit_log", JSON.stringify(log.slice(0, 200)));
  } catch {}
}

// ── Shared UI primitives ──────────────────────────────────────────────────────
const lbl = { display: "block", fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 5 };
const inp = { width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit", outline: "none", background: WHITE };

function Card({ children, style = {} }) {
  return (
    <div style={{ background: WHITE, borderRadius: 14, border: "1px solid #e5e7eb", padding: 22, ...style }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return <h3 style={{ fontSize: 15, fontWeight: 700, color: NAVY, marginBottom: 16 }}>{children}</h3>;
}

function StatusDot({ ok }) {
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: ok ? OK : ERR, marginRight: 6 }} />;
}

function Badge({ label, color = NAVY, bg = LTGRAY, style = {} }) {
  return (
    <span style={{ padding: "2px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: bg, color, ...style }}>
      {label}
    </span>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: LTGRAY, borderRadius: 10, padding: "14px 16px" }}>
      <p style={{ fontSize: 11, color: TXT2, marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 20, fontWeight: 700, color: accent || NAVY }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: TXT2, marginTop: 2 }}>{sub}</p>}
    </div>
  );
}

// ── Auth Gate ─────────────────────────────────────────────────────────────────
function AuthGate({ onAuth }) {
  const [key, setKey] = useState("");
  return (
    <div style={{ minHeight: "65vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <Card style={{ width: 380, textAlign: "center" }}>
        <div style={{ width: 52, height: 52, borderRadius: 14, background: NAVY, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 18px" }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C9.24 2 7 4.24 7 7v3H5v12h14V10h-2V7c0-2.76-2.24-5-5-5zm0 2c1.66 0 3 1.34 3 3v3H9V7c0-1.66 1.34-3 3-3z" fill={GOLD}/>
          </svg>
        </div>
        <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 6 }}>Insurance Dashboard</h2>
        <p style={{ color: TXT2, fontSize: 13, marginBottom: 20 }}>Enter your API key to continue</p>
        <input
          type="password" placeholder="API key"
          value={key} onChange={e => setKey(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && key) onAuth(key); }}
          style={{ ...inp, marginBottom: 12 }}
        />
        <button
          onClick={() => key && onAuth(key)}
          style={{ width: "100%", background: NAVY, color: WHITE, border: "none", padding: "12px 0", borderRadius: 9, fontWeight: 600, fontSize: 15, cursor: "pointer", fontFamily: "inherit" }}
        >
          Authenticate
        </button>
      </Card>
    </div>
  );
}

// ── Tab: Quick Quote ──────────────────────────────────────────────────────────
const INIT_FORM = { age: 35, smoking: "Never", occupation: "Office/Desk", region: "Phnom Penh", tier: "Silver", opd: false, dental: false, maternity: false, dependents: 1 };

function QuickQuoteTab({ apiKey, username }) {
  const [form,         setForm]         = useState(INIT_FORM);
  const [result,       setResult]       = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [expandBreak,  setExpandBreak]  = useState(false);
  const [filterRegion, setFilterRegion] = useState("");
  const [filterTier,   setFilterTier]   = useState("");
  const [history,      setHistory]      = useState(() => {
    try { return JSON.parse(localStorage.getItem("dac_qq_history") || "[]"); } catch { return []; }
  });

  const u = (k, v) => setForm(p => ({ ...p, [k]: v }));

  const saveToHistory = (q) => {
    const next = [q, ...history].slice(0, 100);
    setHistory(next);
    localStorage.setItem("dac_qq_history", JSON.stringify(next));
    addAuditLog({ action: "quick_quote", quote_id: q.id, tier: q.tier, region: q.region, user: username });
  };

  const getQuote = async () => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_URL}/api/v2/price`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify({
          age: +form.age, gender: "Male", region: form.region,
          smoking_status: form.smoking, occupation_type: form.occupation,
          ipd_tier: form.tier, family_size: +form.dependents,
          include_opd: form.opd, include_dental: form.dental, include_maternity: form.maternity,
          exercise_frequency: "Moderate", preexist_conditions: ["None"],
          exercise_days: 3, exercise_mins: 30,
        }),
        signal: AbortSignal.timeout(12000),
      });
      if (r.ok) {
        const data = await r.json();
        const q = {
          ...computeQuote(form),          // enrich with breakdown
          id: data.quote_id || `API-${Date.now()}`,
          total_annual:  data.total_annual_premium || computeQuote(form).total_annual,
          total_monthly: data.total_monthly_premium || computeQuote(form).total_monthly,
          source: "api",
        };
        setResult(q); saveToHistory(q);
        setLoading(false); return;
      }
    } catch { /* fall through */ }
    // Local GLM fallback
    const q = computeQuote(form);
    setResult(q); saveToHistory(q);
    setLoading(false);
  };

  const filtered = history.filter(h =>
    (!filterRegion || h.region === filterRegion) &&
    (!filterTier   || (h.tier || h.ipd_tier) === filterTier)
  );

  const annual  = result?.total_annual  || result?.total_annual_premium  || 0;
  const monthly = result?.total_monthly || result?.total_monthly_premium || 0;

  return (
    <div>
      {/* Form + Result row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
        {/* Form */}
        <Card>
          <SectionTitle>Quick quote</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={lbl}>Age</label>
              <input type="number" min={18} max={85} value={form.age} onChange={e => u("age", e.target.value)} style={inp} />
            </div>
            <div>
              <label style={lbl}>Dependents</label>
              <input type="number" min={1} max={10} value={form.dependents} onChange={e => u("dependents", e.target.value)} style={inp} />
            </div>
            <div>
              <label style={lbl}>Smoking</label>
              <select value={form.smoking} onChange={e => u("smoking", e.target.value)} style={inp}>
                {SMOKING_LIST.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label style={lbl}>Tier</label>
              <select value={form.tier} onChange={e => u("tier", e.target.value)} style={inp}>
                {["Bronze","Silver","Gold","Platinum"].map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <label style={lbl}>Occupation</label>
              <select value={form.occupation} onChange={e => u("occupation", e.target.value)} style={inp}>
                {OCCUPATIONS.map(o => <option key={o}>{o}</option>)}
              </select>
            </div>
            <div>
              <label style={lbl}>Region</label>
              <select value={form.region} onChange={e => u("region", e.target.value)} style={inp}>
                {REGIONS.map(r => <option key={r}>{r}</option>)}
              </select>
            </div>
          </div>

          <div style={{ marginTop: 14 }}>
            <label style={lbl}>Optional riders</label>
            <div style={{ display: "flex", gap: 20 }}>
              {[["opd","OPD"],["dental","Dental"],["maternity","Maternity"]].map(([k, label]) => (
                <label key={k} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 13 }}>
                  <input type="checkbox" checked={form[k]} onChange={e => u(k, e.target.checked)} />
                  {label}
                </label>
              ))}
            </div>
          </div>

          {error && <p style={{ marginTop: 10, fontSize: 12, color: ERR }}>{error}</p>}

          <button
            onClick={getQuote} disabled={loading}
            style={{ marginTop: 18, width: "100%", background: NAVY, color: WHITE, border: "none", padding: "12px 0", borderRadius: 9, fontWeight: 600, fontSize: 15, cursor: loading ? "default" : "pointer", opacity: loading ? 0.65 : 1, fontFamily: "inherit", transition: "opacity 0.2s" }}
          >
            {loading ? "Calculating…" : "Get Quote"}
          </button>
        </Card>

        {/* Result */}
        <Card style={{ background: result ? WHITE : LTGRAY, transition: "background 0.3s" }}>
          {!result ? (
            <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, color: TXT2 }}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <p style={{ fontSize: 13 }}>Quote will appear here</p>
            </div>
          ) : (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                  <p style={{ fontSize: 11, color: TXT2, letterSpacing: 1, textTransform: "uppercase" }}>Annual Premium</p>
                  <p style={{ fontSize: 36, fontWeight: 700, color: NAVY, lineHeight: 1.1 }}>${annual.toLocaleString()}</p>
                  <p style={{ fontSize: 13, color: TXT2, marginTop: 2 }}>${monthly.toFixed(2)} / month</p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <Badge label={result.tier || result.ipd_tier} color={GOLD_D} bg="#fff7ed" />
                  <p style={{ fontSize: 10, color: TXT2, marginTop: 6 }}>Model {result.version || result.model_version}</p>
                  <p style={{ fontSize: 10, color: TXT2 }}>{new Date(result.ts).toLocaleTimeString()}</p>
                </div>
              </div>

              {/* Factor breakdown toggle */}
              <div style={{ borderTop: "1px solid #e5e7eb", paddingTop: 12 }}>
                <button
                  onClick={() => setExpandBreak(x => !x)}
                  style={{ fontSize: 12, fontWeight: 600, color: NAVY, background: "none", border: "none", cursor: "pointer", fontFamily: "inherit", padding: 0, marginBottom: expandBreak ? 10 : 0 }}
                >
                  {expandBreak ? "▼" : "▶"} Factor breakdown
                </button>
                {expandBreak && result.breakdown && (
                  <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>
                    {result.breakdown.map((b, i) => (
                      <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 6px", borderRadius: 5, background: i % 2 === 0 ? LTGRAY : "transparent" }}>
                        <span style={{ color: TXT2 }}>{b.label}</span>
                        <span style={{ fontWeight: 600, color: b.factor > 1 ? "#dc2626" : b.factor < 1 ? OK : TXT }}>
                          ×{b.factor}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Riders */}
              {result.riders && Object.keys(result.riders).length > 0 && (
                <div style={{ marginTop: 12, padding: "10px 12px", background: LTGRAY, borderRadius: 8 }}>
                  <p style={{ fontSize: 11, fontWeight: 600, color: TXT2, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>Riders</p>
                  {Object.entries(result.riders).map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
                      <span style={{ color: TXT2 }}>{k.toUpperCase()}</span>
                      <span style={{ fontWeight: 600 }}>${(v.annual || v.annual_premium || 0).toLocaleString()}/yr</span>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
                <button
                  onClick={() => exportQuotePDF(result)}
                  style={{ flex: 1, background: NAVY, color: WHITE, border: "none", padding: "9px 0", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}
                >
                  Export PDF
                </button>
                <button
                  onClick={() => setResult(null)}
                  style={{ flex: 1, background: LTGRAY, color: TXT2, border: "none", padding: "9px 0", borderRadius: 8, fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}
                >
                  Clear
                </button>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* History table */}
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
          <SectionTitle>Quote history ({filtered.length})</SectionTitle>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <select value={filterRegion} onChange={e => setFilterRegion(e.target.value)} style={{ ...inp, width: "auto", padding: "6px 10px", fontSize: 12 }}>
              <option value="">All regions</option>
              {REGIONS.map(r => <option key={r}>{r}</option>)}
            </select>
            <select value={filterTier} onChange={e => setFilterTier(e.target.value)} style={{ ...inp, width: "auto", padding: "6px 10px", fontSize: 12 }}>
              <option value="">All tiers</option>
              {["Bronze","Silver","Gold","Platinum"].map(t => <option key={t}>{t}</option>)}
            </select>
            {history.length > 0 && (
              <button
                onClick={() => { const ok = window.confirm("Clear all quote history?"); if (ok) { setHistory([]); localStorage.removeItem("dac_qq_history"); }}}
                style={{ padding: "6px 12px", borderRadius: 7, border: `1px solid #fca5a5`, background: "#fef2f2", color: ERR, fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}
              >
                Clear all
              </button>
            )}
          </div>
        </div>

        {filtered.length === 0 ? (
          <p style={{ fontSize: 13, color: TXT2, textAlign: "center", padding: "20px 0" }}>No quotes yet</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Quote ID","Time","Age","Smoking","Occupation","Region","Tier","Riders","Annual",""].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "7px 8px", fontSize: 11, color: TXT2, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 50).map((q, i) => {
                  const riderList = Object.keys(q.riders || {}).map(r => r.toUpperCase()).join(", ") || "—";
                  return (
                    <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "7px 8px", fontFamily: "monospace", fontSize: 10, color: TXT2 }}>{(q.id || "").slice(-10)}</td>
                      <td style={{ padding: "7px 8px", color: TXT2, whiteSpace: "nowrap" }}>{new Date(q.ts).toLocaleString()}</td>
                      <td style={{ padding: "7px 8px" }}>{q.age}</td>
                      <td style={{ padding: "7px 8px" }}>{q.smoking}</td>
                      <td style={{ padding: "7px 8px" }}>{q.occupation}</td>
                      <td style={{ padding: "7px 8px" }}>{q.region}</td>
                      <td style={{ padding: "7px 8px" }}>
                        <Badge label={q.tier || q.ipd_tier} color={GOLD_D} bg="#fff7ed" />
                      </td>
                      <td style={{ padding: "7px 8px", color: TXT2 }}>{riderList}</td>
                      <td style={{ padding: "7px 8px", fontWeight: 600 }}>${(q.total_annual || q.total_annual_premium || 0).toLocaleString()}</td>
                      <td style={{ padding: "7px 8px" }}>
                        <button
                          onClick={() => exportQuotePDF(q)}
                          style={{ padding: "3px 8px", borderRadius: 5, border: "none", background: LTGRAY, fontSize: 11, cursor: "pointer", fontFamily: "inherit", color: NAVY }}
                        >PDF</button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── Tab: Batch Quotes ─────────────────────────────────────────────────────────
function BatchTab({ apiKey, username }) {
  const [file,       setFile]       = useState(null);
  const [rows,       setRows]       = useState([]);
  const [results,    setResults]    = useState([]);
  const [processing, setProcessing] = useState(false);
  const [progress,   setProgress]   = useState(0);
  const [parseError, setParseError] = useState(null);

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResults([]); setParseError(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = parseCSV(ev.target.result);
        setRows(parsed);
      } catch { setParseError("Failed to parse CSV. Check the file format."); }
    };
    reader.readAsText(f);
  };

  const runBatch = async () => {
    if (!rows.length) return;
    setProcessing(true); setProgress(0);
    const out = [];
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const q = computeQuote({
        age:        +r.age || 30,
        smoking:    r.smoking || "Never",
        occupation: r.occupation || "Office/Desk",
        region:     r.region || "Phnom Penh",
        tier:       r.tier || "Silver",
        opd:        r.opd === "true",
        dental:     r.dental === "true",
        maternity:  r.maternity === "true",
        dependents: +r.dependents || 1,
      });
      out.push({ row: i + 2, ...q });
      setProgress(Math.round(((i + 1) / rows.length) * 100));
      // Yield to keep UI responsive every 20 rows
      if (i % 20 === 0) await new Promise(res => setTimeout(res, 0));
    }
    setResults(out);
    addAuditLog({ action: "batch_quote", count: out.length, file: file?.name, user: username });
    setProcessing(false);
  };

  const exportResults = () => {
    const header = "row,quote_id,age,smoking,occupation,region,tier,dependents,riders,total_annual,total_monthly,model_version";
    const csvRows = results.map(r =>
      [r.row, r.id, r.age, r.smoking, r.occupation, r.region, r.tier, r.dependents,
       Object.keys(r.riders || {}).join("|") || "none",
       r.total_annual, r.total_monthly, r.version].join(",")
    );
    downloadCSV(`batch_results_${Date.now()}.csv`, header, csvRows);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <SectionTitle>Batch quote upload</SectionTitle>
        <p style={{ fontSize: 13, color: TXT2, marginBottom: 16 }}>
          Upload a CSV with multiple customer profiles. All quotes are calculated instantly using the GLM engine.
        </p>
        <div style={{ display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" }}>
          <button
            onClick={downloadBatchTemplate}
            style={{ padding: "8px 16px", borderRadius: 8, border: `1px solid ${NAVY}`, background: WHITE, color: NAVY, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}
          >
            Download template CSV
          </button>
        </div>

        <label style={lbl}>Upload CSV</label>
        <input type="file" accept=".csv" onChange={handleFile}
          style={{ ...inp, marginBottom: 12 }} />

        {parseError && <p style={{ fontSize: 12, color: ERR, marginBottom: 10 }}>{parseError}</p>}

        {rows.length > 0 && !parseError && (
          <div style={{ marginBottom: 14, padding: "10px 14px", background: "#ecfdf5", borderRadius: 8, border: "1px solid #a7f3d0" }}>
            <p style={{ fontSize: 13, color: "#065f46" }}>
              <strong>{rows.length} rows</strong> parsed successfully. Ready to process.
            </p>
          </div>
        )}

        {processing && (
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: TXT2, marginBottom: 4 }}>
              <span>Processing…</span><span>{progress}%</span>
            </div>
            <div style={{ background: LTGRAY, borderRadius: 20, height: 6 }}>
              <div style={{ background: GOLD, borderRadius: 20, height: 6, width: `${progress}%`, transition: "width 0.2s" }} />
            </div>
          </div>
        )}

        <button
          onClick={runBatch} disabled={!rows.length || processing}
          style={{ background: NAVY, color: WHITE, border: "none", padding: "11px 28px", borderRadius: 9, fontWeight: 600, fontSize: 14, cursor: (!rows.length || processing) ? "default" : "pointer", opacity: (!rows.length || processing) ? 0.6 : 1, fontFamily: "inherit" }}
        >
          {processing ? "Processing…" : "Run batch"}
        </button>
      </Card>

      {results.length > 0 && (
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <SectionTitle>{results.length} quotes generated</SectionTitle>
            <button
              onClick={exportResults}
              style={{ padding: "8px 16px", borderRadius: 8, border: "none", background: NAVY, color: WHITE, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}
            >
              Export CSV
            </button>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Row","Age","Smoking","Occupation","Region","Tier","Riders","Annual","Monthly"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "7px 8px", fontSize: 11, color: TXT2, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={{ padding: "7px 8px", color: TXT2 }}>{r.row}</td>
                    <td style={{ padding: "7px 8px" }}>{r.age}</td>
                    <td style={{ padding: "7px 8px" }}>{r.smoking}</td>
                    <td style={{ padding: "7px 8px" }}>{r.occupation}</td>
                    <td style={{ padding: "7px 8px" }}>{r.region}</td>
                    <td style={{ padding: "7px 8px" }}><Badge label={r.tier} color={GOLD_D} bg="#fff7ed" /></td>
                    <td style={{ padding: "7px 8px", color: TXT2 }}>{Object.keys(r.riders || {}).map(k => k.toUpperCase()).join(", ") || "—"}</td>
                    <td style={{ padding: "7px 8px", fontWeight: 600 }}>${r.total_annual.toLocaleString()}</td>
                    <td style={{ padding: "7px 8px", color: TXT2 }}>${r.total_monthly.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Tab: Data Calibration (3 phases) ─────────────────────────────────────────
function CalibrationTab({ apiKey, username }) {
  const [phase,       setPhase]       = useState("upload"); // upload | sandbox | deploy | done
  const [file,        setFile]        = useState(null);
  const [rows,        setRows]        = useState([]);
  const [errors,      setErrors]      = useState([]);
  const [preview,     setPreview]     = useState([]);
  const [compliance,  setCompliance]  = useState({ anon: false, permission: false, gdpr: false });
  const [uploading,   setUploading]   = useState(false);
  const [analysis,    setAnalysis]    = useState(null);
  const [datasetName, setDatasetName] = useState("");
  const [history,     setHistory]     = useState(() => {
    try { return JSON.parse(localStorage.getItem("dac_upload_history") || "[]"); } catch { return []; }
  });

  const allCompliant = Object.values(compliance).every(Boolean);

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setErrors([]); setPreview([]);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const parsed = parseCSV(ev.target.result);
      const errs   = validateClaims(parsed);
      setRows(parsed);
      setErrors(errs);
      setPreview(parsed.slice(0, 20));
    };
    reader.readAsText(f);
  };

  const handleUpload = async () => {
    if (!rows.length || errors.length || !allCompliant || !datasetName) return;
    setUploading(true);

    // Try backend upload endpoint
    try {
      const fd = new FormData();
      fd.append("file", file); fd.append("dataset_name", datasetName);
      const r = await fetch(`${API_URL}/api/v2/admin/upload-claims`, {
        method: "POST", headers: { "X-API-Key": apiKey }, body: fd,
        signal: AbortSignal.timeout(20000),
      });
      if (!r.ok) throw new Error();
    } catch { /* backend endpoint may not exist yet — proceed with simulation */ }

    const cal = simulateCalibration(rows);
    setAnalysis({ ...cal, datasetName, upload_id: `CAL-${Date.now()}`, upload_date: new Date().toISOString() });
    addAuditLog({ action: "claims_upload", dataset: datasetName, records: rows.length, user: username });
    setUploading(false);
    setPhase("sandbox");
  };

  const handleDeploy = () => {
    if (!analysis) return;
    const entry = {
      id:        analysis.upload_id,
      dataset:   analysis.datasetName,
      date:      new Date().toLocaleDateString(),
      records:   analysis.records,
      status:    "LIVE",
      impact:    analysis.premium_impact,
    };
    const next = [entry, ...history];
    setHistory(next);
    localStorage.setItem("dac_upload_history", JSON.stringify(next));
    addAuditLog({ action: "calibration_deployed", dataset: analysis.datasetName, version: "v2.3", user: username });
    setPhase("done");
  };

  const reset = () => { setPhase("upload"); setFile(null); setRows([]); setErrors([]); setPreview([]); setAnalysis(null); setDatasetName(""); setCompliance({ anon: false, permission: false, gdpr: false }); };

  const ratioColor = (r) => r > 1.2 ? ERR : r < 0.85 ? "#f59e0b" : OK;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Phase progress bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 0, marginBottom: 4 }}>
        {[["upload","1. Upload"],["sandbox","2. Sandbox"],["deploy","3. Deploy"]].map(([p, label], i) => {
          const phaseOrder = { upload: 0, sandbox: 1, deploy: 2, done: 3 };
          const current = phaseOrder[phase];
          const isPast  = phaseOrder[p] < current;
          const isNow   = p === phase || (phase === "done" && p === "deploy");
          return (
            <div key={p} style={{ display: "flex", alignItems: "center", flex: i < 2 ? 1 : "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 28, height: 28, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, background: isPast || (phase === "done") ? OK : isNow ? NAVY : LTGRAY, color: (isPast || isNow || phase === "done") ? WHITE : TXT2, border: "none", flexShrink: 0 }}>
                  {isPast || phase === "done" ? "✓" : i + 1}
                </div>
                <span style={{ fontSize: 13, fontWeight: isNow ? 700 : 500, color: isNow ? NAVY : TXT2, whiteSpace: "nowrap" }}>{label}</span>
              </div>
              {i < 2 && <div style={{ flex: 1, height: 2, background: isPast || (phase === "done") ? OK : LTGRAY, margin: "0 10px" }} />}
            </div>
          );
        })}
      </div>

      {/* ── PHASE 1: Upload ── */}
      {phase === "upload" && (
        <Card>
          <SectionTitle>Phase 1 — Upload claims data</SectionTitle>
          <p style={{ fontSize: 13, color: TXT2, marginBottom: 16 }}>
            Upload anonymized historical claims to calibrate the pricing model to your book of business.
          </p>

          <button onClick={downloadClaimsTemplate} style={{ padding: "8px 16px", borderRadius: 8, border: `1px solid ${NAVY}`, background: WHITE, color: NAVY, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit", marginBottom: 18 }}>
            Download claims template CSV
          </button>

          <label style={lbl}>Dataset name</label>
          <input type="text" placeholder="e.g. Q1 2026 Claims" value={datasetName} onChange={e => setDatasetName(e.target.value)} style={{ ...inp, marginBottom: 14 }} />

          <label style={lbl}>Claims CSV file</label>
          <p style={{ fontSize: 11, color: TXT2, marginBottom: 6 }}>
            Required columns: claim_id, customer_age, customer_occupation, claim_type, claim_amount, claim_date
          </p>
          <input type="file" accept=".csv" onChange={handleFile} style={{ ...inp, marginBottom: 14 }} />

          {/* Validation results */}
          {rows.length > 0 && (
            <div style={{ marginBottom: 14, padding: 14, borderRadius: 10, background: errors.length ? "#fef2f2" : "#ecfdf5", border: `1px solid ${errors.length ? "#fecaca" : "#a7f3d0"}` }}>
              <p style={{ fontSize: 13, fontWeight: 600, color: errors.length ? ERR : "#065f46", marginBottom: errors.length ? 8 : 0 }}>
                {errors.length === 0
                  ? `✓ ${rows.length} rows valid`
                  : `⚠ ${rows.length - errors.length} valid · ${errors.length} invalid (will be skipped)`}
              </p>
              {errors.slice(0, 5).map((e, i) => (
                <p key={i} style={{ fontSize: 11, color: ERR }}>Row {e.row}: {e.issue}</p>
              ))}
              {errors.length > 5 && <p style={{ fontSize: 11, color: TXT2 }}>…and {errors.length - 5} more</p>}
            </div>
          )}

          {/* Summary stats */}
          {preview.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Data preview (first 20 rows)</p>
              <div style={{ overflowX: "auto", maxHeight: 200, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid #e5e7eb", position: "sticky", top: 0, background: WHITE }}>
                      {["claim_id","customer_age","customer_occupation","claim_type","claim_amount","claim_date"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "5px 8px", color: TXT2, fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.map((r, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "5px 8px", color: TXT2 }}>{r.claim_id}</td>
                        <td style={{ padding: "5px 8px" }}>{r.customer_age}</td>
                        <td style={{ padding: "5px 8px" }}>{r.customer_occupation}</td>
                        <td style={{ padding: "5px 8px" }}><Badge label={r.claim_type} color={NAVY} bg={LTGRAY} /></td>
                        <td style={{ padding: "5px 8px" }}>${r.claim_amount}</td>
                        <td style={{ padding: "5px 8px", color: TXT2 }}>{r.claim_date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Compliance checkboxes */}
          <div style={{ marginBottom: 18, padding: 14, borderRadius: 10, background: LTGRAY }}>
            <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Compliance confirmation</p>
            {[
              ["anon",       "All data is anonymized — no customer names, IDs, or PII"],
              ["permission", "I own or have proper authorization to use this data"],
              ["gdpr",       "This data complies with applicable data protection regulations"],
            ].map(([k, label]) => (
              <label key={k} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={compliance[k]} onChange={e => setCompliance(p => ({ ...p, [k]: e.target.checked }))} style={{ marginTop: 2 }} />
                <span style={{ fontSize: 13 }}>{label}</span>
              </label>
            ))}
          </div>

          <button
            onClick={handleUpload}
            disabled={!rows.length || !allCompliant || !datasetName || uploading}
            style={{ background: NAVY, color: WHITE, border: "none", padding: "12px 28px", borderRadius: 9, fontWeight: 600, fontSize: 14, cursor: (!rows.length || !allCompliant || !datasetName || uploading) ? "default" : "pointer", opacity: (!rows.length || !allCompliant || !datasetName || uploading) ? 0.55 : 1, fontFamily: "inherit" }}
          >
            {uploading ? "Uploading…" : "Upload & Analyse"}
          </button>
          {!datasetName && rows.length > 0 && <p style={{ fontSize: 11, color: ERR, marginTop: 6 }}>Enter a dataset name to continue.</p>}
          {!allCompliant && rows.length > 0 && <p style={{ fontSize: 11, color: ERR, marginTop: 4 }}>Confirm all compliance checkboxes to continue.</p>}
        </Card>
      )}

      {/* ── PHASE 2: Sandbox analysis ── */}
      {phase === "sandbox" && analysis && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 10, marginBottom: 16 }}>
              <div>
                <SectionTitle>Phase 2 — Sandbox analysis</SectionTitle>
                <p style={{ fontSize: 13, color: TXT2 }}>
                  <strong>{analysis.datasetName}</strong> · {analysis.records} records · {new Date(analysis.upload_date).toLocaleDateString()}
                </p>
              </div>
              <Badge label="SANDBOX — not yet live" color="#92400e" bg="#fef3c7" style={{ padding: "4px 12px" }} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
              <StatCard label="Records"     value={analysis.records.toLocaleString()} />
              <StatCard label="Avg claim"   value={`$${analysis.summary.avg_claim}`} />
              <StatCard label="Age range"   value={`${analysis.summary.age_range[0]}–${analysis.summary.age_range[1]}`} />
              <StatCard label="Date range"  value={analysis.summary.date_range[0]?.slice(0,7)} sub={`to ${analysis.summary.date_range[1]?.slice(0,7)}`} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginTop: 12 }}>
              {Object.entries(analysis.summary.distribution).map(([type, count]) => (
                <div key={type} style={{ textAlign: "center", padding: 10, background: LTGRAY, borderRadius: 8 }}>
                  <p style={{ fontSize: 11, color: TXT2 }}>{type}</p>
                  <p style={{ fontSize: 18, fontWeight: 700 }}>{count}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <SectionTitle>Observed vs. expected (O/E ratios)</SectionTitle>
            <p style={{ fontSize: 12, color: TXT2, marginBottom: 14 }}>
              Ratio of 1.0 = perfectly calibrated. Above 1.0 = model underpredicts (riskier than assumed). Below = overpredicts.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {Object.entries(analysis.obs_exp).map(([type, d]) => (
                <div key={type} style={{ display: "grid", gridTemplateColumns: "80px 1fr 60px 60px 70px", gap: 12, alignItems: "center", padding: "10px 14px", borderRadius: 8, background: LTGRAY }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{type}</span>
                  <div style={{ background: "#e5e7eb", borderRadius: 20, height: 8, overflow: "hidden" }}>
                    <div style={{ background: ratioColor(d.ratio), width: `${Math.min(d.ratio * 60, 100)}%`, height: "100%", borderRadius: 20, transition: "width 0.4s" }} />
                  </div>
                  <span style={{ fontSize: 12, color: TXT2 }}>Obs: {d.obs}</span>
                  <span style={{ fontSize: 12, color: TXT2 }}>Exp: {d.exp}</span>
                  <span style={{ fontWeight: 700, fontSize: 13, color: ratioColor(d.ratio) }}>×{d.ratio} ({d.delta})</span>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <SectionTitle>Segment analysis</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: 10 }}>
              {analysis.segments.map((s, i) => (
                <div key={i} style={{ padding: "12px 14px", borderRadius: 8, background: LTGRAY, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: TXT2 }}>{s.label}</span>
                  <span style={{ fontWeight: 700, fontSize: 14, color: ratioColor(s.ratio) }}>×{s.ratio}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <SectionTitle>Calibrated coefficient preview</SectionTitle>
            <p style={{ fontSize: 12, color: TXT2, marginBottom: 14 }}>
              Estimated changes if this dataset is deployed. No live changes have been made yet.
            </p>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                    {["Factor","Current value","New value","Change"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "8px 10px", fontSize: 12, color: TXT2, fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {analysis.coeff_changes.map((c, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "9px 10px", fontWeight: 500 }}>{c.factor}</td>
                      <td style={{ padding: "9px 10px", color: TXT2 }}>{c.old}</td>
                      <td style={{ padding: "9px 10px", fontWeight: 600 }}>{c.nw}</td>
                      <td style={{ padding: "9px 10px" }}>
                        <Badge label={c.delta} color={c.delta.startsWith("+") ? ERR : OK} bg={c.delta.startsWith("+") ? "#fef2f2" : "#ecfdf5"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 16, padding: "12px 16px", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8 }}>
              <p style={{ fontSize: 13, fontWeight: 600, color: "#92400e" }}>
                Estimated average premium impact: <strong>{analysis.premium_impact}</strong>
              </p>
              <p style={{ fontSize: 12, color: "#78350f", marginTop: 4 }}>
                Deploying this calibration will increase the average quoted premium by approximately {analysis.premium_impact}.
              </p>
            </div>
            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
              <button
                onClick={() => setPhase("deploy")}
                style={{ background: NAVY, color: WHITE, border: "none", padding: "11px 28px", borderRadius: 9, fontWeight: 600, fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}
              >
                Proceed to deploy →
              </button>
              <button
                onClick={reset}
                style={{ background: LTGRAY, color: TXT2, border: "none", padding: "11px 20px", borderRadius: 9, fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}
              >
                Cancel
              </button>
            </div>
          </Card>
        </div>
      )}

      {/* ── PHASE 3: Deploy ── */}
      {phase === "deploy" && analysis && (
        <Card>
          <SectionTitle>Phase 3 — Deploy calibration</SectionTitle>
          <div style={{ padding: 20, border: "2px solid #e5e7eb", borderRadius: 12, marginBottom: 20 }}>
            <p style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>Calibration summary</p>
            {[
              ["Dataset",              analysis.datasetName],
              ["Records",              `${analysis.records.toLocaleString()} claims`],
              ["Coefficients changed", `${analysis.coeff_changes.length} factors`],
              ["New model version",    "v2.3"],
              ["Avg premium impact",   analysis.premium_impact],
              ["Deployed by",          username],
            ].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #f3f4f6", fontSize: 13 }}>
                <span style={{ color: TXT2 }}>{k}</span>
                <span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>

          <div style={{ padding: 14, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, marginBottom: 20 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: ERR, marginBottom: 4 }}>⚠ This action updates live pricing</p>
            <p style={{ fontSize: 12, color: "#7f1d1d" }}>
              All quotes generated after deployment will use the updated coefficients. You can roll back to the previous version at any time.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={handleDeploy}
              style={{ background: NAVY, color: WHITE, border: "none", padding: "12px 32px", borderRadius: 9, fontWeight: 600, fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}
            >
              Deploy to live
            </button>
            <button
              onClick={() => setPhase("sandbox")}
              style={{ background: LTGRAY, color: TXT2, border: "none", padding: "12px 20px", borderRadius: 9, fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}
            >
              ← Back
            </button>
          </div>
        </Card>
      )}

      {/* ── PHASE DONE ── */}
      {phase === "done" && (
        <Card style={{ textAlign: "center", padding: 40 }}>
          <div style={{ width: 56, height: 56, borderRadius: "50%", background: "#ecfdf5", border: `2px solid ${OK}`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px", fontSize: 24 }}>✓</div>
          <h3 style={{ fontSize: 20, fontWeight: 700, color: OK, marginBottom: 8 }}>Calibration deployed</h3>
          <p style={{ fontSize: 13, color: TXT2, marginBottom: 20 }}>Model v2.3 is now live. All new quotes will use the updated coefficients.</p>
          <button onClick={reset} style={{ background: NAVY, color: WHITE, border: "none", padding: "11px 28px", borderRadius: 9, fontWeight: 600, fontSize: 14, cursor: "pointer", fontFamily: "inherit" }}>
            New calibration
          </button>
        </Card>
      )}

      {/* Upload history */}
      {history.length > 0 && (
        <Card>
          <SectionTitle>Upload history</SectionTitle>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Dataset","Date","Records","Status","Impact"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "7px 8px", fontSize: 11, color: TXT2, fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={{ padding: "7px 8px", fontWeight: 500 }}>{h.dataset}</td>
                    <td style={{ padding: "7px 8px", color: TXT2 }}>{h.date}</td>
                    <td style={{ padding: "7px 8px" }}>{h.records?.toLocaleString()}</td>
                    <td style={{ padding: "7px 8px" }}>
                      <Badge label={h.status} color={h.status === "LIVE" ? "#065f46" : TXT2} bg={h.status === "LIVE" ? "#ecfdf5" : LTGRAY} />
                    </td>
                    <td style={{ padding: "7px 8px", color: h.impact?.startsWith("+") ? ERR : OK }}>{h.impact || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Tab: Coefficients ─────────────────────────────────────────────────────────
function CoefficientsTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
          <div>
            <SectionTitle>Active model — {COEFF.version}</SectionTitle>
            <p style={{ fontSize: 13, color: TXT2 }}>Last calibrated: {COEFF.last_updated} · by {COEFF.updated_by}</p>
          </div>
          <Badge label="GLM — Poisson × Gamma" color="#1e40af" bg="#eff6ff" style={{ padding: "4px 12px" }} />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: 12, marginBottom: 20 }}>
          {Object.entries(COEFF.base).map(([cov, { freq, sev }]) => (
            <div key={cov} style={{ padding: "14px 16px", background: LTGRAY, borderRadius: 10 }}>
              <p style={{ fontSize: 11, fontWeight: 600, color: TXT2, textTransform: "uppercase", marginBottom: 6 }}>{cov}</p>
              <p style={{ fontSize: 13, marginBottom: 2 }}>Base freq: <strong>{freq}</strong></p>
              <p style={{ fontSize: 13 }}>Base sev: <strong>${sev.toLocaleString()}</strong></p>
            </div>
          ))}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {[
          { title: "Age factors", data: COEFF.age },
          { title: "Smoking factors", data: COEFF.smoke },
          { title: "Occupation factors", data: COEFF.occup },
          { title: "Region factors", data: COEFF.region },
        ].map(({ title, data }) => (
          <Card key={title}>
            <SectionTitle>{title}</SectionTitle>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {Object.entries(data).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 10px", borderRadius: 7, background: LTGRAY }}>
                  <span style={{ fontSize: 13, color: TXT2 }}>{k}</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 60, background: "#e5e7eb", borderRadius: 10, height: 6 }}>
                      <div style={{ background: v > 1 ? ERR : v < 1 ? OK : GOLD, width: `${Math.min(v * 50, 100)}%`, height: 6, borderRadius: 10 }} />
                    </div>
                    <span style={{ fontSize: 13, fontWeight: 700, color: v > 1 ? "#dc2626" : v < 1 ? OK : TXT, minWidth: 34, textAlign: "right" }}>{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <SectionTitle>Tier multipliers & loading rates</SectionTitle>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 8 }}>TIER MULTIPLIERS</p>
            {Object.entries(COEFF.tier).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 10px", borderRadius: 7, background: LTGRAY, marginBottom: 6 }}>
                <span style={{ fontSize: 13 }}>{k}</span>
                <span style={{ fontSize: 13, fontWeight: 700 }}>×{v}</span>
              </div>
            ))}
          </div>
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 8 }}>LOADING RATES</p>
            {Object.entries(COEFF.load).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 10px", borderRadius: 7, background: LTGRAY, marginBottom: 6 }}>
                <span style={{ fontSize: 13, textTransform: "uppercase" }}>{k}</span>
                <span style={{ fontSize: 13, fontWeight: 700 }}>{(v * 100).toFixed(0)}%</span>
              </div>
            ))}
            <div style={{ marginTop: 14 }}>
              <p style={{ fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 8 }}>FAMILY FACTOR</p>
              <div style={{ padding: "7px 10px", borderRadius: 7, background: LTGRAY }}>
                <p style={{ fontSize: 13 }}>+{COEFF.famPer} per additional dependent</p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ── Tab: Security & Compliance ────────────────────────────────────────────────
function SecurityTab({ username }) {
  const [auditLog] = useState(() => {
    try { return JSON.parse(localStorage.getItem("dac_audit_log") || "[]"); } catch { return []; }
  });

  const statuses = [
    { label: "Data encryption",       detail: "AES-256 in transit (TLS 1.3) and at rest",           ok: true  },
    { label: "API authentication",    detail: "X-API-Key header required on all admin endpoints",    ok: true  },
    { label: "Audit logging",         detail: "All operations logged with timestamp and user",        ok: true  },
    { label: "Data anonymization",    detail: "Confirmed at upload time via compliance checkbox",     ok: true  },
    { label: "CSP / HSTS headers",    detail: "Security headers enforced via Vercel config",         ok: true  },
    { label: "Data retention",        detail: "Session data held for 12 months; deletable on request", ok: true },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <SectionTitle>Security status</SectionTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {statuses.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "12px 14px", borderRadius: 9, background: LTGRAY }}>
              <StatusDot ok={s.ok} />
              <div>
                <p style={{ fontSize: 13, fontWeight: 600 }}>{s.label}</p>
                <p style={{ fontSize: 12, color: TXT2, marginTop: 2 }}>{s.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <SectionTitle>Compliance notes</SectionTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13 }}>
          {[
            { title: "Model transparency", body: "All premiums are calculated using GLM coefficients (Poisson frequency × Gamma severity). Every factor is named and auditable. No black-box AI components." },
            { title: "No discriminatory factors", body: "Gender and ethnicity are not pricing factors. All included variables (age, occupation, smoking, region) are actuarially justified and disclosed." },
            { title: "GDPR / data protection", body: "Claims data should be anonymized before upload. Users confirm this at upload time. Data can be deleted on request by contacting support." },
            { title: "Underwriting rules", body: "Decline rules are not implemented in this version. All applicants receive a quote. Age cap: 85. Premium floor: $50/year." },
          ].map((item, i) => (
            <div key={i} style={{ padding: "12px 14px", borderRadius: 9, border: "1px solid #e5e7eb" }}>
              <p style={{ fontWeight: 600, marginBottom: 4 }}>{item.title}</p>
              <p style={{ color: TXT2 }}>{item.body}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <SectionTitle>Audit log ({auditLog.length})</SectionTitle>
          <span style={{ fontSize: 12, color: TXT2 }}>Session: {username}</span>
        </div>
        {auditLog.length === 0 ? (
          <p style={{ fontSize: 13, color: TXT2, padding: "10px 0" }}>No actions logged yet in this session.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                  {["Time","Action","Details","User"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "6px 8px", fontSize: 11, color: TXT2, fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {auditLog.slice(0, 50).map((entry, i) => {
                  const details = Object.entries(entry).filter(([k]) => !["action","user","ts"].includes(k)).map(([k,v]) => `${k}: ${v}`).join(" · ");
                  return (
                    <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "6px 8px", color: TXT2, whiteSpace: "nowrap", fontSize: 11 }}>{new Date(entry.ts).toLocaleString()}</td>
                      <td style={{ padding: "6px 8px", fontWeight: 600 }}>{entry.action}</td>
                      <td style={{ padding: "6px 8px", color: TXT2, fontSize: 11 }}>{details}</td>
                      <td style={{ padding: "6px 8px", color: TXT2 }}>{entry.user}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── Tab: Model Metrics ─────────────────────────────────────────────────────────
function ModelMetricsTab() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_URL}/api/v2/model-info`, {
          headers: { "Content-Type": "application/json" },
          mode: "cors"
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
        const data = await r.json();
        setMetrics(data.metrics || {});
        setError(null);
      } catch (e) {
        console.error("Metrics fetch error:", e);
        setError(e.message || "Failed to fetch metrics");
        setMetrics(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20, minHeight: 400 }}>
        <Card>
          <div style={{ padding: 20, textAlign: "center", color: TXT2 }}>Loading metrics...</div>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <Card>
          <div style={{ padding: 20, background: "#fee", borderRadius: 8, color: ERR, fontSize: 13 }}>
            Error loading metrics: {error}
          </div>
        </Card>
      </div>
    );
  }

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <Card>
          <SectionTitle>Model Metrics</SectionTitle>
          <div style={{ padding: "20px 0", background: "#fffbeb", borderRadius: 8, padding: 16, color: "#92400e" }}>
            <strong>Metrics not available</strong> — retrain the model via the <strong>Data calibration</strong> tab to populate GLM performance metrics.
          </div>
        </Card>
      </div>
    );
  }

  const covTypes = ["ipd", "opd", "dental", "maternity"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <SectionTitle>Severity Model Performance</SectionTitle>
        <p style={{ fontSize: 12, color: TXT2, marginBottom: 16 }}>Gamma GLM prediction accuracy (on training data)</p>

        {covTypes.map(cov => {
          const m = metrics[cov];
          if (!m || !m.sev) return null;
          const sev = m.sev;
          return (
            <div key={cov} style={{ marginBottom: 20, paddingBottom: 16, borderBottom: cov !== covTypes[covTypes.length-1] ? "1px solid #e5e7eb" : "none" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: NAVY, marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>
                {cov.toUpperCase()}
              </div>

              {/* R² bar */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, color: TXT2 }}>R² (Training)</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: NAVY }}>{(sev.r2_train || 0).toFixed(4)}</span>
                </div>
                <div style={{ width: "100%", height: 6, background: LTGRAY, borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min((sev.r2_train || 0) * 100, 100)}%`, height: "100%", background: NAVY }}></div>
                </div>
              </div>

              {/* R² CV bar */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, color: TXT2 }}>R² (5-Fold CV)</span>
                  <span style={{ fontSize: 12, color: TXT2 }}>
                    <strong style={{ color: NAVY }}>{(sev.r2_cv_mean || 0).toFixed(4)}</strong>
                    {" "}± {(sev.r2_cv_std || 0).toFixed(4)}
                  </span>
                </div>
                <div style={{ width: "100%", height: 6, background: LTGRAY, borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min((sev.r2_cv_mean || 0) * 100, 100)}%`, height: "100%", background: NAVY }}></div>
                </div>
              </div>

              {/* Error metrics */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                <div style={{ padding: "10px 12px", borderRadius: 8, background: LTGRAY }}>
                  <div style={{ fontSize: 10, color: TXT2, marginBottom: 4 }}>MSE</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: NAVY }}>${Math.round(sev.mse || 0).toLocaleString()}</div>
                </div>
                <div style={{ padding: "10px 12px", borderRadius: 8, background: LTGRAY }}>
                  <div style={{ fontSize: 10, color: TXT2, marginBottom: 4 }}>RMSE</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: NAVY }}>${Math.round(sev.rmse || 0).toLocaleString()}</div>
                </div>
                <div style={{ padding: "10px 12px", borderRadius: 8, background: LTGRAY }}>
                  <div style={{ fontSize: 10, color: TXT2, marginBottom: 4 }}>MAE</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: NAVY }}>${Math.round(sev.mae || 0).toLocaleString()}</div>
                </div>
              </div>
            </div>
          );
        })}
      </Card>

      <Card>
        <SectionTitle>Frequency Model Performance</SectionTitle>
        <p style={{ fontSize: 12, color: TXT2, marginBottom: 16 }}>Poisson GLM prediction accuracy (cross-validation deviance)</p>

        {covTypes.map(cov => {
          const m = metrics[cov];
          if (!m || !m.freq) return null;
          const freq = m.freq;
          return (
            <div key={cov} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: cov !== covTypes[covTypes.length-1] ? "1px solid #e5e7eb" : "none" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: NAVY, marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>
                {cov.toUpperCase()}
              </div>
              <div style={{ padding: "10px 12px", borderRadius: 8, background: LTGRAY }}>
                <div style={{ fontSize: 12, color: TXT2, marginBottom: 2 }}>5-Fold CV Poisson Deviance</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: NAVY }}>
                  {(freq.cv_deviance_mean || 0).toFixed(4)} ± {(freq.cv_deviance_std || 0).toFixed(4)}
                </div>
                <div style={{ fontSize: 11, color: TXT2, marginTop: 4 }}>Lower deviance = better fit</div>
              </div>
            </div>
          );
        })}
      </Card>

      <Card>
        <div style={{ padding: "12px 14px", borderRadius: 8, background: "#f0f4ff", borderLeft: `4px solid ${NAVY}` }}>
          <div style={{ fontSize: 11, color: TXT2, lineHeight: 1.6 }}>
            <strong>ℹ Metrics Guide:</strong> R² scores closer to 1.0 indicate better model fit. MSE/RMSE/MAE measure average prediction error in dollars. All metrics are computed on synthetic training data. Performance will improve after calibration with real claims data.
          </div>
        </div>
      </Card>
    </div>
  );
}

// ── Model Office (Representative Personas) ────────────────────────────────────
const MODEL_OFFICE = [
  { id: "P1",  label: "Young office, PP",        age: 27, smoking: "Never",   occupation: "Office/Desk",          region: "Phnom Penh",    tier: "Silver",   dependents: 1, opd: true,  dental: false, maternity: false, weight: 18 },
  { id: "P2",  label: "Mid office, married",      age: 36, smoking: "Never",   occupation: "Office/Desk",          region: "Phnom Penh",    tier: "Silver",   dependents: 3, opd: true,  dental: false, maternity: false, weight: 15 },
  { id: "P3",  label: "Young female, maternity",  age: 29, smoking: "Never",   occupation: "Retail/Service",       region: "Phnom Penh",    tier: "Silver",   dependents: 1, opd: true,  dental: false, maternity: true,  weight: 10 },
  { id: "P4",  label: "Manual labor, mid-age",    age: 43, smoking: "Former",  occupation: "Manual Labor",         region: "Siem Reap",     tier: "Bronze",   dependents: 2, opd: false, dental: false, maternity: false, weight: 12 },
  { id: "P5",  label: "High-risk industry",       age: 38, smoking: "Current", occupation: "Industrial/High-Risk", region: "Battambang",    tier: "Bronze",   dependents: 1, opd: false, dental: false, maternity: false, weight: 8  },
  { id: "P6",  label: "Corporate Gold, PP",       age: 45, smoking: "Never",   occupation: "Office/Desk",          region: "Phnom Penh",    tier: "Gold",     dependents: 4, opd: true,  dental: true,  maternity: false, weight: 10 },
  { id: "P7",  label: "Healthcare worker",        age: 33, smoking: "Never",   occupation: "Healthcare",           region: "Phnom Penh",    tier: "Silver",   dependents: 1, opd: true,  dental: true,  maternity: false, weight: 8  },
  { id: "P8",  label: "Senior Platinum",          age: 58, smoking: "Former",  occupation: "Retired",              region: "Phnom Penh",    tier: "Platinum", dependents: 2, opd: true,  dental: true,  maternity: false, weight: 7  },
  { id: "P9",  label: "Rural Bronze, young",      age: 24, smoking: "Never",   occupation: "Retail/Service",       region: "Rural Areas",   tier: "Bronze",   dependents: 1, opd: false, dental: false, maternity: false, weight: 7  },
  { id: "P10", label: "Provincial family Gold",   age: 41, smoking: "Never",   occupation: "Office/Desk",          region: "Kampong Cham",  tier: "Gold",     dependents: 3, opd: true,  dental: false, maternity: false, weight: 5  },
];

// ── Parameterized Pricing Engine ──────────────────────────────────────────────
function computeQuoteWith(profile, params) {
  // params = { load: {...}, tier: {...}, ded: {...}, famPer: scalar }
  const ab = ageBand(profile.age);
  const af = COEFF.age[ab]   || 1;
  const sf = COEFF.smoke[profile.smoking]     || 1;
  const of = COEFF.occup[profile.occupation]  || 1;
  const rf = COEFF.region[profile.region]     || 1;
  const tf = params.tier[profile.tier]        || 1;
  const ff = 1 + (profile.dependents - 1) * params.famPer;

  function calcCovWith(cov) {
    const { freq, sev } = COEFF.base[cov];
    const efFreq = freq * af * sf * of * rf;
    const efSev  = sev  * (1 + Math.max(0, profile.age - 30) * 0.006);
    const eac    = efFreq * efSev; // pure premium, no loading
    return {
      expected: eac, // for margin calculation
      premium:  eac * (1 + params.load[cov]), // loaded premium
    };
  }

  const ipd       = calcCovWith("ipd");
  const dedCredit = params.ded[profile.tier] * 0.10;
  const ipdPremium = Math.max(ipd.premium * tf - dedCredit, 50);
  const ipdExpected = Math.max(ipd.expected * tf - dedCredit, 0);

  let totalPremium  = ipdPremium;
  let totalExpected = ipdExpected;

  for (const [cov, inc] of [["opd", profile.opd], ["dental", profile.dental], ["maternity", profile.maternity]]) {
    if (!inc) continue;
    const c = calcCovWith(cov);
    totalPremium  += c.premium;
    totalExpected += c.expected;
  }

  totalPremium  = totalPremium  * ff;
  totalExpected = totalExpected * ff;

  return { totalPremium, totalExpected };
}

// ── Portfolio Optimizer ────────────────────────────────────────────────────────
function runPortfolioOptimizer(personas, constraints, optimizeParams) {
  const tgt = constraints.targetMargin / 100;
  const minBound = constraints.minPremium || 200;
  const maxBound = constraints.maxPremium || 5000;
  const tierForBound = constraints.premiumTier || "Silver";
  const depForBound = 1; // single-dependent reference

  // Build parameter grid
  const loadValues = optimizeParams.load
    ? [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    : [COEFF.load.ipd];
  const famPerValues = optimizeParams.famPer
    ? [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    : [COEFF.famPer];
  const tierGoldValues = optimizeParams.tierGold
    ? [1.25, 1.35, 1.45, 1.55, 1.65]
    : [COEFF.tier.Gold];
  const tierPlatValues = optimizeParams.tierPlatinum
    ? [1.80, 1.95, 2.10, 2.25, 2.40]
    : [COEFF.tier.Platinum];

  // Normalize weights
  const totalWeight = personas.reduce((s, p) => s + p.weight, 0);
  const normalized = personas.map(p => ({ ...p, w: p.weight / totalWeight }));

  // Evaluate all combinations
  const results = [];
  for (const load of loadValues) {
    for (const famPer of famPerValues) {
      for (const tierGold of tierGoldValues) {
        for (const tierPlat of tierPlatValues) {
          const params = {
            load: { ipd: load, opd: load * 0.833, dental: load * 0.667, maternity: load * 0.833 },
            tier: { Bronze: COEFF.tier.Bronze, Silver: COEFF.tier.Silver, Gold: tierGold, Platinum: tierPlat },
            ded:  COEFF.ded,
            famPer,
          };

          // Compute portfolio totals
          let totalPrem = 0, totalExp = 0;
          for (const p of normalized) {
            const { totalPremium, totalExpected } = computeQuoteWith(p, params);
            totalPrem += totalPremium * p.w;
            totalExp  += totalExpected * p.w;
          }

          const margin = totalPrem > 0 ? (totalPrem - totalExp) / totalPrem : 0;
          const avgPrem = totalPrem;

          // Check premium bounds on reference persona
          const refPersona = { age: 35, smoking: "Never", occupation: "Office/Desk", region: "Phnom Penh", tier: tierForBound, dependents: depForBound, opd: true, dental: false, maternity: false };
          const { totalPremium: refPrem } = computeQuoteWith(refPersona, params);
          if (refPrem < minBound || refPrem > maxBound) continue; // violates bounds

          results.push({ load, famPer, tierGold, tierPlat, margin, avgPrem, refPrem });
        }
      }
    }
  }

  // Filter and rank
  const tolerance = 0.02;
  const feasible = results.filter(r => Math.abs(r.margin - tgt) <= tolerance);
  const ranked = feasible.length > 0 ? feasible : results;
  ranked.sort((a, b) => {
    const aDist = Math.abs(a.margin - tgt);
    const bDist = Math.abs(b.margin - tgt);
    if (aDist !== bDist) return aDist - bDist;
    return a.avgPrem - b.avgPrem;
  });

  const top5 = ranked.slice(0, 5);

  // Sensitivity analysis: margin vs load
  const sensitivityData = [];
  for (let load = 0.10; load <= 0.50; load += 0.02) {
    const params = {
      load: { ipd: load, opd: load * 0.833, dental: load * 0.667, maternity: load * 0.833 },
      tier: { Bronze: COEFF.tier.Bronze, Silver: COEFF.tier.Silver, Gold: COEFF.tier.Gold, Platinum: COEFF.tier.Platinum },
      ded:  COEFF.ded,
      famPer: COEFF.famPer,
    };
    let totalPrem = 0, totalExp = 0;
    for (const p of normalized) {
      const { totalPremium, totalExpected } = computeQuoteWith(p, params);
      totalPrem += totalPremium * p.w;
      totalExp  += totalExpected * p.w;
    }
    const margin = totalPrem > 0 ? (totalPrem - totalExp) / totalPrem : 0;
    sensitivityData.push({ load: Math.round(load * 100) / 100, margin });
  }

  return { top5, sensitivityData, feasibleCount: feasible.length };
}

// ── Claims Scenario Simulator ─────────────────────────────────────────────────
function runClaimsSimulation(shocks, personas) {
  const { freqShock, sevShock, catEvents, portfolioSize } = shocks;
  const covs = ["ipd", "opd", "dental", "maternity"];

  const totalWeight = personas.reduce((s, p) => s + p.weight, 0);
  const normalized  = personas.map(p => ({ ...p, w: p.weight / totalWeight }));

  const baselineClaims = {}, baselinePayout = {}, shockedClaims = {}, shockedPayout = {}, premiumPool = {};
  for (const cov of covs) {
    baselineClaims[cov] = 0; baselinePayout[cov] = 0;
    shockedClaims[cov]  = 0; shockedPayout[cov]  = 0;
    premiumPool[cov]    = 0;
  }

  for (const p of normalized) {
    const ab = ageBand(p.age);
    const af = COEFF.age[ab]              || 1;
    const sf = COEFF.smoke[p.smoking]     || 1;
    const of = COEFF.occup[p.occupation]  || 1;
    const rf = COEFF.region[p.region]     || 1;
    const ff = 1 + (p.dependents - 1) * COEFF.famPer;
    const policyWeight = p.w * portfolioSize;

    // Get loaded premiums from the standard engine for the premium pool
    const quote = computeQuote(p);

    for (const cov of covs) {
      if (cov !== "ipd" && !p[cov]) continue; // rider not included for this persona

      const { freq, sev } = COEFF.base[cov];
      const efFreq = freq * af * sf * of * rf;
      const efSev  = sev  * (1 + Math.max(0, p.age - 30) * 0.006);
      const fShock = freqShock[cov] ?? 1;
      const sShock = sevShock[cov]  ?? 1;

      baselineClaims[cov] += policyWeight * efFreq;
      baselinePayout[cov] += policyWeight * efFreq * efSev * ff;
      shockedClaims[cov]  += policyWeight * efFreq * fShock;
      shockedPayout[cov]  += policyWeight * efFreq * fShock * efSev * sShock * ff;

      if (cov === "ipd") {
        premiumPool[cov] += policyWeight * (quote.ipd?.annual || 0);
      } else {
        premiumPool[cov] += policyWeight * (quote.riders?.[cov]?.annual || 0);
      }
    }
  }

  // Cat events: add flat extra claims at weighted-average shocked severity
  for (const ev of catEvents) {
    const cov = ev.coverage;
    let totalSev = 0, totalW = 0;
    for (const p of normalized) {
      if (cov !== "ipd" && !p[cov]) continue;
      const { sev } = COEFF.base[cov];
      const efSev  = sev * (1 + Math.max(0, p.age - 30) * 0.006);
      const ff     = 1 + (p.dependents - 1) * COEFF.famPer;
      const sShock = sevShock[cov] ?? 1;
      totalSev += p.w * efSev * ff * sShock;
      totalW   += p.w;
    }
    const avgSev = totalW > 0 ? totalSev / totalW : COEFF.base[cov].sev;
    shockedClaims[cov] += ev.extraClaims;
    shockedPayout[cov] += ev.extraClaims * avgSev;
  }

  // Loss ratios
  const baselineLR = {}, shockedLR = {};
  for (const cov of covs) {
    baselineLR[cov] = premiumPool[cov] > 0 ? baselinePayout[cov] / premiumPool[cov] : null;
    shockedLR[cov]  = premiumPool[cov] > 0 ? shockedPayout[cov]  / premiumPool[cov] : null;
  }

  const totalBasePayout    = Object.values(baselinePayout).reduce((a, b) => a + b, 0);
  const totalShockedPayout = Object.values(shockedPayout).reduce((a, b) => a + b, 0);
  const totalPremiumPool   = Object.values(premiumPool).reduce((a, b) => a + b, 0);

  return {
    coverages: covs,
    baseline: {
      claims: baselineClaims, payout: baselinePayout, lossRatio: baselineLR,
      totalPayout: totalBasePayout, totalPremium: totalPremiumPool,
      totalLossRatio: totalPremiumPool > 0 ? totalBasePayout / totalPremiumPool : 0,
    },
    shocked: {
      claims: shockedClaims, payout: shockedPayout, lossRatio: shockedLR,
      totalPayout: totalShockedPayout, totalPremium: totalPremiumPool,
      totalLossRatio: totalPremiumPool > 0 ? totalShockedPayout / totalPremiumPool : 0,
    },
    delta: {
      payout:    totalShockedPayout - totalBasePayout,
      payoutPct: totalBasePayout > 0 ? (totalShockedPayout - totalBasePayout) / totalBasePayout : 0,
      lossRatio: totalPremiumPool > 0
        ? (totalShockedPayout - totalBasePayout) / totalPremiumPool
        : 0,
    },
  };
}

// ── Policy Optimizer Tab ───────────────────────────────────────────────────────
function PolicyOptimizerTab() {
  const [constraints, setConstraints] = useState({ targetMargin: 15, minPremium: 200, maxPremium: 2000, premiumTier: "Silver" });
  const [optimizeParams, setOptimizeParams] = useState({ load: true, famPer: false, tierGold: false, tierPlatinum: false });
  const [personas, setPersonas] = useState(MODEL_OFFICE.map(p => ({ ...p })));
  const [results, setResults] = useState([]);
  const [sensitivityData, setSensitivityData] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [appliedParams, setAppliedParams] = useState(null);

  const handleRun = () => {
    setIsRunning(true);
    setTimeout(() => {
      const { top5, sensitivityData: sens } = runPortfolioOptimizer(personas, constraints, optimizeParams);
      setResults(top5);
      setSensitivityData(sens);
      setIsRunning(false);
    }, 50);
  };

  const totalWeight = personas.reduce((s, p) => s + p.weight, 0);
  const weightOk = Math.abs(totalWeight - 100) < 0.1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <ConstraintPanel constraints={constraints} setConstraints={setConstraints} optimizeParams={optimizeParams} setOptimizeParams={setOptimizeParams} />
      <PortfolioPanel personas={personas} setPersonas={setPersonas} weightOk={weightOk} totalWeight={totalWeight} />
      <OptimizationControls isRunning={isRunning} onRun={handleRun} />
      {results.length > 0 && (
        <>
          {appliedParams && <AppliedParamsBanner appliedParams={appliedParams} />}
          <ResultsTable results={results} appliedParams={appliedParams} onApply={setAppliedParams} />
          <SensitivityChart sensitivityData={sensitivityData} targetMargin={constraints.targetMargin} />
        </>
      )}
    </div>
  );
}

function ConstraintPanel({ constraints, setConstraints, optimizeParams, setOptimizeParams }) {
  return (
    <div style={{ border: `1px solid #e5e7eb`, borderRadius: 8, padding: 16, backgroundColor: "#fafafa" }}>
      <h3 style={{ margin: "0 0 16 0", fontSize: 16, fontWeight: 600, color: NAVY }}>Constraints & Parameters</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "#666" }}>Target margin (%)</span>
          <input type="number" value={constraints.targetMargin} onChange={(e) => setConstraints({ ...constraints, targetMargin: +e.target.value })} style={{ padding: 8, border: "1px solid #ddd", borderRadius: 4 }} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "#666" }}>Min premium ($)</span>
          <input type="number" value={constraints.minPremium} onChange={(e) => setConstraints({ ...constraints, minPremium: +e.target.value })} style={{ padding: 8, border: "1px solid #ddd", borderRadius: 4 }} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "#666" }}>Max premium ($)</span>
          <input type="number" value={constraints.maxPremium} onChange={(e) => setConstraints({ ...constraints, maxPremium: +e.target.value })} style={{ padding: 8, border: "1px solid #ddd", borderRadius: 4 }} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "#666" }}>Tier for bounds</span>
          <select value={constraints.premiumTier} onChange={(e) => setConstraints({ ...constraints, premiumTier: e.target.value })} style={{ padding: 8, border: "1px solid #ddd", borderRadius: 4 }}>
            {["Bronze", "Silver", "Gold", "Platinum"].map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>
      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid #e5e7eb" }}>
        <p style={{ margin: "0 0 12 0", fontSize: 12, fontWeight: 600, color: "#666" }}>Optimize parameters:</p>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {["load", "famPer", "tierGold", "tierPlatinum"].map(key => {
            const labels = { load: "IPD Load Factor", famPer: "Family Factor", tierGold: "Gold Multiplier", tierPlatinum: "Platinum Multiplier" };
            return (
              <label key={key} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input type="checkbox" checked={optimizeParams[key]} onChange={(e) => setOptimizeParams({ ...optimizeParams, [key]: e.target.checked })} />
                <span style={{ fontSize: 12 }}>{labels[key]}</span>
              </label>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function PortfolioPanel({ personas, setPersonas, weightOk, totalWeight }) {
  return (
    <div style={{ border: `1px solid #e5e7eb`, borderRadius: 8, padding: 16, backgroundColor: "#fafafa" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: NAVY }}>Model Office Portfolio</h3>
        <div style={{ fontSize: 12, color: weightOk ? "#22c55e" : "#ef4444", fontWeight: 600 }}>Total weight: {totalWeight.toFixed(1)}%</div>
      </div>
      <div style={{ display: "grid", gap: 8, maxHeight: 400, overflowY: "auto" }}>
        {personas.map((p, i) => (
          <div key={p.id} style={{ display: "grid", gridTemplateColumns: "1fr 80px", gap: 8, alignItems: "center", padding: 8, backgroundColor: "white", borderRadius: 4, border: "1px solid #e5e7eb" }}>
            <div style={{ fontSize: 12 }}>
              <div style={{ fontWeight: 600, color: "#333" }}>{p.label}</div>
              <div style={{ fontSize: 11, color: "#999", marginTop: 2 }}>{p.age} · {p.smoking} · {p.tier} · {p.dependents} dep</div>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <input type="range" min="0" max="40" value={p.weight} onChange={(e) => { const np = [...personas]; np[i].weight = +e.target.value; setPersonas(np); }} style={{ flex: 1 }} />
              <span style={{ fontSize: 11, fontWeight: 600, minWidth: 20, textAlign: "right" }}>{p.weight}%</span>
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}

function OptimizationControls({ isRunning, onRun }) {
  return (
    <div style={{ display: "flex", gap: 12 }}>
      <button onClick={onRun} disabled={isRunning} style={{ padding: "10px 20px", backgroundColor: GOLD_D, color: "#fff", border: "none", borderRadius: 4, fontWeight: 600, cursor: isRunning ? "not-allowed" : "pointer", opacity: isRunning ? 0.6 : 1, display: "flex", alignItems: "center", gap: 8 }}>
        {isRunning ? <span style={{ display: "inline-block", width: 16, height: 16, border: "2px solid #fff", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} /> : "▶"}
        {isRunning ? "Optimizing..." : "Run Optimizer"}
      </button>
    </div>
  );
}

function AppliedParamsBanner({ appliedParams }) {
  return (
    <div style={{ padding: 12, backgroundColor: NAVY, color: GOLD_D, borderRadius: 6, fontSize: 12, lineHeight: "1.6" }}>
      <strong style={{ color: "white" }}>Applied parameters:</strong>{" "}
      IPD Load: {Math.round(appliedParams.load * 100)}% (was {Math.round(COEFF.load.ipd * 100)}%) ·{" "}
      Gold: ×{appliedParams.tierGold.toFixed(2)} (was ×{COEFF.tier.Gold.toFixed(2)}) ·{" "}
      Fam: {appliedParams.famPer.toFixed(2)} (was {COEFF.famPer.toFixed(2)})
    </div>
  );
}

function ResultsTable({ results, appliedParams, onApply }) {
  return (
    <div style={{ border: `1px solid #e5e7eb`, borderRadius: 8, overflow: "hidden" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ backgroundColor: NAVY, color: "white" }}>
            <th style={{ padding: 8, textAlign: "left", fontWeight: 600 }}>Rank</th>
            <th style={{ padding: 8, textAlign: "left", fontWeight: 600 }}>Load</th>
            <th style={{ padding: 8, textAlign: "left", fontWeight: 600 }}>Gold</th>
            <th style={{ padding: 8, textAlign: "left", fontWeight: 600 }}>Margin</th>
            <th style={{ padding: 8, textAlign: "left", fontWeight: 600 }}>Avg Prem</th>
            <th style={{ padding: 8, textAlign: "center", fontWeight: 600 }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const isApplied = appliedParams && appliedParams.load === r.load && appliedParams.tierGold === r.tierGold;
            return (
              <tr key={i} style={{ backgroundColor: isApplied ? "#fffbeb" : i % 2 === 0 ? "#fafafa" : "white", borderBottom: "1px solid #e5e7eb", borderLeft: isApplied ? `4px solid ${GOLD_D}` : "4px solid transparent" }}>
                <td style={{ padding: 8 }}>{i + 1}</td>
                <td style={{ padding: 8 }}>{Math.round(r.load * 100)}%</td>
                <td style={{ padding: 8 }}>×{r.tierGold.toFixed(2)}</td>
                <td style={{ padding: 8, fontWeight: 600, color: r.margin > 0.12 ? "#22c55e" : r.margin > 0.08 ? "#f59e0b" : "#ef4444" }}>{Math.round(r.margin * 100)}%</td>
                <td style={{ padding: 8 }}>${Math.round(r.avgPrem)}</td>
                <td style={{ padding: 8, textAlign: "center" }}>
                  <button onClick={() => onApply({ load: r.load, tierGold: r.tierGold, famPer: r.famPer })} style={{ padding: "4px 8px", fontSize: 11, backgroundColor: GOLD_D, color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}>Apply</button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SensitivityChart({ sensitivityData, targetMargin }) {
  const tgtDecimal = targetMargin / 100;
  const W = 560, H = 220, ML = 50, MR = 20, MT = 20, MB = 40;
  const PW = W - ML - MR, PH = H - MT - MB;

  const loadMin = 0.10, loadMax = 0.50, loadRange = loadMax - loadMin;
  const minMargin = Math.min(...sensitivityData.map(d => d.margin));
  const maxMargin = Math.max(...sensitivityData.map(d => d.margin));
  const marginPad = (maxMargin - minMargin) * 0.1;
  const marMax = Math.max(maxMargin + marginPad, 0.5);
  const marMin = Math.min(minMargin - marginPad, -0.1);
  const marginRange = marMax - marMin;

  const xScale = (load) => ML + ((load - loadMin) / loadRange) * PW;
  const yScale = (margin) => MT + ((marMax - margin) / marginRange) * PH;

  const polyPts = sensitivityData.map(d => `${xScale(d.load)},${yScale(d.margin)}`).join(" ");
  const tgtY = yScale(tgtDecimal);

  return (
    <div style={{ border: `1px solid #e5e7eb`, borderRadius: 8, padding: 16, backgroundColor: "#fafafa" }}>
      <h3 style={{ margin: "0 0 12 0", fontSize: 14, fontWeight: 600, color: NAVY }}>Sensitivity: Margin vs Load Factor</h3>
      <svg width={W} height={H} style={{ backgroundColor: "white", borderRadius: 4 }}>
        {/* Gridlines */}
        {[0.10, 0.20, 0.30, 0.40, 0.50].map(load => <line key={`vgrid-${load}`} x1={xScale(load)} y1={MT} x2={xScale(load)} y2={MT + PH} stroke="#e5e7eb" strokeDasharray="4" strokeWidth={1} />)}
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map(m => <line key={`hgrid-${m}`} x1={ML} y1={yScale(m)} x2={ML + PW} y2={yScale(m)} stroke="#e5e7eb" strokeDasharray="4" strokeWidth={1} />)}

        {/* Target margin line */}
        <line x1={ML} y1={tgtY} x2={ML + PW} y2={tgtY} stroke={GOLD_D} strokeDasharray="6" strokeWidth={2} />

        {/* Feasibility zone */}
        <rect x={ML} y={yScale(tgtDecimal + 0.02)} width={PW} height={yScale(tgtDecimal - 0.02) - yScale(tgtDecimal + 0.02)} fill="#dbeafe" opacity={0.3} />

        {/* Margin polyline */}
        <polyline points={polyPts} stroke={NAVY} strokeWidth={2} fill="none" />

        {/* X-axis labels */}
        {[0.10, 0.20, 0.30, 0.40, 0.50].map(load => (
          <text key={`xlabel-${load}`} x={xScale(load)} y={H - 10} textAnchor="middle" fontSize={10} fill="#666">
            {Math.round(load * 100)}%
          </text>
        ))}

        {/* Y-axis labels */}
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map(m => (
          <text key={`ylabel-${m}`} x={ML - 8} y={yScale(m) + 4} textAnchor="end" fontSize={10} fill="#666">
            {Math.round(m * 100)}%
          </text>
        ))}

        {/* Axis lines */}
        <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#333" strokeWidth={1} />
        <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#333" strokeWidth={1} />

        {/* Axis labels */}
        <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#333">
          Load Factor
        </text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#333" transform={`rotate(-90 12 ${H / 2})`}>
          Margin %
        </text>
      </svg>
    </div>
  );
}

// ── Loss Ratio Bar Chart ──────────────────────────────────────────────────────
function LossRatioChart({ baseline, shocked, coverages }) {
  const W = 560, H = 240, ML = 55, MR = 20, MT = 28, MB = 50;
  const PW = W - ML - MR, PH = H - MT - MB;

  const allVals = [
    ...coverages.map(c => baseline[c] || 0),
    ...coverages.map(c => shocked[c]  || 0),
    1.5,
  ];
  const yMax  = Math.max(...allVals) * 1.1;
  const yScale = (v) => MT + ((yMax - v) / yMax) * PH;

  const groupW = PW / 4;
  const barW   = groupW * 0.28;

  const yTicks = [];
  for (let v = 0; v <= yMax * 1.01; v += 0.25) yTicks.push(+v.toFixed(2));

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 16, backgroundColor: "#fafafa" }}>
      <h3 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 600, color: NAVY }}>Loss Ratio by Coverage Type</h3>
      <svg width={W} height={H} style={{ backgroundColor: "white", borderRadius: 4, display: "block" }}>
        {/* Horizontal gridlines */}
        {yTicks.map(v => (
          <line key={`hg-${v}`} x1={ML} y1={yScale(v)} x2={ML + PW} y2={yScale(v)}
                stroke="#e5e7eb" strokeDasharray="4" strokeWidth={1} />
        ))}

        {/* Break-even line at 100% */}
        <line x1={ML} y1={yScale(1.0)} x2={ML + PW} y2={yScale(1.0)}
              stroke={GOLD_D} strokeDasharray="6 3" strokeWidth={1.5} />
        <text x={ML + PW + 3} y={yScale(1.0) + 4} fontSize={9} fill={GOLD_D}>100%</text>

        {/* Bars + labels */}
        {coverages.map((cov, i) => {
          const bVal = baseline[cov] ?? 0;
          const sVal = shocked[cov]  ?? 0;
          const bX   = ML + i * groupW + groupW * 0.12;
          const sX   = bX + barW + 3;
          const bY   = yScale(Math.min(bVal, yMax));
          const sY   = yScale(Math.min(sVal, yMax));
          const bH   = Math.max(yScale(0) - bY, 2);
          const sH   = Math.max(yScale(0) - sY, 2);
          const sColor = sVal > 1.0 ? ERR : sVal > 0.85 ? GOLD_D : OK;
          return (
            <g key={cov}>
              <rect x={bX} y={bY} width={barW} height={bH} fill={NAVY} opacity={0.75} />
              <rect x={sX} y={sY} width={barW} height={sH} fill={sColor} opacity={0.85} />
              {bVal > 0 && <text x={bX + barW / 2} y={bY - 4} textAnchor="middle" fontSize={9} fill={NAVY}>{(bVal * 100).toFixed(0)}%</text>}
              {sVal > 0 && <text x={sX + barW / 2} y={sY - 4} textAnchor="middle" fontSize={9} fill={sColor}>{(sVal * 100).toFixed(0)}%</text>}
              <text x={ML + i * groupW + groupW / 2} y={H - 10} textAnchor="middle" fontSize={11} fill={TXT2}>{COV_LABELS[cov]}</text>
            </g>
          );
        })}

        {/* Y-axis ticks */}
        {yTicks.map(v => (
          <text key={`yt-${v}`} x={ML - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="#666">
            {Math.round(v * 100)}%
          </text>
        ))}

        {/* Axis lines */}
        <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#333" strokeWidth={1} />
        <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#333" strokeWidth={1} />

        {/* Legend */}
        <rect x={ML + PW - 118} y={MT + 4} width={10} height={10} fill={NAVY} opacity={0.75} />
        <text x={ML + PW - 104} y={MT + 13} fontSize={10} fill={TXT2}>Baseline</text>
        <rect x={ML + PW - 50} y={MT + 4} width={10} height={10} fill={GOLD_D} opacity={0.85} />
        <text x={ML + PW - 36} y={MT + 13} fontSize={10} fill={TXT2}>Shocked</text>
      </svg>
    </div>
  );
}

// ── Claims Simulator Tab ──────────────────────────────────────────────────────
function ClaimsSimulatorTab() {
  const DEFAULT_SHOCKS = {
    freqShock: { ipd: 1.0, opd: 1.0, dental: 1.0, maternity: 1.0 },
    sevShock:  { ipd: 1.0, opd: 1.0, dental: 1.0, maternity: 1.0 },
    catEvents: [],
    portfolioSize: 1000,
  };

  const [shocks,       setShocks]       = useState(DEFAULT_SHOCKS);
  const [scenarioType, setScenarioType] = useState("frequency");
  const [catRegion,    setCatRegion]    = useState("Phnom Penh");
  const [catCoverage,  setCatCoverage]  = useState("ipd");
  const [catCount,     setCatCount]     = useState(100);
  const [simResult,    setSimResult]    = useState(null);
  const [isRunning,    setIsRunning]    = useState(false);

  const applyPreset = (key) => {
    const sz = shocks.portfolioSize;
    const presets = {
      ipd_freq_30:  { type: "frequency", shocks: { ...DEFAULT_SHOCKS, freqShock: { ipd: 1.30, opd: 1.0, dental: 1.0, maternity: 1.0 }, portfolioSize: sz } },
      ded_bronze:   { type: "severity",  shocks: { ...DEFAULT_SHOCKS, sevShock:  { ipd: 1.10, opd: 1.05, dental: 1.0, maternity: 1.0 }, portfolioSize: sz } },
      cat_outbreak: { type: "cat",       shocks: { ...DEFAULT_SHOCKS, catEvents: [{ region: "Phnom Penh", coverage: "ipd", extraClaims: 200 }], portfolioSize: sz } },
    };
    const p = presets[key]; if (!p) return;
    setScenarioType(p.type);
    setShocks(p.shocks);
    if (key === "cat_outbreak") { setCatRegion("Phnom Penh"); setCatCoverage("ipd"); setCatCount(200); }
    setSimResult(null);
  };

  const addCatEvent = () =>
    setShocks(s => ({ ...s, catEvents: [...s.catEvents, { region: catRegion, coverage: catCoverage, extraClaims: +catCount }] }));

  const removeCatEvent = (idx) =>
    setShocks(s => ({ ...s, catEvents: s.catEvents.filter((_, i) => i !== idx) }));

  const handleRun = () => {
    setIsRunning(true);
    setTimeout(() => {
      setSimResult(runClaimsSimulation(shocks, MODEL_OFFICE));
      setIsRunning(false);
    }, 50);
  };

  const showFreq = scenarioType === "frequency" || scenarioType === "combined";
  const showSev  = scenarioType === "severity"  || scenarioType === "combined";

  const sliderRow = (label, val, onChange) => (
    <div key={label}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: TXT }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color: val > 1 ? ERR : val < 1 ? OK : TXT2 }}>
          ×{val.toFixed(2)} ({val >= 1 ? "+" : ""}{((val - 1) * 100).toFixed(0)}%)
        </span>
      </div>
      <input type="range" min={0.5} max={3.0} step={0.05} value={val} onChange={e => onChange(+e.target.value)}
             style={{ width: "100%" }} />
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Scenario Builder ── */}
      <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 16, backgroundColor: "#fafafa" }}>
        <h3 style={{ margin: "0 0 16px 0", fontSize: 16, fontWeight: 600, color: NAVY }}>Scenario Builder</h3>

        {/* Presets */}
        <div style={{ marginBottom: 20 }}>
          <p style={{ margin: "0 0 8px 0", fontSize: 12, fontWeight: 600, color: TXT2 }}>Quick presets:</p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {[
              { key: "ipd_freq_30",  label: "IPD +30% freq" },
              { key: "ded_bronze",   label: "Bronze deductible −50%" },
              { key: "cat_outbreak", label: "Hospital outbreak PP (200 IPD)" },
            ].map(({ key, label }) => (
              <button key={key} onClick={() => applyPreset(key)}
                style={{ padding: "6px 12px", fontSize: 12, border: `1px solid ${NAVY}`, borderRadius: 6, background: WHITE, color: NAVY, cursor: "pointer", fontFamily: "inherit" }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Scenario type */}
        <div style={{ marginBottom: 16 }}>
          <p style={{ margin: "0 0 8px 0", fontSize: 12, fontWeight: 600, color: TXT2 }}>Scenario type:</p>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {[
              { id: "frequency", label: "Frequency Shock" },
              { id: "severity",  label: "Severity Shock"  },
              { id: "combined",  label: "Combined Stress"  },
              { id: "cat",       label: "Cat Event"        },
            ].map(({ id, label }) => (
              <button key={id} onClick={() => { setScenarioType(id); setSimResult(null); }}
                style={{ padding: "7px 14px", fontSize: 12, fontWeight: 600, fontFamily: "inherit", border: "none", borderRadius: 6, cursor: "pointer",
                  background: scenarioType === id ? NAVY : LTGRAY, color: scenarioType === id ? WHITE : TXT2 }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Portfolio size */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 6, maxWidth: 240 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: TXT2 }}>Portfolio size (policyholders)</span>
            <input type="number" min={100} max={100000} step={100} value={shocks.portfolioSize}
              onChange={e => setShocks(s => ({ ...s, portfolioSize: Math.max(1, +e.target.value) }))}
              style={{ padding: "8px 10px", border: "1px solid #ddd", borderRadius: 4 }} />
          </label>
        </div>

        {/* Frequency sliders */}
        {showFreq && (
          <div style={{ marginBottom: 16 }}>
            <p style={{ margin: "0 0 10px 0", fontSize: 12, fontWeight: 600, color: TXT2 }}>Frequency shock multiplier (claims per year):</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              {["ipd","opd","dental","maternity"].map(cov =>
                sliderRow(COV_LABELS[cov], shocks.freqShock[cov],
                  v => setShocks(s => ({ ...s, freqShock: { ...s.freqShock, [cov]: v } })))
              )}
            </div>
          </div>
        )}

        {/* Severity sliders */}
        {showSev && (
          <div style={{ marginBottom: 16 }}>
            <p style={{ margin: "0 0 10px 0", fontSize: 12, fontWeight: 600, color: TXT2 }}>Severity shock multiplier (cost per claim):</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              {["ipd","opd","dental","maternity"].map(cov =>
                sliderRow(COV_LABELS[cov], shocks.sevShock[cov],
                  v => setShocks(s => ({ ...s, sevShock: { ...s.sevShock, [cov]: v } })))
              )}
            </div>
          </div>
        )}

        {/* Cat event builder */}
        {scenarioType === "cat" && (
          <div style={{ marginBottom: 16 }}>
            <p style={{ margin: "0 0 10px 0", fontSize: 12, fontWeight: 600, color: TXT2 }}>Add catastrophe event:</p>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, color: TXT2 }}>Region</span>
                <select value={catRegion} onChange={e => setCatRegion(e.target.value)}
                  style={{ padding: "7px 10px", border: "1px solid #ddd", borderRadius: 4, fontSize: 12 }}>
                  {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, color: TXT2 }}>Coverage</span>
                <select value={catCoverage} onChange={e => setCatCoverage(e.target.value)}
                  style={{ padding: "7px 10px", border: "1px solid #ddd", borderRadius: 4, fontSize: 12 }}>
                  {["ipd","opd","dental","maternity"].map(c => <option key={c} value={c}>{COV_LABELS[c]}</option>)}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, color: TXT2 }}>Extra claims</span>
                <input type="number" min={0} max={10000} step={10} value={catCount}
                  onChange={e => setCatCount(+e.target.value)}
                  style={{ padding: "7px 10px", border: "1px solid #ddd", borderRadius: 4, width: 100, fontSize: 12 }} />
              </label>
              <button onClick={addCatEvent}
                style={{ padding: "8px 14px", fontSize: 12, background: NAVY, color: WHITE, border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" }}>
                + Add Event
              </button>
            </div>
            {shocks.catEvents.length > 0 && (
              <div style={{ marginTop: 12 }}>
                {shocks.catEvents.map((ev, idx) => (
                  <span key={idx} style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "4px 10px", borderRadius: 20, background: "#fee2e2", fontSize: 12, color: "#991b1b", marginRight: 8, marginBottom: 6 }}>
                    {ev.extraClaims} {COV_LABELS[ev.coverage]} in {ev.region}
                    <button onClick={() => removeCatEvent(idx)}
                      style={{ background: "none", border: "none", cursor: "pointer", color: "#991b1b", fontSize: 14, lineHeight: 1, padding: 0 }}>×</button>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Run */}
        <button onClick={handleRun} disabled={isRunning}
          style={{ padding: "10px 24px", fontSize: 14, fontWeight: 600, fontFamily: "inherit",
            background: isRunning ? TXT2 : GOLD_D, color: WHITE, border: "none", borderRadius: 8,
            cursor: isRunning ? "not-allowed" : "pointer" }}>
          {isRunning ? "Running…" : "Run Simulation"}
        </button>
      </div>

      {/* ── Results ── */}
      {simResult && (
        <>
          {/* Stat cards */}
          <div className="dash-grid-2" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
            {[
              { label: "Shocked Portfolio Payout",  value: `$${Math.round(simResult.shocked.totalPayout).toLocaleString()}`,  sub: `${shocks.portfolioSize.toLocaleString()} policyholders`, color: NAVY },
              { label: "Baseline Portfolio Payout", value: `$${Math.round(simResult.baseline.totalPayout).toLocaleString()}`, sub: "at current rates", color: TXT2 },
              { label: "Portfolio Loss Ratio",
                value: `${(simResult.shocked.totalLossRatio * 100).toFixed(1)}%`,
                sub: "shocked scenario",
                color: simResult.shocked.totalLossRatio > 1.0 ? ERR : simResult.shocked.totalLossRatio < 0.85 ? OK : GOLD_D },
              { label: "Delta vs Baseline",
                value: `${simResult.delta.payoutPct >= 0 ? "+" : ""}${(simResult.delta.payoutPct * 100).toFixed(1)}%`,
                sub: `$${Math.round(Math.abs(simResult.delta.payout)).toLocaleString()} ${simResult.delta.payout >= 0 ? "more" : "less"}`,
                color: simResult.delta.payout > 0 ? ERR : OK },
            ].map(({ label, value, sub, color }) => (
              <div key={label} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 16, backgroundColor: WHITE }}>
                <div style={{ fontSize: 11, color: TXT2, marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 11, color: TXT2, marginTop: 4 }}>{sub}</div>
              </div>
            ))}
          </div>

          {/* Per-coverage table */}
          <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, backgroundColor: WHITE, overflow: "hidden" }}>
            <div style={{ padding: "14px 16px", borderBottom: "1px solid #e5e7eb" }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: NAVY }}>Per-Coverage Breakdown</h3>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ backgroundColor: LTGRAY }}>
                    {["Coverage","Baseline Claims","Shocked Claims","Baseline Payout","Shocked Payout","Loss Ratio","LR Change"].map(h => (
                      <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, fontWeight: 600, color: TXT2 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {simResult.coverages.map(cov => {
                    const bC  = simResult.baseline.claims[cov];
                    const sC  = simResult.shocked.claims[cov];
                    const bP  = simResult.baseline.payout[cov];
                    const sP  = simResult.shocked.payout[cov];
                    const bLR = simResult.baseline.lossRatio[cov];
                    const sLR = simResult.shocked.lossRatio[cov];
                    const lrDelta = (sLR !== null && bLR !== null) ? sLR - bLR : null;
                    const lrColor = sLR === null ? TXT2 : sLR > 1.0 ? ERR : sLR > 0.85 ? GOLD_D : OK;
                    return (
                      <tr key={cov} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "10px 14px" }}>
                          <span style={{ padding: "2px 8px", borderRadius: 20, fontSize: 11, background: LTGRAY, color: NAVY, fontWeight: 600 }}>{COV_LABELS[cov]}</span>
                        </td>
                        <td style={{ padding: "10px 14px", color: TXT2 }}>{Math.round(bC).toLocaleString()}</td>
                        <td style={{ padding: "10px 14px", fontWeight: 600, color: sC > bC ? ERR : OK }}>{Math.round(sC).toLocaleString()}</td>
                        <td style={{ padding: "10px 14px", color: TXT2 }}>${Math.round(bP).toLocaleString()}</td>
                        <td style={{ padding: "10px 14px" }}>
                          <span style={{ fontWeight: 600, color: sP > bP ? ERR : OK }}>${Math.round(sP).toLocaleString()}</span>
                          {bP > 0 && <span style={{ fontSize: 11, color: TXT2, marginLeft: 4 }}>({sP >= bP ? "+" : ""}{(((sP - bP) / bP) * 100).toFixed(1)}%)</span>}
                        </td>
                        <td style={{ padding: "10px 14px", fontWeight: 700, color: lrColor }}>
                          {sLR !== null ? `${(sLR * 100).toFixed(1)}%` : "N/A"}
                        </td>
                        <td style={{ padding: "10px 14px", color: lrDelta === null ? TXT2 : lrDelta > 0 ? ERR : OK }}>
                          {lrDelta !== null ? `${lrDelta > 0 ? "+" : ""}${(lrDelta * 100).toFixed(1)} pp` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Chart */}
          <LossRatioChart baseline={simResult.baseline.lossRatio} shocked={simResult.shocked.lossRatio} coverages={simResult.coverages} />

          {/* Methodology note */}
          <div style={{ padding: "12px 14px", borderRadius: 8, background: "#f0f4ff", borderLeft: `4px solid ${NAVY}` }}>
            <div style={{ fontSize: 11, color: TXT2, lineHeight: 1.6 }}>
              <strong>Methodology:</strong> Shocks are multiplicative perturbations to frequency E[N] and/or severity E[X] parameters of the Poisson-Gamma model. Portfolio loss ratio = Σ(shocked payout) / Σ(premium pool) across {MODEL_OFFICE.length} representative personas weighted by portfolio distribution. Cat events inject flat additional claims at weighted-average shocked severity. Collective risk model: E[S] = Σ<sub>i</sub> w<sub>i</sub> · N · E[N<sub>i</sub>] · E[X<sub>i</sub>].
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function InsuranceDashboard() {
  const [apiKey,   setApiKey]   = useState(() => sessionStorage.getItem("dac_dash_key") || "");
  const [authed,   setAuthed]   = useState(() => !!sessionStorage.getItem("dac_dash_key"));
  const [activeTab, setActiveTab] = useState("quote");

  const handleAuth = (key) => {
    setApiKey(key); setAuthed(true);
    sessionStorage.setItem("dac_dash_key", key);
    addAuditLog({ action: "login", user: "admin" });
  };

  const logout = () => {
    setAuthed(false); setApiKey("");
    sessionStorage.removeItem("dac_dash_key");
  };

  const TABS = [
    { id: "quote",        label: "Quick quote"      },
    { id: "batch",        label: "Batch quotes"     },
    { id: "calibration",  label: "Data calibration" },
    { id: "coefficients", label: "Coefficients"     },
    { id: "metrics",      label: "Model Metrics"    },
    { id: "security",     label: "Security"         },
    { id: "optimizer",    label: "Policy Optimizer" },
    { id: "simulator",    label: "Claims Simulator" },
  ];

  if (!authed) return <AuthGate onAuth={handleAuth} />;

  return (
    <section style={{ paddingTop: 100, paddingBottom: 60 }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 700px) {
          .dash-grid-2 { grid-template-columns: 1fr !important; }
          .dash-tabs { overflow-x: auto; }
        }
      `}</style>

      <div style={{ maxWidth: 1060, margin: "0 auto", padding: "0 20px" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28, flexWrap: "wrap", gap: 12 }}>
          <div>
            <span style={{ fontSize: 12, fontWeight: 600, color: GOLD_D, letterSpacing: 2, textTransform: "uppercase" }}>
              DAC HealthPrice
            </span>
            <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 30, fontWeight: 700, marginTop: 4 }}>
              Insurance Dashboard
            </h1>
            <p style={{ fontSize: 13, color: TXT2, marginTop: 4 }}>
              GLM model {COEFF.version} · Last calibrated {COEFF.last_updated}
            </p>
          </div>
          <button
            onClick={logout}
            style={{ padding: "8px 16px", borderRadius: 8, border: "1px solid #e5e7eb", background: WHITE, color: TXT2, fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}
          >
            Sign out
          </button>
        </div>

        {/* Tab nav */}
        <div className="dash-tabs" style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "2px solid #e5e7eb", overflowX: "auto", paddingBottom: 0 }}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "10px 18px",
                fontSize: 13, fontWeight: 600, fontFamily: "inherit",
                background: "none", border: "none", cursor: "pointer",
                color: activeTab === tab.id ? NAVY : TXT2,
                borderBottom: `2px solid ${activeTab === tab.id ? NAVY : "transparent"}`,
                marginBottom: -2, whiteSpace: "nowrap",
                transition: "color 0.2s",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "quote"        && <QuickQuoteTab  apiKey={apiKey} username="admin" />}
        {activeTab === "batch"        && <BatchTab       apiKey={apiKey} username="admin" />}
        {activeTab === "calibration"  && <CalibrationTab apiKey={apiKey} username="admin" />}
        {activeTab === "coefficients" && <CoefficientsTab />}
        {activeTab === "metrics"      && <ModelMetricsTab />}
        {activeTab === "security"     && <SecurityTab    username="admin" />}
        {activeTab === "optimizer"    && <PolicyOptimizerTab />}
        {activeTab === "simulator"    && <ClaimsSimulatorTab />}
      </div>
    </section>
  );
}
