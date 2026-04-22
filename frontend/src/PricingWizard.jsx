import React, { useState, useEffect, useCallback, useRef } from "react";

const API = "https://dac-healthprice-api.onrender.com";
const LOGO_URL = "/DAC.jpg"; // Your logo in /public

function getBrowserId() {
  let id = localStorage.getItem("dac_browser_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("dac_browser_id", id);
  }
  return id;
}

async function apiCall(path, body, extraHeaders = {}) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json", ...extraHeaders }, body: JSON.stringify(body) }
    : { headers: extraHeaders };
  const r = await fetch(`${API}${path}`, { ...opts, signal: AbortSignal.timeout(45000) });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `Error ${r.status}`);
  }
  return r.json();
}

// ─── Constants ──────────────────────────────────────────────────────────────
const REGIONS = ["Phnom Penh", "Siem Reap", "Battambang", "Sihanoukville", "Kampong Cham", "Rural Areas"];
const GENDERS = ["Male", "Female", "Other"];
const SMOKING = ["Never", "Former", "Current"];
const OCCUPATIONS = ["Office/Desk", "Retail/Service", "Healthcare", "Manual Labor", "Industrial/High-Risk", "Retired"];
const PREEXIST = ["None", "Hypertension", "Diabetes", "Heart Disease", "Asthma/COPD", "Cancer (remission)", "Kidney Disease", "Liver Disease", "Obesity", "Mental Health"];
const TIERS = {
  Bronze: { limit: "$15,000", room: "General Ward", surg: "$5,000", icu: "3 days", ded: "$500", dedN: 500 },
  Silver: { limit: "$40,000", room: "Semi-Private", surg: "$15,000", icu: "7 days", ded: "$250", dedN: 250 },
  Gold: { limit: "$80,000", room: "Private Room", surg: "$40,000", icu: "14 days", ded: "$100", dedN: 100 },
  Platinum: { limit: "$150,000", room: "Private Suite", surg: "$80,000", icu: "30 days", ded: "$0", dedN: 0 },
};
const STEPS = ["Profile", "Health", "Lifestyle", "Plan", "Review", "Quote"];
const FB_FREQ = { ipd: 0.12, opd: 2.5, dental: 0.8, maternity: 0.15 };
const FB_SEV = { ipd: 2500, opd: 60, dental: 120, maternity: 3500 };
const TIER_F = { Bronze: 0.70, Silver: 1.00, Gold: 1.45, Platinum: 2.10 };
const LOAD = { ipd: 0.30, opd: 0.25, dental: 0.20, maternity: 0.25 };

// ─── Session jitter seed (stored in localStorage for consistency) ─────────────
function getJitterSeed() {
  try {
    const stored = localStorage.getItem("dac_jitter");
    if (stored) return parseFloat(stored);
    const seed = (Math.random() * 0.06) - 0.03; // -3% to +3%
    localStorage.setItem("dac_jitter", seed.toString());
    return seed;
  } catch { return 0; }
}
const JITTER = getJitterSeed();

// ─── Display range: applies jitter + ±5% band, rounds to nearest $1 ──────────
function getDisplayRange(exactMonthly) {
  const mid = exactMonthly * (1 + JITTER);
  const lo = Math.round(mid * 0.95);
  const hi = Math.round(mid * 1.05);
  return { lo, hi, text: `$${lo.toLocaleString()} – $${hi.toLocaleString()}` };
}

// ─── Escalation cost-factor calculation (shared by localPrice + calculate) ───
function computeEscalation(baseTotal, ipd_tier) {
  const r = 0.05, d = 0.06, n = 20, cap = 2.5, util = 0.20;
  const pvFlat = Array.from({ length: n }, (_, i) => 1 / (1 + d) ** (i + 1)).reduce((a, b) => a + b, 0);
  const pvEsc  = Array.from({ length: n }, (_, i) => Math.min((1 + r) ** i, cap) / (1 + d) ** (i + 1)).reduce((a, b) => a + b, 0);
  const costFactor = (pvEsc / pvFlat - 1) * util;
  const escalatedAnnual = Math.round(baseTotal * (1 + costFactor) * 100) / 100;
  const baseCoverage = parseFloat((TIERS[ipd_tier]?.limit || "$50,000").replace(/[$,]/g, "")) || 50000;
  const projections = Array.from({ length: n }, (_, i) => ({
    year: i + 1,
    coverage_multiple: Math.round(Math.min((1 + r) ** i, cap) * 10000) / 10000,
    coverage_amount: Math.round(Math.min((1 + r) ** i, cap) * baseCoverage),
    cumulative_escalation_pct: Math.round((Math.min((1 + r) ** i, cap) - 1) * 1000) / 10,
    loyalty_bonus: i + 1 === 10 ? Math.round(escalatedAnnual * 0.25) : i + 1 === 20 ? Math.round(escalatedAnnual * 0.50) : 0,
  }));
  return {
    enabled: true,
    cost_factor: Math.round(costFactor * 10000) / 10000,
    cost_factor_pct: `${(costFactor * 100).toFixed(1)}%`,
    base_annual_premium: baseTotal,
    escalated_annual_premium: escalatedAnnual,
    escalated_monthly_premium: Math.round(escalatedAnnual / 12 * 100) / 100,
    base_coverage: baseCoverage,
    terminal_coverage: baseCoverage * cap,
    projections,
  };
}

// ─── GLM-style log-linear fallback pricing (actuarially consistent) ──────────
// Uses log-linear structure mirroring the Poisson backend model.
function localPrice(inp) {
  // Log-linear coefficients (calibrated to match backend fallback means)
  const BASE_F = { ipd: -2.12, opd: 0.92, dental: -0.22, maternity: -1.90 };
  const BASE_S = { ipd: 7.82, opd: 4.09, dental: 4.79, maternity: 8.16 }; // log scale
  const weeklyMins = (inp.exercise_days || 0) * (inp.exercise_mins || 0);
  const peCount = inp.preexist_conditions.filter(p => p !== "None").length;

  // Shared covariate vector
  const b_age_f = 0.008; const b_age_s = 0.006;
  const b_smoke = { Never: 0, Former: 0.14, Current: 0.34 }[inp.smoking_status] || 0;
  const b_exercise = weeklyMins <= 0 ? 0.18 : weeklyMins < 60 ? 0.10 : weeklyMins < 150 ? -0.05 : -0.16;
  const b_occ = { "Office/Desk": -0.16, "Retail/Service": 0, "Healthcare": 0.05, "Manual Labor": 0.14, "Industrial/High-Risk": 0.26, "Retired": 0.09 }[inp.occupation_type] || 0;
  const b_pe = 0.18;
  const REG_LOG = { "Phnom Penh": 0.18, "Siem Reap": 0.05, "Battambang": -0.11, "Sihanoukville": 0.09, "Kampong Cham": -0.16, "Ho Chi Minh City": 0.22, "Hanoi": 0.18, "Da Nang": 0.05, "Can Tho": -0.11, "Hai Phong": -0.05, "Rural Areas": -0.29 };
  const b_reg = REG_LOG[inp.region] || 0;

  const calc = (cov) => {
    const ageF = inp.age > 35 ? b_age_f * (inp.age - 35) : 0;
    const ageS = inp.age > 30 ? b_age_s * (inp.age - 30) : 0;
    const freq = Math.min(20, Math.max(0.001, Math.exp(BASE_F[cov] + ageF + b_smoke + b_exercise + b_occ + b_pe * peCount)));
    const sev = Math.min(100000, Math.max(10, Math.exp(BASE_S[cov] + ageS + b_reg + b_pe * peCount * 0.15)));
    return { frequency: Math.round(freq * 1000) / 1000, severity: Math.round(sev), expected_annual_cost: Math.round(freq * sev * 100) / 100, source: "local-glm" };
  };

  const ipd = calc("ipd");
  const tf = TIER_F[inp.ipd_tier] || 1;
  const ipd_loaded = Math.round(ipd.expected_annual_cost * (1 + LOAD.ipd) * tf * 100) / 100;
  const ded_credit = Math.round(((TIERS[inp.ipd_tier]?.dedN || 0) * 0.10) * 100) / 100;
  const ipd_prem = Math.max(Math.round((ipd_loaded - ded_credit) * 100) / 100, 50);

  let total = ipd_prem;
  const riders = {};
  for (const [cov, inc] of [["opd", inp.include_opd], ["dental", inp.include_dental], ["maternity", inp.include_maternity]]) {
    if (!inc) continue;
    const r = calc(cov);
    const rp = Math.round(r.expected_annual_cost * (1 + LOAD[cov]) * 100) / 100;
    riders[cov] = { ...r, name: cov.toUpperCase() + " Rider", annual_premium: rp, monthly_premium: Math.round(rp / 12 * 100) / 100 };
    total += rp;
  }

  const ff = 1 + (inp.family_size - 1) * 0.65;
  total = Math.round(total * ff * 100) / 100;

  // ── Escalation product calculation ──────────────────────────────────────────
  let escalation_details = null;
  if (inp.include_escalation) {
    escalation_details = computeEscalation(total, inp.ipd_tier);
    total = escalation_details.escalated_annual_premium;
  }

  return {
    quote_id: `LOCAL-${Date.now()}`, model_version: "local-glm", ipd_tier: inp.ipd_tier,
    tier_benefits: TIERS[inp.ipd_tier],
    ipd_core: { ...ipd, annual_premium: ipd_prem, monthly_premium: Math.round(ipd_prem / 12 * 100) / 100, tier_factor: tf, deductible_credit: ded_credit, loading_pct: LOAD.ipd, source: "local-glm" },
    riders, family_size: inp.family_size, family_factor: Math.round(ff * 100) / 100,
    total_annual_premium: total, total_monthly_premium: Math.round(total / 12 * 100) / 100,
    escalation_details,
    risk_profile: { age: inp.age, gender: inp.gender, smoking: inp.smoking_status, exercise: inp.exercise_frequency, occupation: inp.occupation_type, preexist_conditions: inp.preexist_conditions },
  };
}

// ─── Price band (information-disclosure control) ────────────────────────────
// Shows a range rather than an exact figure. The midpoint is shifted by a
// small session-consistent jitter derived from the browser ID, so different
// sessions produce different ranges for the same profile — preventing a
// competitor from pinpointing the exact price with a single representative query.
function priceBand(annualPremium, browserId) {
  let h = 0;
  for (let i = 0; i < Math.min(browserId.length, 16); i++) {
    h = Math.imul(h ^ browserId.charCodeAt(i), 0x9e3779b9);
  }
  const jitter = ((Math.abs(h) % 601) - 300) / 10000; // ±3 %, deterministic per browser
  const mid = annualPremium * (1 + jitter);
  const B = 0.07; // ±7 % display band
  return {
    monthly: { low: Math.floor(mid * (1 - B) / 12), high: Math.ceil(mid * (1 + B) / 12) },
    annual: { low: Math.floor(mid * (1 - B)), high: Math.ceil(mid * (1 + B)) },
  };
}

// ─── Underwriting / Rule Engine ─────────────────────────────────────────────
function checkUnderwriting(inp) {
  const flags = [];
  const conditions = inp.preexist_conditions.filter(c => c !== "None");
  if (inp.age > 70) flags.push({ level: "decline", msg: "Age exceeds maximum insurable limit (70). Manual underwriting required before coverage can be offered." });
  if (inp.age < 18) flags.push({ level: "decline", msg: "Minimum insurable age is 18. Coverage is not available for this applicant." });
  if (conditions.includes("Cancer (remission)")) flags.push({ level: "warn", msg: "Cancer history detected: additional medical underwriting review required. Premium loading and exclusion clauses may apply." });
  if (conditions.includes("Kidney Disease")) flags.push({ level: "warn", msg: "Kidney disease: renal-related claims may be subject to an exclusion clause. Additional loading applies." });
  if (conditions.includes("Liver Disease")) flags.push({ level: "warn", msg: "Liver disease: hepatic-related claims may be subject to an exclusion clause. Additional loading applies." });
  if (conditions.length >= 3) flags.push({ level: "warn", msg: `${conditions.length} pre-existing conditions declared — medical underwriting referral is recommended before binding coverage.` });
  if (inp.smoking_status === "Current" && (conditions.includes("Heart Disease") || conditions.includes("Asthma/COPD"))) {
    flags.push({ level: "warn", msg: "High-risk combination: current smoker with cardiovascular or respiratory condition. Significant additional loading will apply." });
  }
  return flags;
}

// ─── Small components ───────────────────────────────────────────────────────
function Logo({ size = 48 }) {
  if (LOGO_URL) return <img src={LOGO_URL} alt="DAC" style={{ width: size, height: size, objectFit: "contain" }} />;
  return <div style={{ width: size, height: size, borderRadius: size * 0.22, background: "#0d2b7a", display: "flex", alignItems: "center", justifyContent: "center", color: "#f5a623", fontWeight: 600, fontSize: size * 0.35 }}>DAC</div>;
}

function Chev() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>;
}

function Ck({ s = 10 }) {
  return <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>;
}

function Spinner() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>;
}

// ─── Styles ─────────────────────────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=Instrument+Serif:ital@0;1&display=swap');
:root{--navy:#0d2b7a;--navy-l:#1a4fba;--gold:#f5a623;--gold-d:#c46800;--gold-bg:#fff7ed;--gold-bd:#fed7aa;--bg:#f7f8fa;--surf:#fff;--surf2:#f1f3f5;--surf3:#e2e5ea;--txt:#111827;--txt2:#4b5563;--txt3:#6b7280;--ok:#059669;--danger:#ef4444;--fd:'Instrument Serif',serif;--fb:'DM Sans',sans-serif;--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font-family:var(--fb);-webkit-font-smoothing:antialiased}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.app{min-height:100vh;display:flex;flex-direction:column}
.nav{background:var(--navy);padding:0 24px;height:52px;display:flex;align-items:center;justify-content:space-between}
.nav-brand{display:flex;align-items:center;gap:10px;cursor:pointer}
.nav-title{font-family:var(--fd);font-size:16px;color:white;font-style:italic}
.nav-right{display:flex;align-items:center;gap:10px}
.ctry-sel{display:flex;gap:2px;padding:2px;background:rgba(255,255,255,.08);border-radius:6px}
.ctry-btn{padding:4px 10px;border-radius:4px;border:none;cursor:pointer;font-size:11px;font-family:var(--fb);transition:all .15s;color:rgba(255,255,255,.5);background:transparent}
.ctry-btn.sel{background:rgba(255,255,255,.12);color:var(--gold);font-weight:600}
input[type="range"]{height:6px;border-radius:3px;outline:none;cursor:pointer}
.status{display:flex;align-items:center;gap:4px;font-size:10px;color:rgba(255,255,255,.4)}
.dot{width:6px;height:6px;border-radius:50%}.dot.ok{background:var(--ok)}.dot.off{background:var(--danger)}
.wizard{max-width:640px;width:100%;margin:0 auto;padding:28px 20px 40px;flex:1}
.steps{display:flex;align-items:center;margin-bottom:28px}
.step-item{display:flex;align-items:center;gap:6px;cursor:pointer}
.step-dot{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;transition:all .25s}
.step-dot.done{background:var(--navy);color:white}.step-dot.active{background:var(--gold);color:var(--navy)}.step-dot.pending{background:var(--surf2);color:var(--txt3);border:1.5px solid var(--surf3)}
.step-label{font-size:12px;font-weight:500}.step-label.done{color:var(--navy)}.step-label.active{color:var(--gold-d)}.step-label.pending{color:var(--txt3)}
.step-line{flex:1;height:2px;margin:0 8px;border-radius:1px}.step-line.done{background:var(--navy)}.step-line.pending{background:var(--surf3)}
.step-content{animation:fadeIn .3s ease both}
.step-title{font-size:22px;font-weight:500;margin-bottom:4px}
.step-sub{font-size:13px;color:var(--txt2);margin-bottom:22px}
.card{background:var(--surf);border-radius:var(--r);border:1px solid var(--surf3);padding:20px;margin-bottom:14px}
.card-label{font-size:11px;font-weight:600;color:var(--txt3);letter-spacing:.5px;text-transform:uppercase;margin-bottom:10px}
.fg{margin-bottom:14px}.fl{display:block;font-size:11px;font-weight:600;color:var(--txt2);margin-bottom:4px;letter-spacing:.3px}
.fi,.fs{width:100%;padding:9px 11px;border-radius:8px;border:1.5px solid var(--surf3);font-size:13px;font-family:var(--fb);color:var(--txt);background:white;outline:none;appearance:none;transition:border .15s}
.fi:focus,.fs:focus{border-color:var(--navy)}
.sw{position:relative}.sw svg{position:absolute;right:10px;top:50%;transform:translateY(-50%);pointer-events:none;color:var(--txt3)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{padding:5px 11px;border-radius:7px;font-size:11px;font-weight:500;border:1.5px solid var(--surf3);cursor:pointer;transition:all .12s;background:white;color:var(--txt2);font-family:var(--fb)}
.chip:hover{border-color:var(--navy-l)}.chip.sel{border-color:var(--navy);background:var(--navy);color:white}.chip.warn{border-color:#fca5a5;background:#fef2f2;color:#dc2626}
.tier-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tier-card{padding:14px 8px;border-radius:10px;border:1.5px solid var(--surf3);cursor:pointer;transition:all .2s;background:white;text-align:center;position:relative}
.tier-card:hover{border-color:var(--navy-l)}.tier-card.sel{border-color:var(--gold);background:var(--gold-bg)}
.tier-card .rec{position:absolute;top:-8px;left:50%;transform:translateX(-50%);padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;background:var(--gold);color:var(--navy);white-space:nowrap}
.tier-name{font-size:14px;font-weight:600}.tier-detail{font-size:10px;color:var(--txt3);margin-top:2px}
.tier-card.sel .tier-name{color:var(--gold-d)}.tier-card.sel .tier-detail{color:var(--gold-d)}
.rider-row{display:flex;align-items:center;padding:14px;border-radius:10px;border:1.5px solid var(--surf3);cursor:pointer;transition:all .15s;background:white;margin-bottom:6px;gap:12px}
.rider-row:hover{border-color:var(--navy-l)}.rider-row.on{border-color:var(--gold);background:var(--gold-bg)}
.rider-icon{width:38px;height:38px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.rider-check{width:18px;height:18px;border-radius:5px;border:2px solid var(--surf3);display:flex;align-items:center;justify-content:center;transition:all .15s;flex-shrink:0}
.rider-row.on .rider-check{border-color:var(--gold-d);background:var(--gold);color:var(--navy)}
.rider-info{flex:1}.rider-name{font-size:13px;font-weight:600}.rider-desc{font-size:11px;color:var(--txt3)}
.rider-price{text-align:right;font-size:12px;font-weight:600}.rider-price span{display:block;font-size:10px;font-weight:400;color:var(--txt3)}
.btn-row{display:flex;gap:8px;margin-top:20px}
.btn{flex:1;padding:13px;border-radius:8px;font-size:14px;font-weight:600;border:none;cursor:pointer;font-family:var(--fb);transition:all .15s;display:flex;align-items:center;justify-content:center;gap:6px}
.btn-back{background:var(--surf2);color:var(--txt2);border:1px solid var(--surf3)}.btn-back:hover{background:var(--surf3)}
.btn-next{background:var(--navy);color:white}.btn-next:hover{background:var(--navy-l)}
.btn-gold{background:var(--gold);color:var(--navy)}.btn-gold:hover{background:#eab735}
.btn:disabled{opacity:.5;cursor:not-allowed}
.res-hero{background:var(--navy);border-radius:14px;padding:28px;color:white;text-align:center;margin-bottom:20px;position:relative;overflow:hidden}
.res-hero::before{content:'';position:absolute;top:-30px;right:-20px;width:120px;height:120px;border-radius:50%;background:rgba(245,197,99,.08)}
.res-label{font-size:11px;opacity:.6;letter-spacing:.8px;text-transform:uppercase}
.res-amount{font-family:var(--fd);font-size:44px;font-weight:400;font-style:italic;color:var(--gold);margin:6px 0}
.res-monthly{font-size:13px;opacity:.5}
.res-tier{display:inline-flex;margin-top:8px;padding:3px 10px;border-radius:5px;background:rgba(245,197,99,.12);color:var(--gold);font-size:11px;font-weight:600}
.bk-section{margin-bottom:14px}
.bk-head{font-size:11px;font-weight:600;color:var(--txt3);letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;display:flex;align-items:center;gap:6px}
.bk-badge{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}
.bk-badge.core{background:var(--navy);color:white}.bk-badge.rider{background:var(--gold-bg);color:var(--gold-d)}
.bk-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--surf2);font-size:12px}
.bk-row:last-child{border-bottom:none}
.bk-l{color:var(--txt2)}.bk-v{font-weight:600}.bk-v.hi{color:var(--navy)}.bk-v.gold{color:var(--gold-d)}
.qid{margin-top:12px;padding-top:12px;border-top:1px solid var(--surf3);display:flex;justify-content:space-between;font-size:10px}
.qid span:first-child{color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.qid code{color:var(--txt2);background:var(--surf2);padding:2px 6px;border-radius:4px;font-size:10px}
.ai-bar{background:var(--surf);border-radius:var(--r);border:1px solid var(--gold-bd);padding:12px 14px;margin-top:14px;display:flex;align-items:flex-start;gap:10px}
.ai-dot{width:22px;height:22px;border-radius:50%;background:var(--navy);color:var(--gold);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex-shrink:0;margin-top:1px}
.ai-text{font-size:12px;color:var(--txt2);line-height:1.55;flex:1}
.ai-fab{position:fixed;bottom:20px;right:20px;width:48px;height:48px;border-radius:50%;background:var(--navy);color:var(--gold);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:100;transition:all .2s}
.ai-fab:hover{transform:scale(1.08)}
.ai-panel{position:fixed;bottom:78px;right:20px;width:360px;max-height:480px;background:var(--surf);border-radius:14px;border:1px solid var(--surf3);box-shadow:0 12px 40px rgba(0,0,0,.12);z-index:100;display:flex;flex-direction:column;animation:fadeIn .2s ease both;overflow:hidden}
.ai-head{padding:12px 14px;border-bottom:1px solid var(--surf3);display:flex;align-items:center;justify-content:space-between}
.ai-msgs{flex:1;overflow-y:auto;padding:10px 14px;display:flex;flex-direction:column;gap:8px;min-height:180px;max-height:320px}
.ai-msg{max-width:86%;padding:9px 12px;border-radius:10px;font-size:12px;line-height:1.5;word-wrap:break-word}
.ai-msg.bot{align-self:flex-start;background:var(--surf2);border-bottom-left-radius:3px}
.ai-msg.user{align-self:flex-end;background:var(--navy);color:white;border-bottom-right-radius:3px}
.ai-msg.typing{align-self:flex-start;background:var(--surf2);color:var(--txt3);font-style:italic}
.ai-qchips{display:flex;flex-wrap:wrap;gap:3px;margin-top:4px}
.ai-qchip{padding:3px 8px;border-radius:5px;font-size:10px;border:1px solid var(--surf3);cursor:pointer;background:white;color:var(--txt2);font-family:var(--fb)}
.ai-qchip:hover{border-color:var(--gold);color:var(--gold-d)}
.ai-input-row{padding:8px 10px;border-top:1px solid var(--surf3);display:flex;gap:6px}
.ai-input{flex:1;padding:7px 10px;border-radius:7px;border:1.5px solid var(--surf3);font-size:12px;font-family:var(--fb);outline:none}
.ai-input:focus{border-color:var(--navy)}
.ai-send{padding:7px 12px;border-radius:7px;background:var(--navy);color:var(--gold);border:none;cursor:pointer;font-size:11px;font-weight:600}
.ai-send:disabled{opacity:.4}
.footer{padding:20px;text-align:center;font-size:11px;color:var(--txt3);border-top:1px solid var(--surf3)}
.footer-tags{display:flex;gap:4px;justify-content:center;margin-top:4px}
.footer-tag{padding:2px 7px;border-radius:4px;font-size:10px;background:var(--surf2);color:var(--txt3)}
@media(max-width:640px){.row2,.row3{grid-template-columns:1fr}.tier-grid{grid-template-columns:1fr 1fr}.wizard{padding:20px 14px}.ai-panel{width:calc(100vw - 32px);right:16px;max-width:360px}.res-amount{font-size:34px}.step-label{display:none}}
@media(max-width:360px){.tier-grid{grid-template-columns:1fr}}
`;

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function PricingWizard() {
  const sessionTokenRef = useRef(null);
  const [step, setStep] = useState(0);
  const [apiOk, setApiOk] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isLocal, setIsLocal] = useState(false);
  const [aiTip, setAiTip] = useState("");
  const [prevQuote, setPrevQuote] = useState(null);
  // Lead capture — email gate
  const [leadEmail, setLeadEmail] = useState(() => { try { return localStorage.getItem("dac_lead_email") || ""; } catch { return ""; } });
  const [emailInput, setEmailInput] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(() => { try { return !!localStorage.getItem("dac_lead_email"); } catch { return false; } });

  // Load previous quote from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("dac_hp_last_quote");
      if (saved) setPrevQuote(JSON.parse(saved));
    } catch { }
  }, []);

  const [inp, setInp] = useState({
    age: 35, gender: "Male", country: "cambodia", region: "Phnom Penh",
    marital_status: "Single",
    smoking_status: "Never",
    exercise_days: 3, exercise_mins: 30,
    exercise_frequency: "Moderate",
    occupation_type: "Office/Desk",
    preexist_conditions: ["None"],
    bmi_height: 170, bmi_weight: 70,
    alcohol: "Never",
    prev_hospitalizations: 0,
    medications_count: 0,
    family_history: ["None"],
    diet: "Balanced",
    sleep_quality: "Good (7-9h)",
    stress_level: "Low",
    motorbike_daily: "No",
    water_access: "Piped/Safe",
    healthcare_proximity: "<5km",
    ipd_tier: "Silver", family_size: 1,
    include_opd: false, include_dental: false, include_maternity: false,
    include_escalation: false,
  });

  // Derive exercise_frequency label from days*mins for backend compatibility
  const deriveExercise = (days, mins) => {
    const w = days * mins;
    if (w <= 0) return "Sedentary";
    if (w < 60) return "Light";
    if (w < 150) return "Moderate";
    return "Active";
  };

  useEffect(() => {
    apiCall("/health").then(() => setApiOk(true)).catch(() => setApiOk(false));
  }, []);

  const u = (k, v) => setInp(p => {
    const next = { ...p, [k]: v };
    // Auto-derive exercise_frequency when exercise inputs change
    if (k === "exercise_days" || k === "exercise_mins") {
      const days = k === "exercise_days" ? v : next.exercise_days;
      const mins = k === "exercise_mins" ? v : next.exercise_mins;
      next.exercise_frequency = deriveExercise(days, mins);
    }
    return next;
  });

  const togglePE = (cond) => setInp(p => {
    const cur = p.preexist_conditions;
    if (cond === "None") return { ...p, preexist_conditions: ["None"] };
    const without = cur.filter(x => x !== "None" && x !== cond);
    if (cur.includes(cond)) return { ...p, preexist_conditions: without.length ? without : ["None"] };
    return { ...p, preexist_conditions: [...without, cond] };
  });

  const calculate = useCallback(async (overrideInp) => {
    const target = overrideInp || inp;
    setLoading(true); setResult(null); setIsLocal(false);
    let res;
    try {
      const session = await apiCall("/api/v2/session", { email: "", browser_id: getBrowserId() });
      sessionTokenRef.current = session.token;
      res = await apiCall("/api/v2/price", { ...target, browser_id: getBrowserId() }, { "X-Session-Token": session.token });
      if (target.include_escalation && !res.escalation_details) {
        const esc = computeEscalation(res.total_annual_premium, target.ipd_tier);
        res = { ...res, escalation_details: esc, total_annual_premium: esc.escalated_annual_premium, total_monthly_premium: esc.escalated_monthly_premium };
      }
      setResult(res);
    } catch {
      res = localPrice(target); setResult(res); setIsLocal(true);
    } finally {
      setLoading(false); setStep(3);
      // Save to localStorage for Renewal Advisor
      if (res) {
        try {
          localStorage.setItem("dac_hp_last_quote", JSON.stringify({
            date: new Date().toISOString(),
            input: target,
            premium: res.total_annual_premium,
            monthly: res.total_monthly_premium,
            tier: res.ipd_tier,
            riders: Object.keys(res.riders || {}),
            frequency: res.ipd_core?.frequency,
            severity: res.ipd_core?.severity,
          }));
        } catch { }
      }
    }
    return res;
  }, [inp]);

  const peCount = inp.preexist_conditions.filter(p => p !== "None").length;
  const uwFlags = checkUnderwriting(inp);
  const hasDecline = uwFlags.some(f => f.level === "decline");

  const estRider = (cov) => {
    const af = 1 + Math.max(0, (inp.age - 35)) * 0.008;
    const sf = { Never: 1, Former: 1.15, Current: 1.40 }[inp.smoking_status] || 1;
    const wm = (inp.exercise_days || 0) * (inp.exercise_mins || 0);
    const ef = wm <= 0 ? 1.20 : wm < 60 ? 1.10 : wm < 150 ? 0.95 : wm < 300 ? 0.85 : 0.75;
    const freq = FB_FREQ[cov] * af * sf * ef * (1 + peCount * 0.20);
    const sev = FB_SEV[cov] * (1 + Math.max(0, (inp.age - 30)) * 0.006);
    return Math.round(freq * sev * (1 + LOAD[cov]));
  };

  // AI tip on plan step
  const getAiRecommendedTier = () => {
    if (peCount >= 3 || (inp.smoking_status === "Current" && inp.age > 50)) return "Platinum";
    if (peCount >= 2 || inp.smoking_status === "Current" || inp.age > 55) return "Gold";
    if (peCount >= 1 || inp.age > 40) return "Silver";
    return "Silver"; // default safe choice
  };
  const aiTier = step === 3 ? getAiRecommendedTier() : null;

  useEffect(() => {
    if (step !== 3) { setAiTip(""); return; }
    const rec = getAiRecommendedTier();
    if (rec === "Platinum") {
      setAiTip(`High-risk profile detected (${peCount} conditions${inp.smoking_status === "Current" ? ", smoker" : ""}, age ${inp.age}). <strong>Platinum</strong> gives maximum protection with $150K limit and $0 deductible.`);
    } else if (rec === "Gold" && inp.ipd_tier !== "Gold" && inp.ipd_tier !== "Platinum") {
      setAiTip(`With your risk profile, <strong>Gold tier</strong> is recommended — $40K surgery limit and only $100 deductible provide strong coverage.`);
    } else if (rec === "Silver" && peCount === 0 && inp.age < 40) {
      setAiTip("<strong>Silver tier</strong> is solid for your low-risk profile. Bronze saves more but has a $500 deductible.");
    } else {
      setAiTip("");
    }
  }, [step, inp.ipd_tier, inp.smoking_status, inp.age, peCount]);

  const [showAbout, setShowAbout] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);
  const [adminKey, setAdminKey] = useState("");
  const [adminAuthed, setAdminAuthed] = useState(false);
  const [contactForm, setContactForm] = useState({ name: "", email: "", phone: "", message: "" });
  const [contactSubmitted, setContactSubmitted] = useState(false);

  // Rider config for rendering
  const RIDER_CFG = [
    { key: "include_opd", name: "OPD visits", desc: "Consultations, lab tests, procedures", icon: "#3b82f6", bg: "#eff6ff", cov: "opd" },
    { key: "include_dental", name: "Dental", desc: "Cleanings, fillings, extractions", icon: "#0d9488", bg: "#e1f5ee", cov: "dental" },
    { key: "include_maternity", name: "Maternity", desc: "Prenatal, delivery, newborn (10-mo wait)", icon: "#be185d", bg: "#fce7f3", cov: "maternity" },
  ];

  return (
    <>
      <style>{CSS}</style>
      <div className="app">

        {/* ABOUT PAGE */}
        {showAbout ? (
          <div className="wizard" style={{ animation: "fadeIn .3s ease both", maxWidth: 680 }}>
            {/* Hero */}
            <div style={{ textAlign: "center", marginBottom: 32, paddingTop: 8 }}>
              <Logo size={120} />
              <div style={{ fontSize: 26, fontWeight: 500, marginTop: 14, letterSpacing: -0.3 }}>About DAC HealthPrice</div>
              <div style={{ fontSize: 14, color: "var(--txt2)", marginTop: 6, lineHeight: 1.6, maxWidth: 440, margin: "6px auto 0" }}>
                AI-powered hospital reimbursement insurance pricing for Cambodia
              </div>
            </div>

            {/* Mission */}
            <div className="card" style={{ borderLeft: "3px solid var(--gold)", borderRadius: 0 }}>
              <p style={{ fontSize: 14, color: "var(--txt2)", lineHeight: 1.75, margin: 0, paddingLeft: 4 }}>
                DAC HealthPrice brings transparent, data-driven pricing to hospital reimbursement insurance in Cambodia. Our platform uses actuarial modeling to deliver fair, personalized premiums based on individual risk profiles — not guesswork.
              </p>
            </div>

            {/* How it works */}
            <div className="card">
              <div className="card-label">How it works</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  { num: "1", title: "Tell us about yourself", desc: "Age, gender, region, and family size" },
                  { num: "2", title: "Share your health profile", desc: "Smoking, exercise, occupation, conditions" },
                  { num: "3", title: "Choose your plan", desc: "IPD tier + optional OPD, Dental, Maternity" },
                  { num: "4", title: "Get your quote", desc: "Personalized premium with full breakdown" },
                ].map(s => (
                  <div key={s.num} style={{ display: "flex", gap: 12, alignItems: "flex-start", padding: 14, background: "var(--surf2)", borderRadius: 10 }}>
                    <div style={{ width: 32, height: 32, borderRadius: "50%", background: "var(--navy)", color: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 600, flexShrink: 0 }}>{s.num}</div>
                    <div style={{ paddingTop: 2 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 3 }}>{s.title}</div>
                      <div style={{ fontSize: 11, color: "var(--txt3)", lineHeight: 1.45 }}>{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Technology */}
            <div className="card">
              <div className="card-label">Our technology</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[
                  { icon: "M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z", title: "Frequency-severity model", desc: "Poisson for claim frequency, gradient boosting for severity" },
                  { icon: "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5", title: "8 ML models", desc: "Separate freq + sev for IPD, OPD, Dental, Maternity" },
                  { icon: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z", title: "AI advisor", desc: "Real-time plan recommendations and explanations" },
                  { icon: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z", title: "Secure platform", desc: "API auth, rate limiting, champion/challenger validation" },
                ].map(t => (
                  <div key={t.title} style={{ padding: 14, background: "var(--surf2)", borderRadius: 10, display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <div style={{ width: 32, height: 32, borderRadius: 8, background: "var(--navy)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--gold)" strokeWidth="2"><path d={t.icon} /></svg>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 3 }}>{t.title}</div>
                      <div style={{ fontSize: 11, color: "var(--txt3)", lineHeight: 1.45 }}>{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Coverage */}
            <div className="card">
              <div className="card-label">Coverage options</div>
              <div style={{ display: "flex", gap: 14, alignItems: "flex-start", padding: 16, background: "var(--navy)", borderRadius: 10, marginBottom: 12 }}>
                <div style={{ width: 40, height: 40, borderRadius: 8, background: "rgba(245,197,99,.15)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--gold)" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                </div>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: "white" }}>IPD hospital reimbursement</span>
                    <span style={{ padding: "2px 7px", borderRadius: 4, background: "var(--gold)", color: "var(--navy)", fontSize: 9, fontWeight: 700 }}>CORE</span>
                  </div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {["Bronze $15K", "Silver $40K", "Gold $80K", "Platinum $150K"].map(t => (
                      <span key={t} style={{ padding: "3px 8px", borderRadius: 4, background: "rgba(255,255,255,.1)", color: "rgba(255,255,255,.7)", fontSize: 10 }}>{t}</span>
                    ))}
                  </div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {[
                  { name: "OPD", desc: "Consultations, labs, procedures", color: "#3b82f6", bg: "#eff6ff" },
                  { name: "Dental", desc: "Cleanings, fillings, extractions", color: "#0d9488", bg: "#e1f5ee" },
                  { name: "Maternity", desc: "Prenatal, delivery, newborn", color: "#be185d", bg: "#fce7f3" },
                ].map(r => (
                  <div key={r.name} style={{ padding: 14, borderRadius: 10, border: "1px solid var(--surf3)", textAlign: "center" }}>
                    <div style={{ width: 32, height: 32, borderRadius: 8, background: r.bg, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 8px" }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={r.color} strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                    </div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{r.name} rider</div>
                    <div style={{ fontSize: 10, color: "var(--txt3)", lineHeight: 1.4 }}>{r.desc}</div>
                    <div style={{ marginTop: 6, padding: "2px 7px", borderRadius: 4, background: "var(--gold-bg)", color: "var(--gold-d)", fontSize: 9, fontWeight: 600, display: "inline-block" }}>ADD-ON</div>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA + Contact Form */}
            <div className="card" style={{ background: "var(--navy)", color: "white", overflow: "hidden", padding: 0 }}>
              {/* Top CTA */}
              <div style={{ textAlign: "center", padding: "28px 28px 20px" }}>
                <div style={{ fontSize: 18, fontWeight: 500, marginBottom: 4 }}>Ready to get your quote?</div>
                <div style={{ fontSize: 12, opacity: 0.5, marginBottom: 16 }}>It takes less than 2 minutes</div>
                <button className="btn btn-gold" style={{ maxWidth: 280, margin: "0 auto" }} onClick={() => { setShowAbout(false); setStep(0); }}>
                  Start pricing
                </button>
              </div>

              {/* Divider */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 28px" }}>
                <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,.1)" }} />
                <span style={{ fontSize: 11, opacity: 0.4 }}>or get in touch</span>
                <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,.1)" }} />
              </div>

              {/* Contact Form */}
              <div style={{ padding: "20px 28px 28px" }}>
                <div style={{ fontSize: 15, fontWeight: 500, color: "var(--gold)", marginBottom: 4 }}>Medical insurance inquiry</div>
                <div style={{ fontSize: 12, opacity: 0.5, marginBottom: 18 }}>Fill in the form below and our team will get back to you</div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                  <div>
                    <label style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,.6)", display: "block", marginBottom: 4 }}>Full name <span style={{ color: "var(--gold)" }}>*</span></label>
                    <input
                      value={contactForm.name} onChange={e => setContactForm(p => ({ ...p, name: e.target.value }))}
                      placeholder="Your name"
                      style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "1.5px solid rgba(255,255,255,.12)", background: "rgba(255,255,255,.06)", color: "white", fontSize: 13, fontFamily: "var(--fb)", outline: "none" }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,.6)", display: "block", marginBottom: 4 }}>Email</label>
                    <input
                      value={contactForm.email} onChange={e => setContactForm(p => ({ ...p, email: e.target.value }))}
                      placeholder="your@email.com" type="email"
                      style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "1.5px solid rgba(255,255,255,.12)", background: "rgba(255,255,255,.06)", color: "white", fontSize: 13, fontFamily: "var(--fb)", outline: "none" }}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,.6)", display: "block", marginBottom: 4 }}>Phone number <span style={{ color: "var(--gold)" }}>*</span></label>
                  <div style={{ display: "flex", gap: 6 }}>
                    <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--gold)", color: "var(--navy)", fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
                      +855 <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>
                    </div>
                    <input
                      value={contactForm.phone} onChange={e => setContactForm(p => ({ ...p, phone: e.target.value.replace(/[^0-9]/g, "") }))}
                      placeholder="12 345 678" type="tel"
                      style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: "1.5px solid rgba(255,255,255,.12)", background: "rgba(255,255,255,.06)", color: "white", fontSize: 13, fontFamily: "var(--fb)", outline: "none" }}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: 14 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,.6)", display: "block", marginBottom: 4 }}>Message <span style={{ color: "var(--gold)" }}>*</span></label>
                  <textarea
                    value={contactForm.message} onChange={e => setContactForm(p => ({ ...p, message: e.target.value }))}
                    placeholder="Tell us about your insurance needs..."
                    rows={4}
                    style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "1.5px solid rgba(255,255,255,.12)", background: "rgba(255,255,255,.06)", color: "white", fontSize: 13, fontFamily: "var(--fb)", outline: "none", resize: "vertical" }}
                  />
                </div>

                <button
                  onClick={() => {
                    if (!contactForm.name || !contactForm.phone || !contactForm.message) {
                      alert("Please fill in all required fields");
                      return;
                    }
                    setContactSubmitted(true);
                    setTimeout(() => setContactSubmitted(false), 5000);
                    setContactForm({ name: "", email: "", phone: "", message: "" });
                  }}
                  style={{
                    width: "100%", padding: 13, borderRadius: 8, border: "none", cursor: "pointer",
                    background: contactSubmitted ? "var(--ok)" : "var(--gold)", color: contactSubmitted ? "white" : "var(--navy)",
                    fontSize: 14, fontWeight: 600, fontFamily: "var(--fb)", transition: "all .2s",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                  }}
                >
                  {contactSubmitted ? <><Ck s={14} /> Message sent!</> : "Submit"}
                </button>

                <div style={{ textAlign: "center", marginTop: 10, fontSize: 10, opacity: 0.35 }}>
                  Your information will be kept confidential and used only for insurance consultation purposes.
                </div>
              </div>
            </div>
          </div>
        ) : showAdmin ? (

          /* ADMIN PAGE */
          <div className="wizard" style={{ animation: "fadeIn .3s ease both", maxWidth: 680 }}>
            {!adminAuthed ? (
              /* Login gate */
              <div style={{ maxWidth: 360, margin: "60px auto", textAlign: "center" }}>
                <div style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--navy)", color: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px", fontSize: 18 }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>
                </div>
                <div style={{ fontSize: 20, fontWeight: 500, marginBottom: 4 }}>Admin access</div>
                <div style={{ fontSize: 13, color: "var(--txt3)", marginBottom: 20 }}>Enter your API key to manage models and data</div>
                <input
                  type="password" placeholder="Enter admin API key"
                  value={adminKey} onChange={e => setAdminKey(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") setAdminAuthed(true); }}
                  style={{ width: "100%", padding: "11px 14px", borderRadius: 8, border: "1.5px solid var(--surf3)", fontSize: 13, fontFamily: "var(--fb)", outline: "none", marginBottom: 10, textAlign: "center" }}
                />
                <button className="btn btn-next" onClick={() => { if (adminKey.trim()) setAdminAuthed(true); }}>
                  Unlock
                </button>
              </div>
            ) : (
              /* Admin dashboard */
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <div>
                    <div className="step-title">Admin dashboard</div>
                    <div className="step-sub" style={{ marginBottom: 0 }}>Upload data, manage models, trigger retraining</div>
                  </div>
                  <button onClick={() => { setAdminAuthed(false); setAdminKey(""); }} style={{ background: "var(--surf2)", border: "1px solid var(--surf3)", borderRadius: 6, padding: "5px 12px", fontSize: 11, cursor: "pointer", color: "var(--txt3)", fontFamily: "var(--fb)" }}>Lock</button>
                </div>

                {/* Model status */}
                <div className="card">
                  <div className="card-label">System status</div>
                  <AdminStatus apiOk={apiOk} adminKey={adminKey} />
                </div>

                {/* Upload dataset */}
                <div className="card">
                  <div className="card-label">Upload training data</div>
                  <AdminUpload adminKey={adminKey} />
                </div>

                {/* Upload history */}
                <div className="card">
                  <div className="card-label">Upload history</div>
                  <AdminHistory adminKey={adminKey} />
                </div>

                {/* User behavior data */}
                <div className="card">
                  <div className="card-label">User quote data</div>
                  <AdminUserData adminKey={adminKey} />
                </div>
              </>
            )}
          </div>

        ) : (

          <div className="wizard">
            {/* PROGRESS BAR */}
            <div className="steps">
              {STEPS.map((s, i) => (
                <React.Fragment key={i}>
                  <div className="step-item" onClick={() => { if (i <= step || (i === 3 && result)) setStep(i); }}>
                    <div className={`step-dot ${i < step || (i === 3 && result) ? "done" : i === step ? "active" : "pending"}`}>
                      {i < step ? <Ck s={12} /> : i + 1}
                    </div>
                    <span className={`step-label ${i < step ? "done" : i === step ? "active" : "pending"}`}>{s}</span>
                  </div>
                  {i < 3 && <div className={`step-line ${i < step ? "done" : "pending"}`} />}
                </React.Fragment>
              ))}
            </div>

            {/* STEP 1: PROFILE */}
            {step === 0 && (
              <div className="step-content">
                {/* Renewal Advisor — Welcome Back */}
                {prevQuote && (() => {
                  const days = Math.floor((Date.now() - new Date(prevQuote.date).getTime()) / 86400000);
                  const ago = days === 0 ? "today" : days === 1 ? "yesterday" : days < 30 ? `${days} days ago` : days < 365 ? `${Math.floor(days / 30)} months ago` : `${Math.floor(days / 365)} years ago`;
                  return (
                    <div style={{ background: "linear-gradient(135deg, var(--navy) 0%, #16213e 100%)", borderRadius: 12, padding: 20, marginBottom: 20, position: "relative", overflow: "hidden" }}>
                      <div style={{ position: "absolute", top: -20, right: -10, width: 80, height: 80, borderRadius: "50%", background: "rgba(245,197,99,0.08)" }} />
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: "var(--navy)" }}>AI</span>
                        </div>
                        <span style={{ color: "var(--gold)", fontSize: 12, fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>Renewal Advisor</span>
                      </div>
                      <p style={{ color: "white", fontSize: 15, fontWeight: 500, margin: "0 0 6px" }}>Welcome back!</p>
                      <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 13, lineHeight: 1.6, margin: 0 }}>
                        Your last quote was <strong style={{ color: "var(--gold)" }}>{ago}</strong> — <strong style={{ color: "white" }}>${prevQuote.premium?.toLocaleString()}/year</strong> on {prevQuote.tier} tier.
                        {prevQuote.riders?.length > 0 && ` With ${prevQuote.riders.join(", ")} rider${prevQuote.riders.length > 1 ? "s" : ""}.`}
                        {" "}Update your details below to see how your premium has changed.
                      </p>
                      <button onClick={() => {
                        if (prevQuote.input) {
                          setInp(p => ({ ...p, ...prevQuote.input }));
                        }
                      }} style={{ marginTop: 10, background: "rgba(245,197,99,0.15)", border: "1px solid rgba(245,197,99,0.3)", color: "var(--gold)", padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "var(--fb)" }}>
                        Load previous profile
                      </button>
                    </div>
                  );
                })()}
                <div className="step-title">Tell us about yourself</div>
                <div className="step-sub">Basic demographics for your insurance quote</div>
                <div className="card">
                  <div className="row2">
                    <div className="fg">
                      <label className="fl">Age</label>
                      <input className="fi" type="number" min="0" max="100"
                        inputMode="numeric" pattern="[0-9]*"
                        onKeyDown={e => { if (["e", "E", "+", "-", "."].includes(e.key)) e.preventDefault(); }}
                        value={inp.age === 0 ? "" : inp.age}
                        onChange={e => {
                          const raw = e.target.value;
                          if (raw === "") { u("age", 0); return; }
                          const n = parseInt(raw);
                          if (!isNaN(n)) u("age", Math.min(100, Math.max(0, n)));
                        }}
                        onBlur={() => { if (!inp.age && inp.age !== 0) u("age", 18); }}
                      />
                    </div>
                    <div className="fg">
                      <label className="fl">Gender</label>
                      <div className="sw">
                        <select className="fs" value={inp.gender} onChange={e => u("gender", e.target.value)}>{GENDERS.map(g => <option key={g}>{g}</option>)}</select>
                        <Chev />
                      </div>
                    </div>
                  </div>
                  <div className="row2">
                    <div className="fg">
                      <label className="fl">Region</label>
                      <div className="sw">
                        <select className="fs" value={inp.region} onChange={e => u("region", e.target.value)}>{REGIONS.map(r => <option key={r}>{r}</option>)}</select>
                        <Chev />
                      </div>
                    </div>
                    <div className="fg">
                      <label className="fl">Family size</label>
                      <input className="fi" type="number" min="1" max="10"
                        value={inp.family_size === 0 ? "" : inp.family_size}
                        onChange={e => {
                          const raw = e.target.value;
                          if (raw === "") { u("family_size", 0); return; }
                          const n = parseInt(raw);
                          if (!isNaN(n)) u("family_size", Math.min(10, Math.max(0, n)));
                        }}
                        onBlur={() => { if (!inp.family_size || inp.family_size < 1) u("family_size", 1); }}
                      />
                    </div>
                  </div>
                </div>
                <div className="btn-row">
                  <button className="btn btn-next" onClick={() => setStep(1)}>Continue</button>
                </div>
              </div>
            )}

            {/* STEP 2: HEALTH */}
            {step === 1 && (
              <div className="step-content">
                <div className="step-title">Your health profile</div>
                <div className="step-sub">Lifestyle factors that affect your premium</div>
                <div className="card">
                  <div className="row2">
                    <div className="fg">
                      <label className="fl">Smoking status</label>
                      <div className="sw">
                        <select className="fs" value={inp.smoking_status} onChange={e => u("smoking_status", e.target.value)}>{SMOKING.map(s => <option key={s}>{s}</option>)}</select>
                        <Chev />
                      </div>
                    </div>
                    <div className="fg">
                      <label className="fl">Occupation</label>
                      <div className="sw">
                        <select className="fs" value={inp.occupation_type} onChange={e => u("occupation_type", e.target.value)}>{OCCUPATIONS.map(s => <option key={s}>{s}</option>)}</select>
                        <Chev />
                      </div>
                    </div>
                  </div>

                  {/* Quantified exercise */}
                  <div style={{ background: "var(--surf2)", borderRadius: "var(--r)", padding: 16, marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <label className="fl" style={{ margin: 0 }}>Physical exercise</label>
                      <span style={{
                        padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600,
                        background: inp.exercise_frequency === "Sedentary" ? "#fef2f2" : inp.exercise_frequency === "Light" ? "#fffbeb" : inp.exercise_frequency === "Moderate" ? "#eff6ff" : "#e1f5ee",
                        color: inp.exercise_frequency === "Sedentary" ? "#dc2626" : inp.exercise_frequency === "Light" ? "#b07a0a" : inp.exercise_frequency === "Moderate" ? "#1e40af" : "#059669",
                      }}>{inp.exercise_frequency} — {(inp.exercise_days || 0) * (inp.exercise_mins || 0)} min/week</span>
                    </div>
                    <div className="row2">
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--txt2)", marginBottom: 4 }}>
                          <span>Days per week</span>
                          <strong style={{ color: "var(--txt)" }}>{inp.exercise_days}</strong>
                        </div>
                        <input type="range" min="0" max="7" step="1" value={inp.exercise_days}
                          onChange={e => u("exercise_days", parseInt(e.target.value))}
                          style={{ width: "100%", accentColor: "var(--navy)", cursor: "pointer" }} />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--txt3)", marginTop: 2 }}>
                          <span>0</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6</span><span>7</span>
                        </div>
                      </div>
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--txt2)", marginBottom: 4 }}>
                          <span>Minutes per session</span>
                          <strong style={{ color: "var(--txt)" }}>{inp.exercise_mins}</strong>
                        </div>
                        <input type="range" min="0" max="120" step="5" value={inp.exercise_mins}
                          onChange={e => u("exercise_mins", parseInt(e.target.value))}
                          style={{ width: "100%", accentColor: "var(--navy)", cursor: "pointer" }} />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--txt3)", marginTop: 2 }}>
                          <span>0</span><span>30</span><span>60</span><span>90</span><span>120</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="fg">
                    <label className="fl">Pre-existing conditions {peCount > 0 && <span style={{ color: "var(--danger)", fontWeight: 400 }}>({peCount})</span>}</label>
                    <div className="chips">
                      {PREEXIST.map(p => (
                        <div key={p} className={`chip ${inp.preexist_conditions.includes(p) ? (p === "None" ? "sel" : "warn") : ""}`} onClick={() => togglePE(p)}>{p}</div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="btn-row">
                  <button className="btn btn-back" onClick={() => setStep(0)}>Back</button>
                  <button className="btn btn-next" onClick={() => setStep(2)}>Continue</button>
                </div>
              </div>
            )}

            {/* STEP 3: PLAN */}
            {step === 2 && (
              <div className="step-content">
                <div className="step-title">Choose your plan</div>
                <div className="step-sub">Select an IPD tier and optional riders</div>

                <div className="card">
                  <div className="card-label">IPD hospital reimbursement</div>
                  <div className="tier-grid">
                    {Object.entries(TIERS).map(([k, v]) => (
                      <div key={k} className={`tier-card ${inp.ipd_tier === k ? "sel" : ""}`} onClick={() => u("ipd_tier", k)}>
                        {aiTier === k && <div className="rec">AI recommended</div>}
                        <div className="tier-name">{k}</div>
                        <div className="tier-detail">{v.limit} limit</div>
                        <div className="tier-detail">{v.room}</div>
                        <div className="tier-detail">Ded: {v.ded}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="card">
                  <div className="card-label">Optional riders</div>
                  {RIDER_CFG.map(r => {
                    const est = estRider(r.cov);
                    return (
                      <div key={r.key} className={`rider-row ${inp[r.key] ? "on" : ""}`} onClick={() => u(r.key, !inp[r.key])}>
                        <div className="rider-icon" style={{ background: r.bg }}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={r.icon} strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                        </div>
                        <div className="rider-info">
                          <div className="rider-name">{r.name}</div>
                          <div className="rider-desc">{r.desc}</div>
                        </div>
                        <div className="rider-price">+${Math.round(est / 12)}/mo<span>${est}/yr</span></div>
                        <div className="rider-check">{inp[r.key] && <Ck />}</div>
                      </div>
                    );
                  })}
                </div>

                {/* ── Escalation product toggle ── */}
                <div className="card" style={{ border: inp.include_escalation ? "1.5px solid var(--gold)" : "1.5px solid var(--surf3)", background: inp.include_escalation ? "var(--gold-bg)" : "white" }}>
                  <div className="card-label" style={{ marginBottom: 10 }}>
                    Protection upgrade
                    <span style={{ marginLeft: 8, padding: "2px 7px", borderRadius: 4, fontSize: 9, fontWeight: 700, background: "var(--gold)", color: "var(--navy)" }}>NEW</span>
                  </div>
                  <div
                    className={`rider-row ${inp.include_escalation ? "on" : ""}`}
                    style={{ marginBottom: 0, border: "none", padding: "10px 0", background: "transparent", cursor: "pointer" }}
                    onClick={() => u("include_escalation", !inp.include_escalation)}
                  >
                    <div className="rider-icon" style={{ background: "#f0fdf4" }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
                    </div>
                    <div className="rider-info">
                      <div className="rider-name">Auto-Escalation <span style={{ fontSize: 10, fontWeight: 400, color: "var(--txt3)" }}>+5%/year</span></div>
                      <div className="rider-desc">Coverage grows 5% every year. Same premium. No health re-check. Cap: 250% of base.</div>
                    </div>
                    <div className="rider-price" style={{ color: inp.include_escalation ? "var(--gold-d)" : "var(--txt3)" }}>
                      ~+10%<span>of premium</span>
                    </div>
                    <div className="rider-check">{inp.include_escalation && <Ck />}</div>
                  </div>
                  {inp.include_escalation && (
                    <div style={{ marginTop: 10, padding: "8px 10px", background: "#f0fdf4", borderRadius: 8, fontSize: 11, color: "#166534", lineHeight: 1.5, borderLeft: "3px solid #16a34a" }}>
                      Your coverage grows automatically: Year 1 → 100%, Year 5 → 122%, Year 10 → 155%, Year 20 → 250% of base. Same premium throughout. Loyalty bonuses at Year 10 and Year 20.
                    </div>
                  )}
                </div>

                {aiTip && (
                  <div className="ai-bar">
                    <div className="ai-dot">AI</div>
                    <div className="ai-text" dangerouslySetInnerHTML={{ __html: aiTip }} />
                  </div>
                )}

                <div className="btn-row">
                  <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
                  <button className="btn btn-gold" onClick={calculate} disabled={loading}>
                    {loading ? <><Spinner /> Calculating...</> : "Get my quote"}
                  </button>
                </div>
              </div>
            )}

            {/* STEP 4: QUOTE */}
            {step === 3 && result && (
              <div className="step-content">

                {/* ── Why Choose DAC — value anchor before price ── */}
                <div className="card" style={{ marginBottom: 16, borderLeft: "3px solid var(--gold)" }}>
                  <div className="card-label" style={{ marginBottom: 12 }}>✦ Why customers choose DAC</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {[
                      { icon: "🏥", title: "Direct hospital billing", desc: "No upfront payment at partner hospitals" },
                      { icon: "👤", title: "Personalised underwriting", desc: "Premium built around your exact risk profile" },
                      { icon: "🌐", title: "Bilingual support", desc: "Khmer + English, 7 days a week" },
                      { icon: "🔧", title: "Flexible riders", desc: "Add OPD, Dental, Maternity anytime" },
                    ].map(f => (
                      <div key={f.title} style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "10px 12px", background: "var(--surf2)", borderRadius: 8 }}>
                        <span style={{ fontSize: 18 }}>{f.icon}</span>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{f.title}</div>
                          <div style={{ fontSize: 11, color: "var(--txt3)", lineHeight: 1.4 }}>{f.desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Premium display — indicative range ── */}
                {(() => {
                  const range = getDisplayRange(result.total_monthly_premium);
                  return (
                    <div className="res-hero">
                      <div className="res-label">Your indicative monthly premium</div>
                      <div className="res-amount" style={{ fontSize: 36 }}>{range.text}</div>
                      <div className="res-monthly" style={{ opacity: 0.55, fontSize: 11 }}>per month · indicative range · Family of {result.family_size}</div>
                      <div className="res-tier">
                        {result.ipd_tier} tier{Object.keys(result.riders || {}).length > 0 && ` + ${Object.keys(result.riders).join(", ")}`}
                      </div>
                      <div style={{ marginTop: 10, fontSize: 10, opacity: 0.45, fontStyle: "italic" }}>Exact premium confirmed by your DAC advisor</div>
                    </div>
                  );
                })()}

                {/* ── Escalation projection ── */}
                {result?.escalation_details?.enabled && (() => {
                  const esc = result.escalation_details;
                  const rows = [1, 2, 3, 5, 10, 15, 20].map(yr => esc.projections.find(p => p.year === yr)).filter(Boolean);
                  return (
                    <div className="card" style={{ marginBottom: 16, borderLeft: "3px solid #16a34a" }}>
                      <div className="card-label" style={{ marginBottom: 2, color: "#166534" }}>
                        Auto-Escalation — 20-year coverage growth
                      </div>
                      <div style={{ fontSize: 11, color: "var(--txt3)", marginBottom: 12 }}>
                        Your premium stays at <strong>${Math.round(esc.escalated_monthly_premium)}/mo</strong>. Coverage grows every year.
                        <span style={{ marginLeft: 6, padding: "1px 6px", borderRadius: 4, fontSize: 9, background: "#dcfce7", color: "#166534", fontWeight: 600 }}>
                          {esc.cost_factor_pct} escalation loading
                        </span>
                      </div>
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                          <thead>
                            <tr style={{ borderBottom: "1.5px solid var(--surf3)" }}>
                              {["Year", "Coverage", "vs. Base", "Bonus"].map(h => (
                                <th key={h} style={{ textAlign: h === "Year" ? "left" : "right", padding: "4px 6px", color: "var(--txt3)", fontWeight: 600, letterSpacing: ".3px" }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((p, i) => (
                              <tr key={p.year} style={{ borderBottom: i < rows.length - 1 ? "1px solid var(--surf2)" : "none", background: p.year === 10 || p.year === 20 ? "#f0fdf4" : "transparent" }}>
                                <td style={{ padding: "5px 6px", fontWeight: p.year === 20 ? 600 : 400 }}>Yr {p.year}</td>
                                <td style={{ padding: "5px 6px", textAlign: "right", fontWeight: 600 }}>${p.coverage_amount.toLocaleString()}</td>
                                <td style={{ padding: "5px 6px", textAlign: "right", color: p.cumulative_escalation_pct > 0 ? "#16a34a" : "var(--txt3)" }}>
                                  {p.cumulative_escalation_pct > 0 ? `+${p.cumulative_escalation_pct}%` : "—"}
                                </td>
                                <td style={{ padding: "5px 6px", textAlign: "right", color: "#16a34a", fontWeight: p.loyalty_bonus > 0 ? 600 : 400 }}>
                                  {p.loyalty_bonus > 0 ? `$${p.loyalty_bonus.toLocaleString()} 🎁` : "—"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div style={{ marginTop: 10, fontSize: 10, color: "var(--txt3)", fontStyle: "italic" }}>
                        Coverage capped at 250% of base (${ esc.terminal_coverage.toLocaleString()}). No health re-check required.
                      </div>
                    </div>
                  );
                })()}

                {/* ── Lead capture — email gate ── */}
                {!emailSubmitted ? (
                  <div className="card" style={{ background: "var(--navy)", color: "white", marginBottom: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <div style={{ width: 26, height: 26, borderRadius: "50%", background: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: "var(--navy)" }}>AI</span>
                      </div>
                      <span style={{ color: "var(--gold)", fontSize: 12, fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>Get your exact quote</span>
                    </div>
                    <p style={{ fontSize: 13, color: "rgba(255,255,255,0.75)", lineHeight: 1.6, marginBottom: 14 }}>
                      Enter your email and a DAC advisor will confirm your <strong style={{ color: "white" }}>exact personalised premium</strong> within 1 business day.
                    </p>
                    <div style={{ display: "flex", gap: 8 }}>
                      <input
                        type="email" placeholder="your@email.com"
                        value={emailInput} onChange={e => setEmailInput(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter" && emailInput.includes("@")) { localStorage.setItem("dac_lead_email", emailInput); setLeadEmail(emailInput); setEmailSubmitted(true); } }}
                        style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: "1.5px solid rgba(255,255,255,.15)", background: "rgba(255,255,255,.08)", color: "white", fontSize: 13, fontFamily: "var(--fb)", outline: "none" }}
                      />
                      <button
                        onClick={() => { if (!emailInput.includes("@")) return; localStorage.setItem("dac_lead_email", emailInput); setLeadEmail(emailInput); setEmailSubmitted(true); }}
                        style={{ padding: "10px 18px", borderRadius: 8, border: "none", background: "var(--gold)", color: "var(--navy)", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "var(--fb)", whiteSpace: "nowrap" }}
                      >Request quote →</button>
                    </div>
                    <div style={{ marginTop: 8, fontSize: 10, opacity: 0.35, textAlign: "center" }}>No spam. Used only for your insurance quote confirmation.</div>
                  </div>
                ) : (
                  <div className="card" style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", marginBottom: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 18 }}>✅</span>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: "#166534" }}>Advisor request sent!</div>
                        <div style={{ fontSize: 11, color: "#15803d" }}>We'll confirm your exact premium at <strong>{leadEmail}</strong> within 1 business day.</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Renewal Advisor — Comparison */}
                {prevQuote && result && (() => {
                  const diff = result.total_annual_premium - prevQuote.premium;
                  const pct = prevQuote.premium ? Math.round((diff / prevQuote.premium) * 100) : 0;
                  const changes = [];
                  if (prevQuote.input) {
                    const pi = prevQuote.input;
                    if (pi.age !== inp.age) changes.push({ factor: "Age", from: pi.age, to: inp.age, impact: inp.age > pi.age ? "increases" : "decreases" });
                    if (pi.smoking_status !== inp.smoking_status) changes.push({ factor: "Smoking", from: pi.smoking_status, to: inp.smoking_status, impact: ["Never", "Former", "Current"].indexOf(inp.smoking_status) > ["Never", "Former", "Current"].indexOf(pi.smoking_status) ? "increases" : "decreases" });
                    if (pi.exercise_frequency !== inp.exercise_frequency) changes.push({ factor: "Exercise", from: pi.exercise_frequency, to: inp.exercise_frequency });
                    if (pi.ipd_tier !== inp.ipd_tier) changes.push({ factor: "Tier", from: pi.ipd_tier, to: inp.ipd_tier });
                    const oldPE = (pi.preexist_conditions || []).filter(p => p !== "None").length;
                    const newPE = inp.preexist_conditions.filter(p => p !== "None").length;
                    if (oldPE !== newPE) changes.push({ factor: "Conditions", from: `${oldPE}`, to: `${newPE}`, impact: newPE > oldPE ? "increases" : "decreases" });
                  }
                  return (
                    <div style={{ background: "linear-gradient(135deg, var(--navy) 0%, #16213e 100%)", borderRadius: 12, padding: 20, marginBottom: 20, position: "relative", overflow: "hidden" }}>
                      <div style={{ position: "absolute", top: -20, right: -10, width: 80, height: 80, borderRadius: "50%", background: "rgba(245,197,99,0.08)" }} />
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: "var(--navy)" }}>AI</span>
                        </div>
                        <span style={{ color: "var(--gold)", fontSize: 12, fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>Renewal Advisor</span>
                      </div>
                      <p style={{ color: "white", fontSize: 15, fontWeight: 500, margin: "0 0 8px" }}>Compared to your last quote</p>
                      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                        <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", borderRadius: 8, padding: 12, textAlign: "center" }}>
                          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11, marginBottom: 4 }}>Previous</div>
                          <div style={{ color: "white", fontSize: 18, fontWeight: 600 }}>${prevQuote.premium?.toLocaleString()}</div>
                          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>{prevQuote.tier}/yr</div>
                        </div>
                        <div style={{ display: "flex", alignItems: "center" }}><span style={{ color: diff > 0 ? "#f87171" : diff < 0 ? "#34d399" : "var(--gold)", fontSize: 20 }}>→</span></div>
                        <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", borderRadius: 8, padding: 12, textAlign: "center" }}>
                          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11, marginBottom: 4 }}>Current</div>
                          <div style={{ color: "white", fontSize: 18, fontWeight: 600 }}>${result.total_annual_premium?.toLocaleString()}</div>
                          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>{result.ipd_tier}/yr</div>
                        </div>
                        <div style={{ flex: 1, background: diff > 0 ? "rgba(248,113,113,0.1)" : diff < 0 ? "rgba(52,211,153,0.1)" : "rgba(245,197,99,0.1)", borderRadius: 8, padding: 12, textAlign: "center", border: `1px solid ${diff > 0 ? "rgba(248,113,113,0.3)" : diff < 0 ? "rgba(52,211,153,0.3)" : "rgba(245,197,99,0.3)"}` }}>
                          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11, marginBottom: 4 }}>Change</div>
                          <div style={{ color: diff > 0 ? "#f87171" : diff < 0 ? "#34d399" : "var(--gold)", fontSize: 18, fontWeight: 600 }}>{diff > 0 ? "+" : ""}{diff === 0 ? "$0" : `$${Math.abs(Math.round(diff))}`}</div>
                          {diff !== 0 && <div style={{ color: diff > 0 ? "#f87171" : "#34d399", fontSize: 11 }}>{diff > 0 ? "+" : ""}{pct}%</div>}
                        </div>
                      </div>
                      {changes.length > 0 && (
                        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 10 }}>
                          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: 11, marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>What changed</p>
                          {changes.map((c, i) => (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                              <span style={{ color: c.impact === "increases" ? "#f87171" : c.impact === "decreases" ? "#34d399" : "var(--gold)", fontSize: 13 }}>{c.impact === "increases" ? "▲" : c.impact === "decreases" ? "▼" : "●"}</span>
                              <span style={{ color: "rgba(255,255,255,0.8)", fontSize: 13 }}>{c.factor}: <span style={{ color: "rgba(255,255,255,0.5)" }}>{c.from}</span> → <span style={{ color: "white", fontWeight: 500 }}>{c.to}</span></span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })()}

                <div className="card">
                  {/* IPD Core */}
                  <div className="bk-section">
                    <div className="bk-head"><span className="bk-badge core">IPD</span> Hospital reimbursement</div>
                    <div className="bk-row"><span className="bk-l">Frequency</span><span className="bk-v">{result.ipd_core?.frequency}/yr</span></div>
                    <div className="bk-row"><span className="bk-l">Severity</span><span className="bk-v">${result.ipd_core?.severity?.toLocaleString()}</span></div>
                    <div className="bk-row"><span className="bk-l">Expected cost</span><span className="bk-v">${result.ipd_core?.expected_annual_cost?.toLocaleString()}</span></div>
                    <div className="bk-row"><span className="bk-l">Tier factor ({result.ipd_tier})</span><span className="bk-v hi">{result.ipd_core?.tier_factor}x</span></div>
                    {result.ipd_core?.deductible_credit > 0 && <div className="bk-row"><span className="bk-l">Deductible credit</span><span className="bk-v" style={{ color: "var(--ok)" }}>-${result.ipd_core.deductible_credit}</span></div>}
                    <div className="bk-row" style={{ fontWeight: 600 }}><span className="bk-l" style={{ fontWeight: 600 }}>IPD premium</span><span className="bk-v hi">${result.ipd_core?.annual_premium?.toLocaleString()}/yr</span></div>
                  </div>

                  {/* Riders */}
                  {Object.entries(result.riders || {}).map(([k, v]) => (
                    <div className="bk-section" key={k} style={{ paddingTop: 12, borderTop: "1px solid var(--surf3)" }}>
                      <div className="bk-head"><span className="bk-badge rider">{k.toUpperCase()}</span> {v.name}</div>
                      <div className="bk-row"><span className="bk-l">Freq / Sev</span><span className="bk-v">{v.frequency}/yr · ${v.severity?.toLocaleString()}</span></div>
                      <div className="bk-row" style={{ fontWeight: 600 }}><span className="bk-l" style={{ fontWeight: 600 }}>Rider premium</span><span className="bk-v gold">${v.annual_premium?.toLocaleString()}/yr</span></div>
                    </div>
                  ))}

                  {/* Family */}
                  {result.family_size > 1 && (
                    <div className="bk-row" style={{ paddingTop: 12, borderTop: "1px solid var(--surf3)" }}>
                      <span className="bk-l">Family ({result.family_size})</span><span className="bk-v">{result.family_factor}x</span>
                    </div>
                  )}

                  {/* Benefits */}
                  <div className="bk-section" style={{ paddingTop: 12, borderTop: "1px solid var(--surf3)" }}>
                    <div className="bk-head">{result.ipd_tier} benefits</div>
                    <div className="bk-row"><span className="bk-l">Limit</span><span className="bk-v">{TIERS[result.ipd_tier]?.limit}</span></div>
                    <div className="bk-row"><span className="bk-l">Room</span><span className="bk-v">{TIERS[result.ipd_tier]?.room}</span></div>
                    <div className="bk-row"><span className="bk-l">Surgery</span><span className="bk-v">{TIERS[result.ipd_tier]?.surg}</span></div>
                    <div className="bk-row"><span className="bk-l">ICU</span><span className="bk-v">{TIERS[result.ipd_tier]?.icu}</span></div>
                  </div>

                  <div className="qid"><span>Quote ID</span><code>{result.quote_id}</code></div>
                  {isLocal && <div style={{ marginTop: 8, padding: "10px 12px", borderRadius: 8, background: "#fffbeb", border: "1px solid #fef3c7", color: "#92400e", fontSize: 11, lineHeight: 1.5 }}>⚠️ <strong>Simplified actuarial model used</strong> — backend unavailable. This estimate uses a log-linear GLM formula and may differ from the ML model. Your advisor will confirm the exact figure.</div>}
                </div>

                <div className="btn-row">
                  <button className="btn btn-back" onClick={() => setStep(2)}>Modify plan</button>
                  <button className="btn btn-next" onClick={() => { setStep(0); setResult(null); }}>New quote</button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* AI CHAT */}
        <AIChat
          inp={inp}
          result={result}
          sessionToken={sessionTokenRef}
          onSwitchTier={tier => u("ipd_tier", tier)}
          onToggleRider={(rider, on) => u(rider, on)}
          onCalculateWith={calculate}
          onGoToStep={setStep}
        />

        <footer className="footer">
          DAC HealthPrice · Cambodia
          <div className="footer-tags">
            <span className="footer-tag">Freq-Sev</span>
            <span className="footer-tag">FastAPI</span>
            <span className="footer-tag">Supabase</span>
            <span className="footer-tag">v2.1</span>
          </div>
        </footer>
      </div>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// AI CHAT COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
const TOOLS = [
  {
    name: "switch_tier",
    description: "Change the user's selected IPD tier. Use when the user asks to upgrade, downgrade, or try a different tier.",
    input_schema: {
      type: "object",
      properties: {
        tier: { type: "string", enum: ["Bronze", "Silver", "Gold", "Platinum"], description: "The tier to switch to." }
      },
      required: ["tier"]
    }
  },
  {
    name: "toggle_rider",
    description: "Enable or disable an optional coverage rider (OPD, Dental, or Maternity).",
    input_schema: {
      type: "object",
      properties: {
        rider: { type: "string", enum: ["include_opd", "include_dental", "include_maternity"], description: "Which rider to toggle." },
        enabled: { type: "boolean", description: "true to add the rider, false to remove it." }
      },
      required: ["rider", "enabled"]
    }
  },
  {
    name: "recalculate_quote",
    description: "Run the pricing engine with current inputs and navigate to the results step. Always call this after switching tiers or toggling riders so the user sees updated premium figures.",
    input_schema: { type: "object", properties: {} }
  },
  {
    name: "navigate_to_step",
    description: "Navigate the wizard to a specific step. Steps: 0=Personal info, 1=Health info, 2=Lifestyle, 3=Plan selection, 4=Quote result.",
    input_schema: {
      type: "object",
      properties: {
        step: { type: "number", enum: [0, 1, 2, 3, 4, 5], description: "Step number to navigate to. 0=Personal info, 1=Health info, 2=Lifestyle, 3=Plan selection, 4=Review & confirm, 5=Quote result." }
      },
      required: ["step"]
    }
  }
];

const RIDER_NAMES = { include_opd: "OPD", include_dental: "Dental", include_maternity: "Maternity" };
const STEP_NAMES = ["Personal info", "Health info", "Lifestyle", "Plan selection", "Review & confirm", "Quote result"];

const CHAT_BACKEND = "https://dac-healthprice-api.onrender.com";

function AIChat({ inp, result, sessionToken, onSwitchTier, onToggleRider, onCalculateWith, onGoToStep }) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState([
    { role: "bot", text: "Hi! I'm your AI advisor. Once you have a quote, I can help you optimize your plan — switch tiers, add riders, explain your premium, or recalculate. You can also ask me questions at any step." }
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const apiMsgs = useRef([]);      // Claude conversation history (separate from display)
  const pendingChanges = useRef({}); // tracks in-flight state changes during tool loop
  const warmedUp = useRef(false);

  // Warm up backend the moment chat opens (prevents Render cold-start timeout)
  useEffect(() => {
    if (open && !warmedUp.current) {
      warmedUp.current = true;
      fetch(`${CHAT_BACKEND}/health`, { signal: AbortSignal.timeout(60000) }).catch(() => { });
    }
  }, [open]);

  const scroll = () => setTimeout(() => {
    const el = document.getElementById("ai-m");
    if (el) el.scrollTop = el.scrollHeight;
  }, 50);

  const buildCtx = () => {
    const i = { ...inp, ...pendingChanges.current };
    const wm = (i.exercise_days || 0) * (i.exercise_mins || 0);
    const profile = `Profile: age=${i.age}, gender=${i.gender}, Cambodia, region=${i.region}, smoking=${i.smoking_status}, exercise=${i.exercise_days}d x ${i.exercise_mins}min (${wm}min/wk, ${i.exercise_frequency}), occupation=${i.occupation_type}, conditions=${i.preexist_conditions.join(",")}, family=${i.family_size}.`;
    const plan = `Plan: tier=${i.ipd_tier}, opd=${i.include_opd}, dental=${i.include_dental}, maternity=${i.include_maternity}.`;
    const quote = result
      ? `Quote: $${result.total_annual_premium}/yr, IPD freq=${result.ipd_core?.frequency}, sev=$${result.ipd_core?.severity}, prem=$${result.ipd_core?.annual_premium}. Riders: ${Object.entries(result.riders || {}).map(([k, v]) => `${k}=$${v.annual_premium}`).join(",") || "none"}.`
      : "No quote calculated yet.";
    return `${profile} ${plan} ${quote}`;
  };

  const executeTool = async (name, toolInput) => {
    if (name === "switch_tier") {
      pendingChanges.current.ipd_tier = toolInput.tier;
      onSwitchTier(toolInput.tier);
      setMsgs(p => [...p, { role: "action", label: `Switched tier to ${toolInput.tier}` }]);
      scroll();
      return `Tier switched to ${toolInput.tier}.`;
    }
    if (name === "toggle_rider") {
      pendingChanges.current[toolInput.rider] = toolInput.enabled;
      onToggleRider(toolInput.rider, toolInput.enabled);
      const riderName = RIDER_NAMES[toolInput.rider];
      setMsgs(p => [...p, { role: "action", label: `${toolInput.enabled ? "Added" : "Removed"} ${riderName} rider` }]);
      scroll();
      return `${riderName} rider ${toolInput.enabled ? "enabled" : "disabled"}.`;
    }
    if (name === "recalculate_quote") {
      const merged = { ...inp, ...pendingChanges.current };
      setMsgs(p => [...p, { role: "action", label: "Recalculating quote…" }]);
      scroll();
      const res = await onCalculateWith(merged);
      if (res) {
        setMsgs(p => { const next = [...p]; next[next.length - 1] = { role: "action", label: `Quote updated — $${res.total_annual_premium}/yr` }; return next; });
        pendingChanges.current = {};
        const riderSummary = Object.keys(res.riders || {}).length
          ? `Riders included: ${Object.keys(res.riders).map(k => RIDER_NAMES[`include_${k}`] || k).join(", ")}.`
          : "No riders.";
        return `New quote: $${res.total_annual_premium}/yr ($${res.total_monthly_premium}/mo) with ${merged.ipd_tier} tier. ${riderSummary}`;
      }
      return "Recalculation complete.";
    }
    if (name === "navigate_to_step") {
      onGoToStep(toolInput.step);
      setMsgs(p => [...p, { role: "action", label: `Navigated to ${STEP_NAMES[toolInput.step]}` }]);
      scroll();
      return `Navigated to step ${toolInput.step} (${STEP_NAMES[toolInput.step]}).`;
    }
    return "Unknown tool.";
  };

  const send = async (text) => {
    if (!text.trim() || thinking) return;
    const userText = text.trim();
    setMsgs(p => [...p, { role: "user", text: userText }]);
    setInput("");
    setThinking(true);
    scroll();

    apiMsgs.current = [...apiMsgs.current, { role: "user", content: userText }];

    const systemPrompt = `You are an insurance advisor for DAC HealthPrice Cambodia. You have three roles: (1) recommend tiers and riders, (2) explain premium factors, (3) suggest optimizations. You also have tools to act directly on the user's plan — switch tiers, add/remove riders, recalculate quotes, and navigate the wizard. Use tools when the user asks for changes, not just advice. Always call recalculate_quote after making plan changes so the user sees updated prices. Be concise: 2-3 sentences max. Use **bold** for amounts. Current state:\n${buildCtx()}`;

    try {
      let iterations = 0;
      while (iterations < 5) {
        iterations++;
        const r = await fetch(`${CHAT_BACKEND}/api/v2/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: AbortSignal.timeout(90000),
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            max_tokens: 800,
            system: systemPrompt,
            tools: TOOLS,
            messages: apiMsgs.current.slice(-12),
          }),
        });

        const d = await r.json();
        apiMsgs.current = [...apiMsgs.current, { role: "assistant", content: d.content }];

        if (d.stop_reason === "end_turn") {
          const textBlock = d.content.find(b => b.type === "text");
          if (textBlock) {
            const reply = textBlock.text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
            setMsgs(p => [...p, { role: "bot", text: reply }]);
          }
          break;
        }

        if (d.stop_reason === "tool_use") {
          const toolResults = [];
          for (const block of d.content) {
            if (block.type === "tool_use") {
              const toolResult = await executeTool(block.name, block.input);
              toolResults.push({ type: "tool_result", tool_use_id: block.id, content: toolResult });
            }
          }
          apiMsgs.current = [...apiMsgs.current, { role: "user", content: toolResults }];
          scroll();
        } else {
          break;
        }
      }
    } catch {
      setMsgs(p => [...p, { role: "bot", text: "Connection issue. Try again." }]);
    } finally {
      setThinking(false);
      scroll();
    }
  };

  const qs = result
    ? ["Why this amount?", "Switch to Gold", "Add OPD", "Lower my premium"]
    : ["Best tier for me?", "Explain tiers", "Do I need OPD?", "Pre-existing conditions help"];

  return (
    <>
      <button className="ai-fab" onClick={() => setOpen(!open)}>
        {open
          ? <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
          : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
        }
      </button>

      {open && (
        <div className="ai-panel">
          <div className="ai-head">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 24, height: 24, borderRadius: "50%", background: "var(--navy)", color: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700 }}>AI</div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600 }}>Insurance advisor</div>
                <div style={{ fontSize: 10, color: "var(--txt3)" }}>Plan · Risk · Savings · Actions</div>
              </div>
            </div>
            <button style={{ background: "none", border: "none", cursor: "pointer", color: "var(--txt3)", fontSize: 16 }} onClick={() => setOpen(false)}>×</button>
          </div>

          <div className="ai-msgs" id="ai-m">
            {msgs.map((m, i) => {
              if (m.role === "action") return (
                <div key={i} style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 5, padding: "5px 10px", borderRadius: 7, background: "var(--gold-bg)", border: "1px solid var(--gold-bd)", fontSize: 11, color: "var(--gold-d)", fontWeight: 600, maxWidth: "86%" }}>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="3" /><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14" /></svg>
                  {m.label}
                </div>
              );
              return <div key={i} className={`ai-msg ${m.role === "user" ? "user" : "bot"}`} dangerouslySetInnerHTML={{ __html: m.text }} />;
            })}
            {thinking && <div className="ai-msg typing">Thinking…</div>}
            {!thinking && msgs.length <= 2 && (
              <div className="ai-qchips">
                {qs.map((q, i) => (
                  <div key={i} className="ai-qchip" onClick={() => send(q)}>{q}</div>
                ))}
              </div>
            )}
          </div>

          <div className="ai-input-row">
            <input
              className="ai-input"
              placeholder="Ask or say 'Switch to Gold'…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !thinking) send(input); }}
              disabled={thinking}
            />
            <button className="ai-send" onClick={() => send(input)} disabled={thinking || !input.trim()}>Send</button>
          </div>
        </div>
      )}
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADMIN COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function AdminStatus({ apiOk, adminKey }) {
  const [info, setInfo] = useState(null);
  useEffect(() => {
    apiCall("/api/v2/model-info").then(setInfo).catch(() => { });
  }, []);

  const retrainedAt = info?.last_retrained_at
    ? new Date(info.last_retrained_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : "—";

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 12, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)", marginBottom: 2 }}>API</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: apiOk ? "var(--ok)" : "var(--danger)" }}>{apiOk ? "Connected" : "Offline"}</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 12, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)", marginBottom: 2 }}>Model version</div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{info?.version || "—"}</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 12, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)", marginBottom: 2 }}>Models loaded</div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{info?.models?.length || 0}/8</div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10 }}>
          <div style={{ fontSize: 10, color: "var(--txt3)", marginBottom: 2 }}>Last retrained</div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>{retrainedAt}</div>
          {info?.training_dataset && <div style={{ fontSize: 10, color: "var(--txt3)", marginTop: 2 }}>{info.training_dataset}</div>}
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10 }}>
          <div style={{ fontSize: 10, color: "var(--txt3)", marginBottom: 2 }}>GLM fallback</div>
          <div style={{ fontSize: 12, fontWeight: 600, color: info?.fallback_models_loaded ? "var(--ok)" : "var(--danger)" }}>
            {info?.fallback_models_loaded ? "Loaded" : info === null ? "—" : "Not loaded"}
          </div>
          {info?.r2 !== undefined && <div style={{ fontSize: 10, color: "var(--txt3)", marginTop: 2 }}>Primary R² {info.r2}</div>}
        </div>
      </div>
      {info?.models && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {info.models.map(m => (
            <span key={m} style={{ padding: "3px 8px", borderRadius: 4, background: "var(--surf2)", fontSize: 10, color: "var(--txt2)" }}>{m}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function AdminUpload({ adminKey }) {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [covType, setCovType] = useState("ipd");
  const [autoRetrain, setAutoRetrain] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith(".csv")) { alert("Please select a CSV file"); return; }
    setUploading(true); setUploadResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("coverage_type", covType);
      form.append("auto_retrain", autoRetrain.toString());
      const r = await fetch(`${API}/api/v2/admin/upload-dataset`, {
        method: "POST", body: form,
        headers: { "X-API-Key": adminKey },
        signal: AbortSignal.timeout(120000),
      });
      setUploadResult(await r.json());
    } catch (e) {
      setUploadResult({ status: "error", detail: e.message });
    } finally { setUploading(false); }
  };

  const downloadTemplate = async () => {
    try {
      const r = await fetch(`${API}/api/v2/admin/dataset-template`, { headers: { "X-API-Key": adminKey } });
      const blob = await r.blob();
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "claims_template.csv"; a.click();
    } catch { alert("Failed to download template"); }
  };

  return (
    <div>
      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => document.getElementById("admin-file").click()}
        style={{
          border: `2px dashed ${dragOver ? "var(--gold)" : "var(--surf3)"}`,
          borderRadius: 10, padding: "28px 20px", textAlign: "center", cursor: "pointer",
          background: dragOver ? "var(--gold-bg)" : "var(--surf2)", transition: "all .2s", marginBottom: 12,
        }}
      >
        <input id="admin-file" type="file" accept=".csv" onChange={e => handleFile(e.target.files[0])} style={{ display: "none" }} />
        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--txt2)" }}>
          {uploading ? "Uploading..." : "Drag & drop CSV or click to browse"}
        </div>
        <div style={{ fontSize: 11, color: "var(--txt3)", marginTop: 4 }}>Required: age, gender, smoking, exercise, occupation, region, preexist_count, claim_count, claim_amount</div>
      </div>

      {/* Options */}
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <label style={{ fontSize: 11, color: "var(--txt2)" }}>Coverage:</label>
          <select value={covType} onChange={e => setCovType(e.target.value)}
            style={{ padding: "5px 8px", borderRadius: 6, border: "1px solid var(--surf3)", fontSize: 12, fontFamily: "var(--fb)", background: "white" }}>
            <option value="ipd">IPD</option><option value="opd">OPD</option><option value="dental">Dental</option><option value="maternity">Maternity</option>
          </select>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }} onClick={() => setAutoRetrain(!autoRetrain)}>
          <div style={{ width: 16, height: 16, borderRadius: 4, border: `2px solid ${autoRetrain ? "var(--gold-d)" : "var(--surf3)"}`, background: autoRetrain ? "var(--gold)" : "white", display: "flex", alignItems: "center", justifyContent: "center" }}>
            {autoRetrain && <Ck s={8} />}
          </div>
          <span style={{ fontSize: 11, color: "var(--txt2)" }}>Auto-retrain after upload</span>
        </div>
        <button onClick={downloadTemplate} style={{ marginLeft: "auto", padding: "5px 12px", borderRadius: 6, border: "1px solid var(--surf3)", background: "white", fontSize: 11, cursor: "pointer", fontFamily: "var(--fb)", color: "var(--txt2)" }}>
          Download template
        </button>
      </div>

      {/* Result */}
      {uploadResult && (
        <div style={{
          padding: 14, borderRadius: 8, fontSize: 12,
          background: uploadResult.status === "accepted" ? "#e1f5ee" : "#fef2f2",
          border: `1px solid ${uploadResult.status === "accepted" ? "#9fe1cb" : "#fca5a5"}`,
          color: uploadResult.status === "accepted" ? "#085041" : "#dc2626",
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            {uploadResult.status === "accepted" ? "Dataset accepted" : uploadResult.status === "rejected" ? "Rejected" : "Error"}
          </div>
          {uploadResult.rows_parsed && <div>Rows parsed: {uploadResult.rows_parsed} | Inserted: {uploadResult.rows_inserted || uploadResult.inserted || 0}</div>}
          {uploadResult.retrain && uploadResult.retrain.status === "promoted" && (
            <div style={{ marginTop: 4, color: "#059669" }}>Model retrained: {uploadResult.retrain.version} (R²={uploadResult.retrain.r2})</div>
          )}
          {uploadResult.retrain && uploadResult.retrain.status === "rejected" && (
            <div style={{ marginTop: 4 }}>Challenger rejected — champion R²={uploadResult.retrain.champion_r2} vs challenger R²={uploadResult.retrain.r2}</div>
          )}
          {uploadResult.detail && <div>{uploadResult.detail}</div>}
          {uploadResult.missing && <div>Missing columns: {uploadResult.missing.join(", ")}</div>}
        </div>
      )}
    </div>
  );
}

function AdminHistory({ adminKey }) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v2/admin/upload-history`, { headers: { "X-API-Key": adminKey } });
      const data = await r.json();
      setHistory(data);
    } catch { setHistory({ status: "error" }); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  if (!history) return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>{loading ? "Loading..." : "No data"}</div>;
  if (history.status === "no_db") return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>Database not connected</div>;
  if (history.status === "error") return <div style={{ fontSize: 12, color: "var(--danger)", textAlign: "center", padding: 16 }}>Failed to load history</div>;

  const uploads = history.uploads || [];
  if (uploads.length === 0) return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <div style={{ fontSize: 12, color: "var(--txt3)", marginBottom: 8 }}>No datasets uploaded yet</div>
      <div style={{ fontSize: 11, color: "var(--txt3)" }}>Upload a CSV above to get started</div>
    </div>
  );

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
        <button onClick={load} style={{ padding: "4px 10px", borderRadius: 5, border: "1px solid var(--surf3)", background: "white", fontSize: 10, cursor: "pointer", fontFamily: "var(--fb)", color: "var(--txt3)" }}>
          Refresh
        </button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {uploads.map((u, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 12px", background: "var(--surf2)", borderRadius: 8, fontSize: 12 }}>
            <div>
              <div style={{ fontWeight: 600 }}>{u.batch_id}</div>
              <div style={{ fontSize: 10, color: "var(--txt3)" }}>{u.coverage_type} · {u.rows} rows · avg ${u.avg}</div>
            </div>
            <div style={{ fontSize: 10, color: "var(--txt3)", textAlign: "right" }}>
              {u.uploaded_at ? new Date(u.uploaded_at).toLocaleDateString() : "—"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AdminUserData({ adminKey }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v2/admin/user-behavior`, { headers: { "X-API-Key": adminKey } });
      setData(await r.json());
    } catch { setData({ status: "error" }); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  if (!data) return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>{loading ? "Loading..." : "No data"}</div>;
  if (data.status === "no_db") return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>Database not connected</div>;
  if (data.status === "error") return <div style={{ fontSize: 12, color: "var(--danger)", textAlign: "center", padding: 16 }}>Failed to load user data</div>;

  const { summary = {}, records = [] } = data;

  if (records.length === 0) return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <div style={{ fontSize: 12, color: "var(--txt3)" }}>No user quotes recorded yet</div>
      <div style={{ fontSize: 11, color: "var(--txt3)", marginTop: 4 }}>Quotes will appear here as users calculate premiums</div>
    </div>
  );

  const tierColors = { Bronze: "#92400e", Silver: "#475569", Gold: "#c46800", Platinum: "#1e40af" };
  const tierBg = { Bronze: "#fffbeb", Silver: "#f1f3f5", Gold: "#fff7ed", Platinum: "#eff6ff" };

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 14 }}>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Total quotes</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.total_quotes || 0}</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Avg age</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.avg_age || "—"}</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>OPD rider %</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.rider_rates?.opd || 0}%</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Dental rider %</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.rider_rates?.dental || 0}%</div>
        </div>
      </div>

      {/* Tier distribution */}
      {summary.tier_distribution && Object.keys(summary.tier_distribution).length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: "var(--txt3)", marginBottom: 6, fontWeight: 600 }}>Tier distribution</div>
          <div style={{ display: "flex", gap: 6 }}>
            {Object.entries(summary.tier_distribution).map(([tier, count]) => {
              const pct = Math.round(count / (summary.total_quotes || 1) * 100);
              return (
                <div key={tier} style={{ flex: 1, background: tierBg[tier] || "var(--surf2)", borderRadius: 8, padding: "8px 6px", textAlign: "center" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: tierColors[tier] || "var(--txt)" }}>{tier}</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: tierColors[tier] || "var(--txt)" }}>{pct}%</div>
                  <div style={{ fontSize: 9, color: "var(--txt3)" }}>{count} quotes</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Smoking distribution */}
      {summary.smoking_distribution && Object.keys(summary.smoking_distribution).length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: "var(--txt3)", marginBottom: 6, fontWeight: 600 }}>Smoking status</div>
          <div style={{ display: "flex", gap: 6 }}>
            {Object.entries(summary.smoking_distribution).map(([status, count]) => {
              const pct = Math.round(count / (summary.total_quotes || 1) * 100);
              const colors = { Never: "#059669", Former: "#c46800", Current: "#dc2626" };
              const bgs = { Never: "#e1f5ee", Former: "#fffbeb", Current: "#fef2f2" };
              return (
                <div key={status} style={{ flex: 1, background: bgs[status] || "var(--surf2)", borderRadius: 8, padding: "8px 6px", textAlign: "center" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: colors[status] || "var(--txt)" }}>{status}</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: colors[status] || "var(--txt)" }}>{pct}%</div>
                  <div style={{ fontSize: 9, color: "var(--txt3)" }}>{count} users</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Toggle table */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <button onClick={() => setShowTable(!showTable)} style={{
          padding: "5px 12px", borderRadius: 6, border: "1px solid var(--surf3)", background: showTable ? "var(--navy)" : "white",
          fontSize: 11, cursor: "pointer", fontFamily: "var(--fb)", color: showTable ? "white" : "var(--txt2)",
        }}>
          {showTable ? "Hide" : "Show"} recent quotes ({records.length})
        </button>
        <button onClick={load} style={{ padding: "4px 10px", borderRadius: 5, border: "1px solid var(--surf3)", background: "white", fontSize: 10, cursor: "pointer", fontFamily: "var(--fb)", color: "var(--txt3)" }}>
          Refresh
        </button>
      </div>

      {/* Records table */}
      {showTable && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "1.5px solid var(--surf3)" }}>
                {["Time", "Age", "Gender", "Region", "Smoking", "Occupation", "Conditions", "Tier", "Riders", "Family"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 6px", fontSize: 10, color: "var(--txt3)", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.slice(0, 30).map((r, i) => {
                const riders = [r.include_opd && "OPD", r.include_dental && "Den", r.include_maternity && "Mat"].filter(Boolean).join(", ") || "—";
                return (
                  <tr key={i} style={{ borderBottom: "1px solid var(--surf2)" }}>
                    <td style={{ padding: "6px", color: "var(--txt3)", fontSize: 10 }}>{r.created_at ? new Date(r.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                    <td style={{ padding: "6px" }}>{r.age}</td>
                    <td style={{ padding: "6px" }}>{r.gender}</td>
                    <td style={{ padding: "6px", fontSize: 10 }}>{r.region}</td>
                    <td style={{ padding: "6px" }}>
                      <span style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, background: r.smoking === "Current" ? "#fef2f2" : r.smoking === "Former" ? "#fffbeb" : "#e1f5ee", color: r.smoking === "Current" ? "#dc2626" : r.smoking === "Former" ? "#c46800" : "#059669" }}>{r.smoking}</span>
                    </td>
                    <td style={{ padding: "6px", fontSize: 10 }}>{r.occupation}</td>
                    <td style={{ padding: "6px" }}>{r.preexist_count || 0}</td>
                    <td style={{ padding: "6px" }}>
                      <span style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, background: tierBg[r.ipd_tier] || "var(--surf2)", color: tierColors[r.ipd_tier] || "var(--txt)", fontWeight: 600 }}>{r.ipd_tier}</span>
                    </td>
                    <td style={{ padding: "6px", fontSize: 10 }}>{riders}</td>
                    <td style={{ padding: "6px" }}>{r.family_size}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AdminAuditLog({ adminKey }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v2/admin/audit-log`, { headers: { "X-API-Key": adminKey } });
      setData(await r.json());
    } catch { setData({ status: "error" }); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  if (!data) return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>{loading ? "Loading..." : "No data"}</div>;
  if (data.status === "no_db") return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>Database not connected</div>;
  if (data.status === "error") return <div style={{ fontSize: 12, color: "var(--danger)", textAlign: "center", padding: 16 }}>Failed to load audit log</div>;

  const { summary = {}, records = [] } = data;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 14 }}>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Total quotes</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.total_quotes ?? records.length}</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Fallback rate</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: (summary.fallback_rate ?? 0) > 20 ? "#dc2626" : "var(--txt)" }}>
            {summary.fallback_rate ?? "—"}%
          </div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Decline rate</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.decline_rate ?? "—"}%</div>
        </div>
        <div style={{ background: "var(--surf2)", borderRadius: 8, padding: 10, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "var(--txt3)" }}>Refer rate</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>{summary.refer_rate ?? "—"}%</div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <button onClick={() => setShowTable(!showTable)} style={{
          padding: "5px 12px", borderRadius: 6, border: "1px solid var(--surf3)", background: showTable ? "var(--navy)" : "white",
          fontSize: 11, cursor: "pointer", fontFamily: "var(--fb)", color: showTable ? "white" : "var(--txt2)",
        }}>
          {showTable ? "Hide" : "Show"} log ({records.length})
        </button>
        <button onClick={load} style={{ padding: "4px 10px", borderRadius: 5, border: "1px solid var(--surf3)", background: "white", fontSize: 10, cursor: "pointer", fontFamily: "var(--fb)", color: "var(--txt3)" }}>
          Refresh
        </button>
      </div>

      {showTable && records.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "1.5px solid var(--surf3)" }}>
                {["Time", "Age/Gender", "Tier", "UW Status", "Fallback", "Model", "Email"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 6px", fontSize: 10, color: "var(--txt3)", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.slice(0, 50).map((r, i) => {
                const uwColor = r.underwriting_status === "decline" ? "#dc2626" : r.underwriting_status === "refer" ? "#c46800" : "#059669";
                const uwBg = r.underwriting_status === "decline" ? "#fef2f2" : r.underwriting_status === "refer" ? "#fffbeb" : "#e1f5ee";
                return (
                  <tr key={i} style={{ borderBottom: "1px solid var(--surf2)" }}>
                    <td style={{ padding: "6px", color: "var(--txt3)", fontSize: 10 }}>{r.created_at ? new Date(r.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                    <td style={{ padding: "6px" }}>{r.age}{r.gender ? `/${r.gender[0]}` : ""}</td>
                    <td style={{ padding: "6px", fontSize: 10 }}>{r.ipd_tier || "—"}</td>
                    <td style={{ padding: "6px" }}>
                      <span style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, background: uwBg, color: uwColor, fontWeight: 600 }}>
                        {r.underwriting_status || "accept"}
                      </span>
                    </td>
                    <td style={{ padding: "6px" }}>
                      <span style={{ fontSize: 10, color: r.used_fallback ? "#dc2626" : "var(--txt3)" }}>
                        {r.used_fallback ? "Yes" : "No"}
                      </span>
                    </td>
                    <td style={{ padding: "6px", fontSize: 10, color: "var(--txt3)" }}>{r.model_version ? r.model_version.slice(0, 12) : "—"}</td>
                    <td style={{ padding: "6px", fontSize: 10, color: "var(--txt3)" }}>{r.email || "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AdminModelVersions({ adminKey }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v2/admin/model-versions`, { headers: { "X-API-Key": adminKey } });
      setData(await r.json());
    } catch { setData({ status: "error" }); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  if (!data) return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>{loading ? "Loading..." : "No data"}</div>;
  if (data.status === "no_db") return <div style={{ fontSize: 12, color: "var(--txt3)", textAlign: "center", padding: 16 }}>Database not connected</div>;
  if (data.status === "error") return <div style={{ fontSize: 12, color: "var(--danger)", textAlign: "center", padding: 16 }}>Failed to load version history</div>;

  const versions = data.versions || [];

  if (versions.length === 0) return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <div style={{ fontSize: 12, color: "var(--txt3)" }}>No version history yet</div>
      <div style={{ fontSize: 11, color: "var(--txt3)", marginTop: 4 }}>Versions are logged after each retrain</div>
    </div>
  );

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
        <button onClick={load} style={{ padding: "4px 10px", borderRadius: 5, border: "1px solid var(--surf3)", background: "white", fontSize: 10, cursor: "pointer", fontFamily: "var(--fb)", color: "var(--txt3)" }}>
          Refresh
        </button>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1.5px solid var(--surf3)" }}>
              {["Version", "Trained at", "Dataset", "R²", "Rows", "Promoted by"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "6px 6px", fontSize: 10, color: "var(--txt3)", fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {versions.map((v, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--surf2)", background: i === 0 ? "#f0fdf4" : "transparent" }}>
                <td style={{ padding: "6px", fontWeight: i === 0 ? 600 : 400, fontSize: 10, fontFamily: "monospace" }}>{v.version}</td>
                <td style={{ padding: "6px", color: "var(--txt3)", fontSize: 10 }}>
                  {v.trained_at ? new Date(v.trained_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}
                </td>
                <td style={{ padding: "6px", fontSize: 10 }}>{v.dataset || "—"}</td>
                <td style={{ padding: "6px" }}>{v.r2 !== undefined && v.r2 !== null ? v.r2 : "—"}</td>
                <td style={{ padding: "6px" }}>{v.training_rows ?? "—"}</td>
                <td style={{ padding: "6px", fontSize: 10, color: "var(--txt3)" }}>{v.promoted_by || "auto"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}