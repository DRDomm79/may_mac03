/**
 * ActuarialWorkbench.jsx
 *
 * Health GLM Pricing Workbench — Attribution, Sensitivity, Assumptions Editor
 * Client-side port of: C:\DAC\dac-health\backend\app\main.py  (COEFF + _glm_predict)
 * Model version: v2.3  |  Poisson-Gamma GLM
 *
 * Tabs:
 *   Attribution  — SHAP-style waterfall: log-space contribution per factor
 *   Sensitivity  — vary one factor across all levels, compare premiums
 *   Assumptions  — edit COEFF multipliers in-session; premium recalculates live
 */

import { useState, useMemo } from "react";

// ── Brand colours (matches app) ───────────────────────────────────────────────
const NAVY   = "#0d2b7a";
const NAVY_D = "#091d5e";
const GOLD   = "#f5a623";
const WHITE  = "#ffffff";
const LTGRAY = "#f8f9fb";
const TXT    = "#111827";
const TXT2   = "#4b5563";
const OK     = "#10b981";
const ERR    = "#ef4444";
const WARN   = "#f59e0b";
const TEAL   = "#0d9488";

// ── Default COEFF — mirrors main.py exactly ───────────────────────────────────
const DEFAULT_COEFF = {
  version: "v2.3",
  base_freq:          { ipd: 0.12, opd: 2.5, dental: 0.80, maternity: 0.15 },
  base_sev:           { ipd: 2500, opd: 60,  dental: 120,  maternity: 3500  },
  age_factors:        { "18-24": 0.85, "25-34": 1.00, "35-44": 1.12, "45-54": 1.28, "55-64": 1.48, "65+": 1.72 },
  smoking_factors:    { Never: 1.00, Former: 1.15, Current: 1.40 },
  exercise_factors:   { Sedentary: 1.20, Light: 1.05, Moderate: 0.90, Active: 0.80 },
  occupation_factors: { "Office/Desk": 0.85, "Retail/Service": 1.00, Healthcare: 1.05, "Manual Labor": 1.15, "Industrial/High-Risk": 1.30, Retired: 1.10 },
  region_factors:     { "Phnom Penh": 1.20, "Siem Reap": 1.05, Battambang: 0.90, Sihanoukville: 1.10, "Kampong Cham": 0.85, "Rural Areas": 0.75 },
  tier_factors:       { Bronze: 0.70, Silver: 1.00, Gold: 1.45, Platinum: 2.10 },
  preexist_per_condition: 0.20,
  sev_age_gradient:   0.006,
  load_factor:        0.30,
};

// ── Pure actuarial functions ──────────────────────────────────────────────────
function getAgeBand(age) {
  if (age < 25) return "18-24";
  if (age < 35) return "25-34";
  if (age < 45) return "35-44";
  if (age < 55) return "45-54";
  if (age < 65) return "55-64";
  return "65+";
}

function calcGLM(profile, coeff) {
  const cov = profile.coverage || "ipd";
  const ab  = getAgeBand(profile.age);

  const af   = coeff.age_factors[ab]                          ?? 1.0;
  const sf   = coeff.smoking_factors[profile.smoking]         ?? 1.0;
  const ef   = coeff.exercise_factors[profile.exercise]       ?? 1.0;
  const of_  = coeff.occupation_factors[profile.occupation]   ?? 1.0;
  const rf   = coeff.region_factors[profile.region]           ?? 1.0;
  const pf   = 1 + (profile.n_preexist ?? 0) * coeff.preexist_per_condition;
  const tf   = coeff.tier_factors[profile.tier]               ?? 1.0;

  const freq    = coeff.base_freq[cov] * af * sf * ef * of_ * pf;
  const sev     = coeff.base_sev[cov] * rf * (1 + Math.max(0, profile.age - 30) * coeff.sev_age_gradient);
  const expected = freq * sev;
  const premium  = expected * tf * (1 + coeff.load_factor);

  return {
    premium:  Math.round(premium),
    expected: Math.round(expected),
    freq:     parseFloat(freq.toFixed(4)),
    sev:      Math.round(sev),
    factors:  { af, sf, ef, of: of_, rf, pf, tf },
    ab,
  };
}

function buildAttribution(profile, coeff) {
  const cov  = profile.coverage || "ipd";
  const ab   = getAgeBand(profile.age);
  const r    = calcGLM(profile, coeff);
  const { factors } = r;

  // Base premium = base_freq × base_sev × tier × (1 + load) — no risk adjustments
  const basePremium = coeff.base_freq[cov] * coeff.base_sev[cov]
    * (coeff.tier_factors[profile.tier] ?? 1.0)
    * (1 + coeff.load_factor);

  const items = [
    { label: `Age (${ab})`,                        factor: factors.af,  type: "freq" },
    { label: `Smoking (${profile.smoking})`,        factor: factors.sf,  type: "freq" },
    { label: `Exercise (${profile.exercise})`,      factor: factors.ef,  type: "freq" },
    { label: `Occupation (${profile.occupation})`,  factor: factors.of,  type: "freq" },
    { label: `Pre-existing (×${profile.n_preexist})`, factor: factors.pf, type: "freq" },
    { label: `Region (${profile.region})`,          factor: factors.rf,  type: "sev"  },
  ];

  const totalLogMult = items.reduce((s, it) => s + Math.log(it.factor), 0);

  return {
    basePremium: Math.round(basePremium),
    finalPremium: r.premium,
    items: items.map(it => ({
      ...it,
      logContrib:    Math.log(it.factor),
      pctContrib:    totalLogMult !== 0 ? (Math.log(it.factor) / Math.abs(totalLogMult)) * 100 : 0,
      premiumDelta:  Math.round(r.premium * (it.factor - 1) / (items.reduce((p, x) => p * x.factor, 1))),
    })).sort((a, b) => Math.abs(b.logContrib) - Math.abs(a.logContrib)),
  };
}

const SENSITIVITY_LEVELS = {
  age:        [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
  smoking:    ["Never", "Former", "Current"],
  exercise:   ["Sedentary", "Light", "Moderate", "Active"],
  occupation: Object.keys(DEFAULT_COEFF.occupation_factors),
  region:     Object.keys(DEFAULT_COEFF.region_factors),
  tier:       ["Bronze", "Silver", "Gold", "Platinum"],
  n_preexist: [0, 1, 2, 3, 4, 5],
};

function computeSensitivity(profile, coeff, factor) {
  const levels = SENSITIVITY_LEVELS[factor] ?? [];
  return levels.map(level => ({
    level:   String(level),
    premium: calcGLM({ ...profile, [factor]: level }, coeff).premium,
    current: level === profile[factor] || String(level) === String(profile[factor]),
  }));
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

// ── Sub-components ────────────────────────────────────────────────────────────
const inputStyle = {
  width: "100%", padding: "8px 12px", borderRadius: 8,
  border: "1.5px solid #e5e7eb", fontSize: 13, fontFamily: "inherit",
  outline: "none", background: WHITE, color: TXT,
};

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: TXT2, marginBottom: 5, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </label>
      {children}
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{ background: WHITE, borderRadius: 12, padding: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.07)", ...style }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return <div style={{ fontSize: 14, fontWeight: 700, color: NAVY, marginBottom: 16 }}>{children}</div>;
}

function PremiumBadge({ label, value, accent }) {
  return (
    <div style={{ background: accent ? NAVY : LTGRAY, borderRadius: 10, padding: "14px 18px", textAlign: "center" }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: accent ? "rgba(255,255,255,0.65)" : TXT2, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ fontSize: accent ? 26 : 20, fontWeight: 700, color: accent ? WHITE : NAVY }}>
        ${value.toLocaleString()}
      </div>
    </div>
  );
}

// Attribution waterfall bar
function WaterfallBar({ label, factor, logContrib, pctContrib }) {
  const positive = logContrib >= 0;
  const barWidth = Math.min(Math.abs(pctContrib), 100);
  const color    = positive ? ERR : OK;
  const bg       = positive ? "#fef2f2" : "#ecfdf5";

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: TXT, fontWeight: 500, minWidth: 200 }}>{label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, fontFamily: "monospace", color: TXT2 }}>×{factor.toFixed(3)}</span>
          <span style={{ fontSize: 12, fontWeight: 600, color, background: bg, borderRadius: 4, padding: "1px 6px", minWidth: 50, textAlign: "right" }}>
            {positive ? "+" : ""}{pctContrib.toFixed(1)}%
          </span>
        </div>
      </div>
      <div style={{ height: 8, background: "#f3f4f6", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${barWidth}%`,
          background: color,
          borderRadius: 4,
          float: positive ? "right" : "left",
          transition: "width 0.35s ease",
        }} />
      </div>
    </div>
  );
}

// Sensitivity comparison bar
function SensBar({ label, premium, maxPremium, isCurrent }) {
  const pct = maxPremium > 0 ? (premium / maxPremium) * 100 : 0;
  return (
    <div style={{
      marginBottom: 8, padding: "8px 12px", borderRadius: 8,
      background: isCurrent ? "#eff6ff" : LTGRAY,
      border: isCurrent ? `1.5px solid #bfdbfe` : "1.5px solid transparent",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: TXT, fontWeight: isCurrent ? 600 : 400 }}>
          {isCurrent && <span style={{ color: NAVY, marginRight: 4 }}>▶</span>}
          {label}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: NAVY }}>${premium.toLocaleString()}</span>
      </div>
      <div style={{ height: 6, background: "#e5e7eb", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: isCurrent ? NAVY : TEAL, borderRadius: 3, transition: "width 0.35s ease" }} />
      </div>
    </div>
  );
}

// Inline number input for assumptions editor
function AssumptionInput({ value, onChange }) {
  return (
    <input
      type="number"
      step="0.01"
      min="0.1"
      max="5"
      value={value}
      onChange={e => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) onChange(v);
      }}
      style={{
        width: 72, padding: "4px 8px", borderRadius: 6,
        border: "1.5px solid #d1d5db", fontSize: 12,
        fontFamily: "monospace", outline: "none", textAlign: "right",
      }}
    />
  );
}

// ── Profile input panel (shared across Attribution + Sensitivity tabs) ────────
function ProfilePanel({ profile, setP }) {
  const sel = v => ({ ...inputStyle, fontSize: 13 });
  return (
    <Card>
      <SectionTitle>Applicant Profile</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 16px" }}>
        <Field label="Age">
          <input type="number" min={18} max={75} value={profile.age}
            onChange={e => setP("age", Number(e.target.value))} style={inputStyle} />
        </Field>
        <Field label="Coverage">
          <select value={profile.coverage} onChange={e => setP("coverage", e.target.value)} style={sel()}>
            <option value="ipd">IPD Hospital</option>
            <option value="opd">OPD Rider</option>
            <option value="dental">Dental</option>
            <option value="maternity">Maternity</option>
          </select>
        </Field>
        <Field label="Smoking Status">
          <select value={profile.smoking} onChange={e => setP("smoking", e.target.value)} style={sel()}>
            {["Never", "Former", "Current"].map(v => <option key={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Tier">
          <select value={profile.tier} onChange={e => setP("tier", e.target.value)} style={sel()}>
            {["Bronze", "Silver", "Gold", "Platinum"].map(v => <option key={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Exercise">
          <select value={profile.exercise} onChange={e => setP("exercise", e.target.value)} style={sel()}>
            {["Sedentary", "Light", "Moderate", "Active"].map(v => <option key={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Occupation">
          <select value={profile.occupation} onChange={e => setP("occupation", e.target.value)} style={sel()}>
            {Object.keys(DEFAULT_COEFF.occupation_factors).map(v => <option key={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Region">
          <select value={profile.region} onChange={e => setP("region", e.target.value)} style={sel()}>
            {Object.keys(DEFAULT_COEFF.region_factors).map(v => <option key={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Pre-existing Conditions">
          <select value={profile.n_preexist} onChange={e => setP("n_preexist", Number(e.target.value))} style={sel()}>
            {[0, 1, 2, 3, 4, 5].map(v => <option key={v} value={v}>{v === 0 ? "None" : `${v} condition${v > 1 ? "s" : ""}`}</option>)}
          </select>
        </Field>
      </div>
    </Card>
  );
}

// ── Tab: Attribution ──────────────────────────────────────────────────────────
function AttributionTab({ profile, setP, coeff }) {
  const attr = useMemo(() => buildAttribution(profile, coeff), [profile, coeff]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
      <ProfilePanel profile={profile} setP={setP} />

      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Premium summary */}
        <Card>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
            <PremiumBadge label="Base Premium" value={attr.basePremium} />
            <PremiumBadge label="Annual Premium" value={attr.finalPremium} accent />
            <div style={{ background: LTGRAY, borderRadius: 10, padding: "14px 18px", textAlign: "center" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: TXT2, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>
                Risk Loading
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: attr.finalPremium > attr.basePremium ? ERR : OK }}>
                {attr.finalPremium > attr.basePremium ? "+" : ""}{Math.round((attr.finalPremium / attr.basePremium - 1) * 100)}%
              </div>
            </div>
          </div>
        </Card>

        {/* Waterfall */}
        <Card>
          <SectionTitle>Factor Attribution (log-space contribution)</SectionTitle>
          <div style={{ fontSize: 11, color: TXT2, marginBottom: 16, padding: "8px 12px", background: "#eff6ff", borderRadius: 6, border: "1px solid #dbeafe" }}>
            Each bar shows a factor's % contribution to total log-risk deviation from base. Red = surcharge, green = discount.
          </div>

          {/* Neutral line indicator */}
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 10, color: OK, fontWeight: 600 }}>← DISCOUNT</span>
            <span style={{ fontSize: 10, color: ERR, fontWeight: 600 }}>SURCHARGE →</span>
          </div>

          {attr.items.map((it, i) => (
            <WaterfallBar
              key={i}
              label={it.label}
              factor={it.factor}
              logContrib={it.logContrib}
              pctContrib={it.pctContrib}
            />
          ))}

          <div style={{ marginTop: 16, borderTop: "1px solid #e5e7eb", paddingTop: 12, fontSize: 11, color: TXT2, fontFamily: "monospace" }}>
            Base ${attr.basePremium.toLocaleString()} → adjustments → Final ${attr.finalPremium.toLocaleString()}
          </div>
        </Card>
      </div>
    </div>
  );
}

// ── Tab: Sensitivity ──────────────────────────────────────────────────────────
const FACTOR_LABELS = {
  age:        "Age",
  smoking:    "Smoking Status",
  exercise:   "Exercise Frequency",
  occupation: "Occupation Type",
  region:     "Region",
  tier:       "Coverage Tier",
  n_preexist: "Pre-existing Conditions",
};

function SensitivityTab({ profile, setP, coeff }) {
  const [factor, setFactor] = useState("smoking");

  const data     = useMemo(() => computeSensitivity(profile, coeff, factor), [profile, coeff, factor]);
  const maxPrem  = data.reduce((m, d) => Math.max(m, d.premium), 0);
  const current  = calcGLM(profile, coeff);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <ProfilePanel profile={profile} setP={setP} />

        {/* Factor selector */}
        <Card>
          <Field label="Vary this factor">
            <select value={factor} onChange={e => setFactor(e.target.value)} style={inputStyle}>
              {Object.entries(FACTOR_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </Field>
          <div style={{ fontSize: 12, color: TXT2, marginTop: 4 }}>
            All other inputs are held at the values you set on the left. The highlighted bar (▶) is your current selection.
          </div>
        </Card>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Current premium */}
        <Card>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <PremiumBadge label="Current Annual Premium" value={current.premium} accent />
            <PremiumBadge label="Monthly" value={Math.round(current.premium / 12)} />
          </div>
        </Card>

        {/* Sensitivity chart */}
        <Card>
          <SectionTitle>Premium by {FACTOR_LABELS[factor]}</SectionTitle>
          <div style={{ fontSize: 11, color: TXT2, marginBottom: 16, padding: "8px 12px", background: "#f0fdf4", borderRadius: 6, border: "1px solid #bbf7d0" }}>
            Min: <strong>${Math.min(...data.map(d => d.premium)).toLocaleString()}</strong> &nbsp;|&nbsp;
            Max: <strong>${maxPrem.toLocaleString()}</strong> &nbsp;|&nbsp;
            Range: <strong>${(maxPrem - Math.min(...data.map(d => d.premium))).toLocaleString()}</strong>
          </div>
          {data.map((d, i) => (
            <SensBar
              key={i}
              label={d.level}
              premium={d.premium}
              maxPremium={maxPrem}
              isCurrent={d.current}
            />
          ))}
        </Card>
      </div>
    </div>
  );
}

// ── Tab: Assumptions Editor ───────────────────────────────────────────────────
function AssumptionsTab({ profile, coeff, setCoeff, isDirty, resetCoeff }) {
  const current = useMemo(() => calcGLM(profile, coeff), [profile, coeff]);
  const baseline = useMemo(() => calcGLM(profile, DEFAULT_COEFF), [profile]);

  const setFactor = (group, key, val) => {
    setCoeff(prev => ({
      ...prev,
      [group]: { ...prev[group], [key]: val },
    }));
  };

  const setScalar = (key, val) => {
    setCoeff(prev => ({ ...prev, [key]: val }));
  };

  const GROUPS = [
    { key: "age_factors",        label: "Age Factors",        hint: "Frequency multiplier per age band" },
    { key: "smoking_factors",    label: "Smoking Factors",    hint: "Frequency multiplier" },
    { key: "exercise_factors",   label: "Exercise Factors",   hint: "Frequency multiplier" },
    { key: "occupation_factors", label: "Occupation Factors", hint: "Frequency multiplier" },
    { key: "region_factors",     label: "Region Factors",     hint: "Severity multiplier (healthcare cost)" },
    { key: "tier_factors",       label: "Tier Factors",       hint: "Applied to expected annual cost" },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24, alignItems: "start" }}>

      {/* Left: live premium impact */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <Card>
          <SectionTitle>Live Impact (Current Profile)</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
            <PremiumBadge label="Baseline (v2.3)" value={baseline.premium} />
            <PremiumBadge label="With Edits" value={current.premium} accent />
          </div>
          <div style={{
            padding: "12px 14px", borderRadius: 8, marginBottom: 12,
            background: current.premium > baseline.premium ? "#fef2f2" : "#ecfdf5",
            border: `1px solid ${current.premium > baseline.premium ? "#fecaca" : "#a7f3d0"}`,
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: current.premium > baseline.premium ? ERR : OK }}>
              {current.premium > baseline.premium ? "▲" : "▼"} ${Math.abs(current.premium - baseline.premium).toLocaleString()} (
              {((current.premium / baseline.premium - 1) * 100).toFixed(1)}%) vs baseline
            </div>
          </div>
          {isDirty && (
            <button
              onClick={resetCoeff}
              style={{
                width: "100%", padding: "9px", borderRadius: 8, border: `1.5px solid ${NAVY}`,
                background: WHITE, color: NAVY, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
              }}
            >
              Reset to v2.3 defaults
            </button>
          )}
          {!isDirty && (
            <div style={{ fontSize: 11, color: TXT2, textAlign: "center", padding: "8px 0" }}>No changes from v2.3 defaults</div>
          )}
        </Card>

        {/* Scalars */}
        <Card>
          <SectionTitle>Scalar Assumptions</SectionTitle>
          {[
            { key: "preexist_per_condition", label: "Pre-existing (per condition)", hint: "Additive frequency load per condition" },
            { key: "sev_age_gradient",       label: "Severity age gradient",        hint: "Per year above age 30" },
            { key: "load_factor",            label: "Expense & profit loading",      hint: "Applied to expected cost" },
          ].map(({ key, label, hint }) => (
            <div key={key} style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: TXT2, marginBottom: 2 }}>{label}</div>
              <div style={{ fontSize: 10, color: "#9ca3af", marginBottom: 4 }}>{hint}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input
                  type="number" step="0.005" min="0" max="2"
                  value={coeff[key]}
                  onChange={e => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 0) setScalar(key, v);
                  }}
                  style={{ width: 80, padding: "5px 8px", borderRadius: 6, border: "1.5px solid #d1d5db", fontSize: 13, fontFamily: "monospace", outline: "none", textAlign: "right" }}
                />
                {coeff[key] !== DEFAULT_COEFF[key] && (
                  <span style={{ fontSize: 10, color: WARN }}>was {DEFAULT_COEFF[key]}</span>
                )}
              </div>
            </div>
          ))}
        </Card>

        {/* Base freq/sev */}
        <Card>
          <SectionTitle>Base Frequency & Severity</SectionTitle>
          {["ipd", "opd", "dental", "maternity"].map(cov => (
            <div key={cov} style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: NAVY, marginBottom: 6, textTransform: "uppercase" }}>{cov}</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 10, color: TXT2, marginBottom: 3 }}>Base Freq</div>
                  <input type="number" step="0.01" min="0" value={coeff.base_freq[cov]}
                    onChange={e => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v) && v > 0) setCoeff(prev => ({ ...prev, base_freq: { ...prev.base_freq, [cov]: v } }));
                    }}
                    style={{ width: "100%", padding: "5px 8px", borderRadius: 6, border: "1.5px solid #d1d5db", fontSize: 12, fontFamily: "monospace", outline: "none" }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: TXT2, marginBottom: 3 }}>Base Severity ($)</div>
                  <input type="number" step="10" min="0" value={coeff.base_sev[cov]}
                    onChange={e => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v) && v > 0) setCoeff(prev => ({ ...prev, base_sev: { ...prev.base_sev, [cov]: v } }));
                    }}
                    style={{ width: "100%", padding: "5px 8px", borderRadius: 6, border: "1.5px solid #d1d5db", fontSize: 12, fontFamily: "monospace", outline: "none" }}
                  />
                </div>
              </div>
            </div>
          ))}
        </Card>
      </div>

      {/* Right: factor tables */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {GROUPS.map(({ key, label, hint }) => (
          <Card key={key}>
            <SectionTitle>{label}</SectionTitle>
            <div style={{ fontSize: 11, color: TXT2, marginBottom: 12 }}>{hint}</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${NAVY}` }}>
                  <th style={{ textAlign: "left", padding: "6px 0", color: TXT2, fontWeight: 600 }}>Level</th>
                  <th style={{ textAlign: "center", padding: "6px 8px", color: TXT2, fontWeight: 600 }}>Multiplier</th>
                  <th style={{ textAlign: "center", padding: "6px 8px", color: TXT2, fontWeight: 600 }}>vs Default</th>
                  <th style={{ textAlign: "center", padding: "6px 0", color: TXT2, fontWeight: 600 }}>Direction</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(coeff[key]).map(([lvl, val], i) => {
                  const def    = DEFAULT_COEFF[key][lvl];
                  const changed = val !== def;
                  const dir    = val > 1 ? "▲ surcharge" : val < 1 ? "▼ discount" : "— neutral";
                  const dirCol = val > 1 ? ERR : val < 1 ? OK : TXT2;
                  return (
                    <tr key={lvl} style={{ borderBottom: "1px solid #f0f0f0", background: i % 2 ? LTGRAY : WHITE }}>
                      <td style={{ padding: "8px 0", color: TXT }}>{lvl}</td>
                      <td style={{ padding: "8px 8px", textAlign: "center" }}>
                        <AssumptionInput
                          value={val}
                          onChange={v => setFactor(key, lvl, v)}
                        />
                      </td>
                      <td style={{ padding: "8px 8px", textAlign: "center", fontSize: 11, color: changed ? WARN : TXT2, fontFamily: "monospace" }}>
                        {changed ? `was ${def}` : "—"}
                      </td>
                      <td style={{ padding: "8px 0", textAlign: "center", fontSize: 11, fontWeight: 600, color: dirCol }}>
                        {dir}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ── Vietnam ML: constants ─────────────────────────────────────────────────────
const VN_API = "https://dac-healthprice-api.onrender.com";

const VN_REGIONS = ["Ho Chi Minh City", "Hanoi", "Da Nang", "Can Tho", "Hai Phong", "Rural Areas"];
const VN_OCCUPATIONS = ["Office/Desk", "Retail/Service", "Healthcare", "Manual Labor", "Industrial/High-Risk", "Retired"];
const VN_CONDITIONS = ["Hypertension", "Diabetes", "Heart Disease", "COPD/Asthma", "Arthritis"];
const VN_SMOKING    = ["Never", "Former", "Current"];
const VN_EXERCISE   = ["Sedentary", "Light", "Moderate", "Active"];
const VN_ALCOHOL    = ["Never", "Occasional", "Regular", "Heavy"];
const VN_MARITAL    = ["Single", "Married", "Divorced", "Widowed"];
const VN_DIET       = ["Healthy", "Balanced", "High Processed"];
const VN_SLEEP      = ["Good (7-9h)", "Fair (5-7h)", "Poor (<5h)"];
const VN_STRESS     = ["Low", "Moderate", "High"];
const VN_MOTORBIKE  = ["No", "Occasional", "Daily"];
const VN_WATER      = ["Piped/Safe", "Well/Mixed", "Limited"];
const VN_PROXIMITY  = ["<5km", "5-20km", ">20km"];

const DEFAULT_VN_PROFILE = {
  age: 35, bmi: 22.0,
  gender: "Male",
  smoking_status: "Never",
  exercise_frequency: "Moderate",
  has_family_history: false,
  monthly_income_millions_vnd: 15.0,
  region: "Ho Chi Minh City",
  occupation: "Office/Desk",
  pre_existing_conditions: [],
  alcohol: "Never",
  prev_hospitalizations: 0,
  medications_count: 0,
  family_history: [],
  marital_status: "Single",
  diet: "Balanced",
  sleep_quality: "Good (7-9h)",
  stress_level: "Low",
  motorbike_daily: "No",
  water_access: "Piped/Safe",
  healthcare_proximity: "<5km",
  family_size: 1,
};

function ToggleField({ label, value, onChange, onLabel = "Yes", offLabel = "No" }) {
  return (
    <Field label={label}>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={() => onChange(1)}
          style={{
            flex: 1, padding: "7px 0", borderRadius: 7, fontSize: 12, fontWeight: 600,
            border: `1.5px solid ${value ? NAVY : "#e5e7eb"}`,
            background: value ? NAVY : WHITE, color: value ? WHITE : TXT2,
            cursor: "pointer", fontFamily: "inherit",
          }}
        >{onLabel}</button>
        <button
          onClick={() => onChange(0)}
          style={{
            flex: 1, padding: "7px 0", borderRadius: 7, fontSize: 12, fontWeight: 600,
            border: `1.5px solid ${!value ? ERR : "#e5e7eb"}`,
            background: !value ? "#fef2f2" : WHITE, color: !value ? ERR : TXT2,
            cursor: "pointer", fontFamily: "inherit",
          }}
        >{offLabel}</button>
      </div>
    </Field>
  );
}

// SHAP waterfall bar — scales relative to maxAbs in the group
function ShapBar({ feature, shap_value, direction, maxAbs }) {
  const isRisk   = direction === "increases_risk";
  const color    = isRisk ? ERR : OK;
  const bg       = isRisk ? "#fef2f2" : "#ecfdf5";
  const barWidth = maxAbs > 0 ? Math.min((Math.abs(shap_value) / maxAbs) * 100, 100) : 0;

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: TXT, fontWeight: 500 }}>{feature}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, fontFamily: "monospace", color: TXT2 }}>
            {shap_value > 0 ? "+" : ""}{shap_value.toFixed(4)}
          </span>
          <span style={{
            fontSize: 11, fontWeight: 600, color, background: bg,
            borderRadius: 4, padding: "1px 7px",
          }}>
            {isRisk ? "risk +" : "risk -"}
          </span>
        </div>
      </div>
      <div style={{ height: 7, background: "#f3f4f6", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${barWidth}%`,
          background: color, borderRadius: 4,
          transition: "width 0.35s ease",
        }} />
      </div>
    </div>
  );
}

function ModelPanel({ title, data, accent }) {
  if (!data) return null;
  const r2H = data.r2_health     != null ? (data.r2_health * 100).toFixed(1)     : "—";
  const r2M = data.r2_mortality  != null ? (data.r2_mortality * 100).toFixed(1)  : "—";

  return (
    <Card style={{
      border: accent ? `2px solid ${NAVY}` : "1.5px solid #e5e7eb",
      background: accent ? "#f8faff" : WHITE,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: NAVY }}>{title}</div>
          <div style={{ fontSize: 10, color: TXT2, marginTop: 2, maxWidth: 220 }}>{data.method}</div>
        </div>
        {accent && (
          <span style={{
            background: GOLD, color: "#7c2d12", borderRadius: 20,
            padding: "2px 10px", fontSize: 10, fontWeight: 700, whiteSpace: "nowrap",
          }}>
            HIGHER ACCURACY
          </span>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
        <div style={{ background: accent ? NAVY : LTGRAY, borderRadius: 10, padding: "12px 14px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: accent ? "rgba(255,255,255,0.6)" : TXT2, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>
            Health Score
          </div>
          <div style={{ fontSize: 24, fontWeight: 700, color: accent ? WHITE : NAVY }}>
            {data.health_score}<span style={{ fontSize: 13, fontWeight: 400, opacity: 0.7 }}>/100</span>
          </div>
        </div>
        <div style={{ background: LTGRAY, borderRadius: 10, padding: "12px 14px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: TXT2, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>
            Mortality Mult.
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: NAVY }}>
            x{data.mortality_multiplier}
          </div>
        </div>
      </div>

      <div style={{
        background: "#f3f4f6", borderRadius: 8, padding: "10px 12px",
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8,
      }}>
        {[
          ["R² Health", `${r2H}%`],
          ["R² Mortality", `${r2M}%`],
          ["RMSE Health", data.rmse_health?.toFixed(2) ?? "—"],
          ["RMSE Mortality", data.rmse_mortality?.toFixed(3) ?? "—"],
        ].map(([lbl, val]) => (
          <div key={lbl}>
            <div style={{ fontSize: 10, color: TXT2, marginBottom: 1 }}>{lbl}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: NAVY }}>{val}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ── Tab: Vietnam ML ───────────────────────────────────────────────────────────
function computeQuote(result, model, coverageYears) {
  const data = model === "glm" ? result.glm : result.xgboost;
  const healthScore = data.health_score ?? 70;
  const mortalityMult = data.mortality_multiplier ?? 1.0;
  const BASE_MONTHLY = 20;
  const monthly = Math.round(BASE_MONTHLY * mortalityMult * (1 + (100 - healthScore) / 200));
  const annual = monthly * 12;
  const total = annual * coverageYears;
  const ref = `VN-${model.toUpperCase()}-${Math.random().toString(36).toUpperCase().slice(2, 8)}`;
  const rationale = model === "glm"
    ? "GLM provides transparent, auditable coefficients — preferred when regulatory explainability is required."
    : "XGBoost captures non-linear interactions between risk factors, achieving higher predictive accuracy.";
  const keyRisk = (result.xgboost.shap_health_top3 || []).map(s => ({
    feature: s.feature,
    direction: s.direction === "increases_risk" ? "increases risk" : "reduces risk",
  }));
  return {
    policy_reference: ref,
    premium: { annual_premium_usd: annual, monthly_premium_usd: monthly, total_policy_cost_usd: total },
    model: {
      name: model === "glm" ? "GLM (Gamma + Poisson)" : "XGBoost",
      r2_health: data.r2_health,
      r2_mortality: data.r2_mortality,
      rationale,
    },
    key_risk_factors: keyRisk,
  };
}

function VietnamTab() {
  const [vp, setVP]               = useState(DEFAULT_VN_PROFILE);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [activeShap, setActiveShap] = useState("health");

  const [selectedModel, setSelectedModel] = useState(null);
  const [coverageYears, setCoverageYears] = useState(10);

  const confirmResult = result && selectedModel ? computeQuote(result, selectedModel, coverageYears) : null;

  const setF = (k, v) => setVP(p => ({ ...p, [k]: v }));

  const toggleCondition = (cond) => {
    setVP(p => {
      const existing = p.pre_existing_conditions || [];
      return {
        ...p,
        pre_existing_conditions: existing.includes(cond)
          ? existing.filter(c => c !== cond)
          : [...existing, cond],
      };
    });
  };

  const runPricing = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedModel(null);
    try {
      // Map frontend UX fields to backend API contract (backend uses int flags, not strings)
      const payload = {
        age: vp.age,
        bmi: vp.bmi,
        is_smoking: vp.smoking_status === "Current" ? 1 : 0,
        is_exercise: ["Moderate", "Active"].includes(vp.exercise_frequency) ? 1 : 0,
        has_family_history: vp.has_family_history ? 1 : 0,
        monthly_income_millions_vnd: vp.monthly_income_millions_vnd,
        region: vp.region,
        occupation: vp.occupation,
        pre_existing_conditions: vp.pre_existing_conditions || [],
      };
      const r = await fetch(`${VN_API}/api/vietnam/price`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      setResult(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const selectModel = (model) => {
    setSelectedModel(prev => prev === model ? null : model);
  };

  const sel = { ...inputStyle, fontSize: 13 };

  const shapData = result
    ? (activeShap === "health"
        ? result.xgboost.shap_health_top3
        : result.xgboost.shap_mortality_top3)
    : [];
  const maxAbs = shapData.length
    ? Math.max(...shapData.map(s => Math.abs(s.shap_value)), 0.0001)
    : 0.0001;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24, alignItems: "start" }}>

      {/* ── Left: input form ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Card>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
            <span style={{
              background: "#dc2626", color: WHITE, borderRadius: 6,
              padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
            }}>VN</span>
            <div style={{ fontSize: 14, fontWeight: 700, color: NAVY }}>Vietnam Applicant Profile</div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 14px" }}>
            <Field label="Age">
              <input type="number" min={18} max={80} value={vp.age}
                onChange={e => setF("age", Number(e.target.value))} style={inputStyle} />
            </Field>
            <Field label="BMI">
              <input type="number" min={14} max={50} step={0.1} value={vp.bmi}
                onChange={e => setF("bmi", parseFloat(e.target.value) || 22)} style={inputStyle} />
            </Field>
            <Field label="Gender">
              <select value={vp.gender} onChange={e => setF("gender", e.target.value)} style={sel}>
                <option>Male</option><option>Female</option>
              </select>
            </Field>
            <Field label="Monthly Income (M VND)">
              <input type="number" min={0} step={1} value={vp.monthly_income_millions_vnd}
                onChange={e => setF("monthly_income_millions_vnd", parseFloat(e.target.value) || 0)}
                style={inputStyle} />
            </Field>
          </div>

          <Field label="Region">
            <select value={vp.region} onChange={e => setF("region", e.target.value)} style={sel}>
              {VN_REGIONS.map(r => <option key={r}>{r}</option>)}
            </select>
          </Field>
          <Field label="Occupation">
            <select value={vp.occupation} onChange={e => setF("occupation", e.target.value)} style={sel}>
              {VN_OCCUPATIONS.map(o => <option key={o}>{o}</option>)}
            </select>
          </Field>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 14px" }}>
            <Field label="Smoking Status">
              <select value={vp.smoking_status} onChange={e => setF("smoking_status", e.target.value)} style={sel}>
                {VN_SMOKING.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Exercise Level">
              <select value={vp.exercise_frequency} onChange={e => setF("exercise_frequency", e.target.value)} style={sel}>
                {VN_EXERCISE.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Alcohol Use">
              <select value={vp.alcohol} onChange={e => setF("alcohol", e.target.value)} style={sel}>
                {VN_ALCOHOL.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Marital Status">
              <select value={vp.marital_status} onChange={e => setF("marital_status", e.target.value)} style={sel}>
                {VN_MARITAL.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
          </div>

          <ToggleField label="Family History of CHD" value={vp.has_family_history} onChange={v => setF("has_family_history", v)} />
        </Card>

        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: NAVY, marginBottom: 14 }}>Lifestyle & Clinical Factors</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 14px" }}>
            <Field label="Diet">
              <select value={vp.diet} onChange={e => setF("diet", e.target.value)} style={sel}>
                {VN_DIET.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Sleep Quality">
              <select value={vp.sleep_quality} onChange={e => setF("sleep_quality", e.target.value)} style={sel}>
                {VN_SLEEP.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Stress Level">
              <select value={vp.stress_level} onChange={e => setF("stress_level", e.target.value)} style={sel}>
                {VN_STRESS.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Motorbike Use">
              <select value={vp.motorbike_daily} onChange={e => setF("motorbike_daily", e.target.value)} style={sel}>
                {VN_MOTORBIKE.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Water Access">
              <select value={vp.water_access} onChange={e => setF("water_access", e.target.value)} style={sel}>
                {VN_WATER.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Hospital Proximity">
              <select value={vp.healthcare_proximity} onChange={e => setF("healthcare_proximity", e.target.value)} style={sel}>
                {VN_PROXIMITY.map(s => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Prior Hospitalizations">
              <input type="number" min={0} max={10} value={vp.prev_hospitalizations}
                onChange={e => setF("prev_hospitalizations", Number(e.target.value))} style={inputStyle} />
            </Field>
            <Field label="Medications Count">
              <input type="number" min={0} max={20} value={vp.medications_count}
                onChange={e => setF("medications_count", Number(e.target.value))} style={inputStyle} />
            </Field>
          </div>
          <Field label="Family Size">
            <input type="number" min={1} max={10} value={vp.family_size}
              onChange={e => setF("family_size", Number(e.target.value))} style={inputStyle} />
          </Field>
        </Card>

        <Card>
          <SectionTitle>Pre-existing Conditions</SectionTitle>
          {VN_CONDITIONS.map(c => {
            const checked = (vp.pre_existing_conditions || []).includes(c);
            return (
              <div
                key={c}
                onClick={() => toggleCondition(c)}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "8px 10px", marginBottom: 6, borderRadius: 8,
                  border: `1.5px solid ${checked ? NAVY : "#e5e7eb"}`,
                  background: checked ? "#eff6ff" : WHITE,
                  cursor: "pointer", fontSize: 13,
                  color: checked ? NAVY : TXT, fontWeight: checked ? 600 : 400,
                }}
              >
                <span style={{
                  width: 16, height: 16, borderRadius: 4, flexShrink: 0,
                  border: `2px solid ${checked ? NAVY : "#d1d5db"}`,
                  background: checked ? NAVY : WHITE,
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  {checked && <span style={{ color: WHITE, fontSize: 10, lineHeight: 1 }}>x</span>}
                </span>
                {c}
              </div>
            );
          })}
        </Card>

        <button
          onClick={runPricing}
          disabled={loading}
          style={{
            width: "100%", padding: "12px", borderRadius: 10,
            background: loading ? "#6b7280" : NAVY, color: WHITE,
            border: "none", fontSize: 14, fontWeight: 700,
            cursor: loading ? "not-allowed" : "pointer", fontFamily: "inherit",
          }}
        >
          {loading ? "Running..." : "Run Dual Pricing"}
        </button>

        {loading && (
          <div style={{ fontSize: 11, color: TXT2, textAlign: "center", padding: "4px 0" }}>
            First call wakes the Render server (~30s). Subsequent calls are fast.
          </div>
        )}

        {error && (
          <div style={{ padding: "10px 14px", borderRadius: 8, background: "#fef2f2", border: "1px solid #fecaca", color: ERR, fontSize: 12 }}>
            {error}
          </div>
        )}
      </div>

      {/* ── Right: results ── */}
      <div>
        {!result && !loading && (
          <Card style={{ textAlign: "center", padding: 56 }}>
            <div style={{ fontSize: 16, fontWeight: 600, color: NAVY, marginBottom: 8 }}>
              GLM vs XGBoost — Side-by-Side Comparison
            </div>
            <div style={{ fontSize: 13, color: TXT2, marginBottom: 20 }}>
              Configure a Vietnam applicant and click "Run Dual Pricing" to compare models and see SHAP risk drivers.
            </div>
            <div style={{ display: "inline-flex", gap: 24, fontSize: 12, color: TXT2 }}>
              <span>Health score (0-100)</span>
              <span>Mortality multiplier</span>
              <span>R² / RMSE metrics</span>
              <span>SHAP top-3 drivers</span>
            </div>
          </Card>
        )}

        {loading && (
          <Card style={{ textAlign: "center", padding: 56 }}>
            <div style={{
              width: 40, height: 40, borderRadius: "50%",
              border: `4px solid #e5e7eb`, borderTopColor: NAVY,
              margin: "0 auto 16px",
              animation: "spin 0.8s linear infinite",
            }} />
            <div style={{ fontSize: 14, fontWeight: 600, color: NAVY, marginBottom: 4 }}>Pricing both models...</div>
            <div style={{ fontSize: 12, color: TXT2 }}>GLM inference + XGBoost + SHAP TreeExplainer</div>
          </Card>
        )}

        {result && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* R² gain summary banner */}
            <Card style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", padding: "16px 24px" }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: TXT2, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 12 }}>
                XGBoost vs GLM — Accuracy Gain
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 16, textAlign: "center" }}>
                {[
                  ["R² Gain (Health)", `+${(result.comparison.xgb_r2_gain_health * 100).toFixed(1)}pp`, OK],
                  ["R² Gain (Mortality)", `+${(result.comparison.xgb_r2_gain_mortality * 100).toFixed(1)}pp`, OK],
                  ["Health Score Delta", result.comparison.health_score_diff.toFixed(1), NAVY],
                  ["Mortality Delta", result.comparison.mortality_diff.toFixed(4), NAVY],
                ].map(([lbl, val, col]) => (
                  <div key={lbl}>
                    <div style={{ fontSize: 10, color: TXT2, marginBottom: 4 }}>{lbl}</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: col }}>{val}</div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Side-by-side model panels */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <ModelPanel title="GLM" data={result.glm} accent={false} />
              <ModelPanel title="XGBoost" data={result.xgboost} accent={true} />
            </div>

            {/* Model selection */}
            <Card style={{ border: "1.5px solid #e0e7ff", background: "#f8faff" }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: NAVY, marginBottom: 4 }}>
                Generate Final Quote
              </div>
              <div style={{ fontSize: 12, color: TXT2, marginBottom: 14 }}>
                Choose a model to compute this applicant's premium. The quote below updates instantly.
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <label style={{ fontSize: 12, color: TXT2, whiteSpace: "nowrap" }}>Coverage Years</label>
                <input
                  type="number" min={1} max={30} value={coverageYears}
                  onChange={e => setCoverageYears(Number(e.target.value) || 10)}
                  style={{ ...sel, width: 70 }}
                />
              </div>
              <div style={{ display: "flex", gap: 12 }}>
                <button onClick={() => selectModel("glm")} style={{
                  flex: 1, padding: "10px 0", borderRadius: 8, fontSize: 13, fontWeight: 600,
                  border: `1.5px solid ${selectedModel === "glm" ? NAVY : "#d1d5db"}`,
                  background: selectedModel === "glm" ? NAVY : WHITE,
                  color: selectedModel === "glm" ? WHITE : TXT,
                  cursor: "pointer", fontFamily: "inherit",
                }}>Use GLM</button>
                <button onClick={() => selectModel("xgboost")} style={{
                  flex: 1, padding: "10px 0", borderRadius: 8, fontSize: 13, fontWeight: 600,
                  border: `1.5px solid ${selectedModel === "xgboost" ? "#16a34a" : "#d1d5db"}`,
                  background: selectedModel === "xgboost" ? "#16a34a" : WHITE,
                  color: selectedModel === "xgboost" ? WHITE : TXT2,
                  cursor: "pointer", fontFamily: "inherit",
                }}>
                  ★ Use XGBoost <span style={{ fontSize: 10, fontWeight: 400 }}>(Recommended)</span>
                </button>
              </div>
            </Card>

            {/* Confirmed quote */}
            {confirmResult && (
              <Card style={{ border: `2px solid ${selectedModel === "xgboost" ? "#16a34a" : NAVY}`, background: "#f0fdf4" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: NAVY, marginBottom: 2 }}>Confirmed Quote</div>
                    <div style={{ fontSize: 11, color: TXT2 }}>Valid for 30 days · Present reference to bind coverage</div>
                  </div>
                  <span style={{
                    background: NAVY, color: WHITE, borderRadius: 6,
                    padding: "4px 12px", fontSize: 11, fontWeight: 700, letterSpacing: 0.5, fontFamily: "monospace",
                  }}>
                    {confirmResult.policy_reference}
                  </span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 14 }}>
                  {[
                    ["Annual Premium", `$${confirmResult.premium.annual_premium_usd.toLocaleString()}`],
                    ["Monthly Premium", `$${confirmResult.premium.monthly_premium_usd.toLocaleString()}`],
                    [`Total (${confirmResult.coverage_years}yr)`, `$${confirmResult.premium.total_policy_cost_usd.toLocaleString()}`],
                  ].map(([lbl, val]) => (
                    <div key={lbl} style={{ background: WHITE, borderRadius: 8, padding: "10px 14px", textAlign: "center" }}>
                      <div style={{ fontSize: 10, color: TXT2, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.4 }}>{lbl}</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: NAVY }}>{val}</div>
                    </div>
                  ))}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 12 }}>
                  <div>
                    <div style={{ color: TXT2, marginBottom: 4 }}>Model Used</div>
                    <div style={{ fontWeight: 600, color: NAVY }}>{confirmResult.model.name}</div>
                    <div style={{ color: TXT2, fontSize: 11, marginTop: 2 }}>R² Health: {(confirmResult.model.r2_health * 100).toFixed(1)}% · R² Mortality: {(confirmResult.model.r2_mortality * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div style={{ color: TXT2, marginBottom: 4 }}>Why this model?</div>
                    <div style={{ color: TXT, fontSize: 11 }}>{confirmResult.model.rationale}</div>
                  </div>
                </div>

                {confirmResult.key_risk_factors?.length > 0 && (
                  <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid #d1fae5" }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: TXT2, marginBottom: 6 }}>Key Risk Factors</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {confirmResult.key_risk_factors.map((f, i) => (
                        <span key={i} style={{
                          padding: "3px 10px", borderRadius: 20, fontSize: 11,
                          background: f.direction === "positive" || f.direction === "increases risk" ? "#fef2f2" : "#f0fdf4",
                          color: f.direction === "positive" || f.direction === "increases risk" ? ERR : OK,
                          border: `1px solid ${f.direction === "positive" || f.direction === "increases risk" ? "#fecaca" : "#bbf7d0"}`,
                          fontWeight: 500,
                        }}>
                          {f.feature || f.factor}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </Card>
            )}

            {/* SHAP waterfall */}
            {shapData.length > 0 && (
              <Card>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                  <SectionTitle>SHAP — Top Risk Drivers (XGBoost)</SectionTitle>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[["health", "Health Score"], ["mortality", "Mortality"]].map(([k, lbl]) => (
                      <button key={k} onClick={() => setActiveShap(k)} style={{
                        padding: "5px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                        border: `1.5px solid ${activeShap === k ? NAVY : "#e5e7eb"}`,
                        background: activeShap === k ? NAVY : WHITE,
                        color: activeShap === k ? WHITE : TXT2,
                        cursor: "pointer", fontFamily: "inherit",
                      }}>{lbl}</button>
                    ))}
                  </div>
                </div>

                <div style={{ fontSize: 11, color: TXT2, marginBottom: 14, padding: "8px 12px", background: "#eff6ff", borderRadius: 6, border: "1px solid #dbeafe" }}>
                  {activeShap === "health"
                    ? "Health score: higher = healthier. Red bar = factor lowers health score (more risk). Green = improves score."
                    : "Mortality multiplier: higher = more risk. Red = factor raises mortality risk. Green = reduces it."}
                </div>

                {shapData.map((s, i) => (
                  <ShapBar key={i} feature={s.feature} shap_value={s.shap_value} direction={s.direction} maxAbs={maxAbs} />
                ))}

                <div style={{ marginTop: 10, fontSize: 10, color: "#9ca3af", fontStyle: "italic" }}>
                  SHAP via TreeExplainer — marginal contribution of each feature to the model output.
                </div>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function ActuarialWorkbench() {
  const [tab, setTab] = useState("attribution");

  const [profile, setProfile] = useState({
    age: 35,
    smoking:    "Never",
    exercise:   "Moderate",
    occupation: "Office/Desk",
    region:     "Phnom Penh",
    n_preexist: 0,
    coverage:   "ipd",
    tier:       "Silver",
  });

  const [coeff, setCoeff]     = useState(() => deepClone(DEFAULT_COEFF));
  const [isDirty, setIsDirty] = useState(false);

  const setP = (k, v) => setProfile(p => ({ ...p, [k]: v }));

  const handleSetCoeff = (fn) => {
    setCoeff(fn);
    setIsDirty(true);
  };

  const resetCoeff = () => {
    setCoeff(deepClone(DEFAULT_COEFF));
    setIsDirty(false);
  };

  const TABS = [
    ["attribution", "Attribution"],
    ["sensitivity",  "Sensitivity"],
    ["assumptions",  "Assumptions Editor"],
    ["vietnam",      "Vietnam ML"],
  ];

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px" }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            {tab === "vietnam" ? (
              <>
                <div style={{ display: "inline-block", background: "rgba(220,38,38,0.1)", border: "1px solid rgba(220,38,38,0.3)", borderRadius: 20, padding: "3px 12px", fontSize: 11, color: "#dc2626", fontWeight: 600, marginBottom: 8 }}>
                  VIETNAM CASE STUDY — ML PRICING
                </div>
                <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, margin: 0 }}>GLM vs XGBoost Comparison</h2>
                <p style={{ color: TXT2, fontSize: 13, marginTop: 4 }}>
                  OLS/Gamma GLM · XGBoost (n=300) · SHAP risk attribution · Vietnam health + life
                </p>
              </>
            ) : (
              <>
                <div style={{ display: "inline-block", background: "rgba(245,166,35,0.12)", border: "1px solid rgba(245,166,35,0.3)", borderRadius: 20, padding: "3px 12px", fontSize: 11, color: GOLD, fontWeight: 600, marginBottom: 8 }}>
                  HEALTH GLM — WORKBENCH
                </div>
                <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, margin: 0 }}>Actuarial Pricing Workbench</h2>
                <p style={{ color: TXT2, fontSize: 13, marginTop: 4 }}>
                  Poisson-Gamma GLM v2.3 · {Object.keys(DEFAULT_COEFF.age_factors).length} age bands · {Object.keys(DEFAULT_COEFF.region_factors).length} regions
                </p>
              </>
            )}
          </div>
          {isDirty && tab !== "vietnam" && (
            <div style={{ background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8, padding: "6px 12px", fontSize: 12, color: "#92400e", fontWeight: 600 }}>
              Unsaved edits — assumptions differ from v2.3
            </div>
          )}
        </div>
      </div>

      {/* Sub-tab bar */}
      <div style={{ background: WHITE, borderRadius: 10, border: "1px solid #e5e7eb", padding: "4px", display: "inline-flex", gap: 4, marginBottom: 24 }}>
        {TABS.map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: "8px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer",
              border: "none", borderRadius: 8,
              background: tab === key ? NAVY : "transparent",
              color: tab === key ? WHITE : TXT2,
              transition: "all 0.2s", fontFamily: "inherit",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "attribution" && <AttributionTab profile={profile} setP={setP} coeff={coeff} />}
      {tab === "sensitivity"  && <SensitivityTab profile={profile} setP={setP} coeff={coeff} />}
      {tab === "assumptions"  && (
        <AssumptionsTab
          profile={profile}
          coeff={coeff}
          setCoeff={handleSetCoeff}
          isDirty={isDirty}
          resetCoeff={resetCoeff}
        />
      )}
      {tab === "vietnam" && <VietnamTab />}

      {/* Audit footer — Cambodia workbench only */}
      {tab !== "vietnam" && (
        <div style={{ marginTop: 32, padding: "12px 16px", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8, fontSize: 11, color: "#92400e" }}>
          <strong>Workbench Note:</strong> Assumptions edits are session-only and do not affect the production model at dac-healthprice-api.onrender.com. To promote revised assumptions to production, update COEFF in <code>main.py</code> and redeploy.
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 900px) {
          div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; }
          div[style*="grid-template-columns: 380px 1fr"] { grid-template-columns: 1fr !important; }
          div[style*="grid-template-columns: 320px 1fr"] { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
