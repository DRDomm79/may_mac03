import { useState, useEffect, useCallback } from 'react';
import { API_URL, authFetch } from './auth';
const NAVY = "#0d2b7a";
const GOLD_D = "#e67e00";
const GRAY = "#94a3b8";
const TXT = "#111827";
const TXT2 = "#4b5563";

const MOCK_CASES = [
  {
    id: "DAC-DEMO0001", status: "submitted", full_name: "Sovann Pich",
    submitted_at: new Date(Date.now() - 2 * 3600000).toISOString(),
    region: "Phnom Penh", occupation: "Office/Desk", date_of_birth: "1980-05-15", gender: "Male",
    phone: "+855 12 345 678", email: "sovann@example.com",
    medical_data: { smokingStatus: "Never", preexistingConditions: ["Hypertension"], exerciseFrequency: "Light",
      height: 170, weight: 80, bloodPressure: "130/85", familyHistory: "Heart Disease",
      currentMedications: "Amlodipine 5mg", alcoholConsumption: "Occasional" },
  },
  {
    id: "DAC-DEMO0002", status: "in_review", full_name: "Channary Kim",
    submitted_at: new Date(Date.now() - 24 * 3600000).toISOString(),
    region: "Siem Reap", occupation: "Healthcare", date_of_birth: "1975-11-28", gender: "Female",
    phone: "+855 92 876 543", email: "channary@example.com",
    medical_data: { smokingStatus: "Current", preexistingConditions: ["Diabetes", "Hypertension"],
      exerciseFrequency: "Sedentary", height: 158, weight: 75, bloodPressure: "145/92",
      familyHistory: "Diabetes, Heart Disease", currentMedications: "Metformin, Lisinopril",
      alcoholConsumption: "Never" },
  },
  {
    id: "DAC-DEMO0003", status: "submitted", full_name: "Dara Meas",
    submitted_at: new Date(Date.now() - 30 * 60000).toISOString(),
    region: "Battambang", occupation: "Retail/Service", date_of_birth: "1995-03-10", gender: "Male",
    phone: "+855 78 234 567", email: "dara@example.com",
    medical_data: { smokingStatus: "Never", preexistingConditions: ["None"], exerciseFrequency: "Active",
      height: 175, weight: 68, bloodPressure: "118/75", familyHistory: "None",
      currentMedications: "None", alcoholConsumption: "Never" },
  },
];

function computeRisk(md, dob) {
  if (!md) return "unknown";
  let score = 0;
  const age = dob ? Math.floor((Date.now() - new Date(dob)) / 31557600000) : 30;
  if (age > 55) score += 3; else if (age > 40) score += 2; else if (age > 30) score += 1;
  if (md.smokingStatus === "Current") score += 3; else if (md.smokingStatus === "Former") score += 1;
  const conds = md.preexistingConditions || [];
  score += conds.filter(c => ["Diabetes","Heart Disease","Cancer (remission)","Kidney Disease"].includes(c)).length * 3;
  score += conds.filter(c => ["Hypertension","Asthma/COPD","Obesity","Mental Health"].includes(c)).length;
  const bmi = md.height && md.weight ? md.weight / Math.pow(md.height / 100, 2) : 0;
  if (bmi > 35) score += 2; else if (bmi > 27) score += 1;
  if (md.exerciseFrequency === "Sedentary") score += 1;
  if (score >= 8) return "decline";
  if (score >= 5) return "high";
  if (score >= 2) return "medium";
  return "low";
}

const RISK_PALETTE = {
  low:     { bg: "#dcfce7", color: "#166534" },
  medium:  { bg: "#fef9c3", color: "#854d0e" },
  high:    { bg: "#fee2e2", color: "#991b1b" },
  decline: { bg: "#f3f4f6", color: "#374151" },
  unknown: { bg: "#f3f4f6", color: "#6b7280" },
};

function RiskBadge({ level }) {
  const s = RISK_PALETTE[level] ?? RISK_PALETTE.unknown;
  return (
    <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 9999, fontSize: 11,
      fontWeight: 700, background: s.bg, color: s.color, textTransform: "uppercase", letterSpacing: 0.5 }}>
      {level}
    </span>
  );
}

function timeAgo(ts) {
  if (!ts) return "—";
  const mins = Math.floor((Date.now() - new Date(ts)) / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function InfoCell({ label, value, bg = "#f8fafc", valueColor }) {
  return (
    <div style={{ background: bg, borderRadius: 8, padding: "10px 12px" }}>
      <div style={{ fontSize: 10, color: GRAY, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: valueColor || TXT }}>{value || "—"}</div>
    </div>
  );
}

export default function UnderwriterDashboard() {
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [usingMock, setUsingMock] = useState(false);
  const [filter, setFilter] = useState("pending");
  const [decision, setDecision] = useState({ outcome: "approved", notes: "", reviewer: "" });
  const [submitting, setSubmitting] = useState(false);
  const [events, setEvents] = useState([]);

  const fetchCases = useCallback(async () => {
    setLoading(true);
    try {
      const statusParam = filter === "pending" ? "submitted,in_review" : filter;
      const r = await authFetch(`${API_URL}/api/v1/applications?status=${statusParam}`, { signal: AbortSignal.timeout(8000) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const enriched = (data.applications || []).map(c => ({
        ...c,
        risk_level: c.risk_level || computeRisk(c.medical_data, c.date_of_birth),
      }));
      setCases(enriched);
      setUsingMock(false);
    } catch {
      const filtered = MOCK_CASES.filter(c =>
        filter === "pending" ? ["submitted", "in_review"].includes(c.status) : c.status === filter
      ).map(c => ({ ...c, risk_level: computeRisk(c.medical_data, c.date_of_birth) }));
      setCases(filtered);
      setUsingMock(true);
    }
    setLoading(false);
  }, [filter]);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  useEffect(() => {
    if (!selected) { setEvents([]); return; }
    setDecision({ outcome: "approved", notes: "", reviewer: "" });
    const fetchEvents = async () => {
      try {
        const r = await fetch(`${API_URL}/api/v1/applications/${selected.id}/status`, { signal: AbortSignal.timeout(8000) });
        if (!r.ok) throw new Error();
        const data = await r.json();
        setEvents(data.timeline || []);
      } catch {
        setEvents([
          { event: "Application received", done: true, timestamp: selected.submitted_at },
          { event: "Initial review", done: selected.status === "in_review", timestamp: null },
          { event: "Underwriter decision", done: ["approved","declined","referred"].includes(selected.status), timestamp: null },
          { event: "Policy issued", done: false, timestamp: null },
        ]);
      }
    };
    fetchEvents();
  }, [selected]);

  async function submitDecision() {
    if (!selected || !decision.reviewer.trim()) return;
    setSubmitting(true);
    try {
      if (!usingMock) {
        const r = await authFetch(`${API_URL}/api/v1/applications/${selected.id}/decision`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ outcome: decision.outcome, notes: decision.notes, reviewer_id: decision.reviewer }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
      }
      setCases(prev => prev.map(c => c.id === selected.id ? { ...c, status: decision.outcome } : c));
      setSelected(null);
      setTimeout(fetchCases, 300);
    } catch (e) {
      alert(`Decision failed: ${e.message}`);
    }
    setSubmitting(false);
  }

  const sc = cases.find(c => c.id === selected?.id) || selected;
  const md = sc?.medical_data || {};
  const age = sc?.date_of_birth ? Math.floor((Date.now() - new Date(sc.date_of_birth)) / 31557600000) : "?";
  const bmi = md.height && md.weight ? (md.weight / Math.pow(md.height / 100, 2)).toFixed(1) : "—";
  const conditions = (md.preexistingConditions || []).filter(c => c !== "None").join(", ") || "None";
  const isPending = sc && ["submitted", "in_review"].includes(sc.status);

  return (
    <div style={{ paddingTop: 88, paddingBottom: 60, minHeight: "100vh", background: "#f8fafc", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ maxWidth: 1320, margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: 24 }}>
          <span style={{ color: GOLD_D, fontSize: 13, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase" }}>Internal tool</span>
          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 32, fontWeight: 700, color: NAVY, marginTop: 4, marginBottom: 0 }}>
            Underwriter Review Dashboard
          </h1>
          {usingMock && (
            <div style={{ marginTop: 10, padding: "6px 14px", background: "#fffbeb", border: "1px solid #fcd34d",
              borderRadius: 6, display: "inline-block", fontSize: 12, color: "#92400e" }}>
              Demo mode — backend unavailable. Showing sample cases.
            </div>
          )}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20, alignItems: "start" }}>
          {/* ── Queue panel ── */}
          <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)", overflow: "hidden" }}>
            <div style={{ padding: "14px 16px", borderBottom: "1px solid #e5e7eb", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: NAVY }}>Cases</span>
              <button onClick={fetchCases} style={{ fontSize: 11, padding: "3px 10px", border: "1px solid #d1d5db", borderRadius: 4, background: "#f9fafb", cursor: "pointer", fontFamily: "inherit" }}>
                ↻ Refresh
              </button>
            </div>
            <div style={{ display: "flex", padding: "8px 10px", gap: 6, borderBottom: "1px solid #e5e7eb", background: "#fafafa" }}>
              {[["pending","Pending"],["approved","Approved"],["declined","Declined"],["referred","Referred"]].map(([v, l]) => (
                <button key={v} onClick={() => { setFilter(v); setSelected(null); }}
                  style={{ flex: 1, padding: "5px 0", borderRadius: 5, border: "none", fontSize: 11, fontWeight: 600,
                    cursor: "pointer", fontFamily: "inherit", background: filter === v ? NAVY : "transparent",
                    color: filter === v ? "#fff" : TXT2 }}>
                  {l}
                </button>
              ))}
            </div>
            <div style={{ maxHeight: 560, overflowY: "auto" }}>
              {loading ? (
                <div style={{ padding: 32, textAlign: "center", color: GRAY, fontSize: 13 }}>Loading…</div>
              ) : cases.length === 0 ? (
                <div style={{ padding: 32, textAlign: "center", color: GRAY, fontSize: 13 }}>No cases.</div>
              ) : (
                cases.map(c => {
                  const active = selected?.id === c.id;
                  return (
                    <div key={c.id} onClick={() => setSelected(c)}
                      style={{ padding: "12px 14px", borderBottom: "1px solid #f1f3f5", cursor: "pointer",
                        background: active ? "#eff6ff" : "#fff", borderLeft: active ? `3px solid ${NAVY}` : "3px solid transparent",
                        transition: "background 0.1s" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                        <span style={{ fontWeight: 700, fontSize: 13, color: active ? NAVY : TXT }}>{c.full_name || "—"}</span>
                        <RiskBadge level={c.risk_level || "unknown"} />
                      </div>
                      <div style={{ fontSize: 11, color: TXT2, display: "flex", justifyContent: "space-between" }}>
                        <span style={{ fontFamily: "monospace" }}>{c.id}</span>
                        <span>{timeAgo(c.submitted_at)}</span>
                      </div>
                      <div style={{ fontSize: 11, color: GRAY, marginTop: 2 }}>{c.region} · {c.occupation}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* ── Detail panel ── */}
          {!sc ? (
            <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
              padding: 48, textAlign: "center", color: GRAY }}>
              <div style={{ fontSize: 44, marginBottom: 12 }}>📋</div>
              <p style={{ fontWeight: 600, fontSize: 16, color: TXT2, marginBottom: 6 }}>Select a case to review</p>
              <p style={{ fontSize: 13 }}>Click any case in the queue to view applicant details and record a decision.</p>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Case header */}
              <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)", padding: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: GOLD_D, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>Case Reference</div>
                    <div style={{ fontFamily: "monospace", fontSize: 17, fontWeight: 700, color: NAVY }}>{sc.id}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ marginBottom: 4 }}><RiskBadge level={sc.risk_level || "unknown"} /></div>
                    <div style={{ fontSize: 10, color: GRAY }}>Preliminary risk screen</div>
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                  <InfoCell label="Applicant" value={sc.full_name} />
                  <InfoCell label="Age / Gender" value={`${age} / ${sc.gender || "—"}`} />
                  <InfoCell label="Region" value={sc.region} />
                  <InfoCell label="Submitted" value={timeAgo(sc.submitted_at)} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginTop: 10 }}>
                  <InfoCell label="Occupation" value={sc.occupation} />
                  <InfoCell label="Phone" value={sc.phone} />
                  <InfoCell label="Email" value={sc.email} />
                </div>
              </div>

              {/* Medical assessment */}
              <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)", padding: 20 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: NAVY, marginBottom: 14 }}>Medical Assessment</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 10 }}>
                  <InfoCell label="BMI" value={bmi !== "—" ? `${bmi} (${md.height}cm / ${md.weight}kg)` : "—"}
                    bg={bmi !== "—" && Number(bmi) > 30 ? "#fef2f2" : bmi !== "—" && Number(bmi) < 18.5 ? "#fffbeb" : "#f0fdf4"}
                    valueColor={bmi !== "—" && Number(bmi) > 30 ? "#991b1b" : undefined} />
                  <InfoCell label="Smoking" value={md.smokingStatus}
                    bg={md.smokingStatus === "Current" ? "#fef2f2" : md.smokingStatus === "Former" ? "#fffbeb" : "#f0fdf4"}
                    valueColor={md.smokingStatus === "Current" ? "#991b1b" : md.smokingStatus === "Former" ? "#854d0e" : "#166534"} />
                  <InfoCell label="Exercise" value={md.exerciseFrequency}
                    bg={md.exerciseFrequency === "Sedentary" ? "#fef2f2" : md.exerciseFrequency === "Active" ? "#f0fdf4" : "#f8fafc"}
                    valueColor={md.exerciseFrequency === "Sedentary" ? "#991b1b" : md.exerciseFrequency === "Active" ? "#166534" : undefined} />
                  <InfoCell label="Blood Pressure" value={md.bloodPressure} />
                  <InfoCell label="Alcohol" value={md.alcoholConsumption}
                    bg={md.alcoholConsumption === "Heavy" ? "#fef2f2" : "#f8fafc"}
                    valueColor={md.alcoholConsumption === "Heavy" ? "#991b1b" : undefined} />
                  <InfoCell label="Status" value={sc.status?.toUpperCase()}
                    bg={sc.status === "approved" ? "#f0fdf4" : sc.status === "declined" ? "#fef2f2" : "#f8fafc"}
                    valueColor={sc.status === "approved" ? "#166534" : sc.status === "declined" ? "#991b1b" : NAVY} />
                </div>
                <InfoCell label="Pre-existing Conditions" value={conditions}
                  bg={conditions !== "None" ? "#fef2f2" : "#f0fdf4"}
                  valueColor={conditions !== "None" ? "#991b1b" : "#166534"} />
                {md.familyHistory && md.familyHistory !== "None" && (
                  <div style={{ marginTop: 10 }}>
                    <InfoCell label="Family History" value={md.familyHistory} />
                  </div>
                )}
                {md.currentMedications && md.currentMedications !== "None" && (
                  <div style={{ marginTop: 10 }}>
                    <InfoCell label="Current Medications" value={md.currentMedications} />
                  </div>
                )}
              </div>

              {/* Decision form */}
              {isPending && (
                <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)", padding: 20, border: "1px solid #e0e7ff" }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: NAVY, marginBottom: 14 }}>Underwriting Decision</div>
                  <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
                    {[
                      ["approved",  "Approve",  "#22c55e", "#dcfce7"],
                      ["declined",  "Decline",  "#ef4444", "#fee2e2"],
                      ["referred",  "Refer",    "#f59e0b", "#fef9c3"],
                    ].map(([val, label, clr, bg]) => (
                      <button key={val} onClick={() => setDecision(d => ({ ...d, outcome: val }))}
                        style={{ flex: 1, padding: "10px 0", borderRadius: 8,
                          border: `2px solid ${decision.outcome === val ? clr : "#e5e7eb"}`,
                          background: decision.outcome === val ? bg : "#fff",
                          color: decision.outcome === val ? clr : TXT2,
                          fontWeight: 700, fontSize: 13, cursor: "pointer", fontFamily: "inherit", transition: "all 0.15s" }}>
                        {label}
                      </button>
                    ))}
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <label style={{ fontSize: 12, fontWeight: 600, color: TXT2, display: "block", marginBottom: 5 }}>Reviewer ID *</label>
                    <input value={decision.reviewer} onChange={e => setDecision(d => ({ ...d, reviewer: e.target.value }))}
                      placeholder="Your name or staff ID"
                      style={{ width: "100%", padding: "8px 12px", borderRadius: 6, border: "1px solid #d1d5db",
                        fontSize: 13, fontFamily: "inherit", boxSizing: "border-box", outline: "none" }} />
                  </div>
                  <div style={{ marginBottom: 14 }}>
                    <label style={{ fontSize: 12, fontWeight: 600, color: TXT2, display: "block", marginBottom: 5 }}>Decision notes</label>
                    <textarea value={decision.notes} onChange={e => setDecision(d => ({ ...d, notes: e.target.value }))}
                      rows={3} placeholder="Rationale, conditions, follow-up actions…"
                      style={{ width: "100%", padding: "8px 12px", borderRadius: 6, border: "1px solid #d1d5db",
                        fontSize: 13, fontFamily: "inherit", resize: "vertical", boxSizing: "border-box", outline: "none" }} />
                  </div>
                  <button onClick={submitDecision} disabled={submitting || !decision.reviewer.trim()}
                    style={{ width: "100%", padding: "11px",
                      background: decision.outcome === "approved" ? "#22c55e" : decision.outcome === "declined" ? "#ef4444" : "#f59e0b",
                      color: "#fff", border: "none", borderRadius: 8, fontSize: 14, fontWeight: 700,
                      cursor: submitting || !decision.reviewer.trim() ? "not-allowed" : "pointer",
                      fontFamily: "inherit", opacity: submitting || !decision.reviewer.trim() ? 0.5 : 1 }}>
                    {submitting ? "Saving…" : `Confirm — ${decision.outcome.charAt(0).toUpperCase() + decision.outcome.slice(1)}`}
                  </button>
                </div>
              )}

              {/* Audit trail */}
              <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 1px 4px rgba(0,0,0,0.08)", padding: 20 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: NAVY, marginBottom: 14 }}>Audit Trail</div>
                <div>
                  {events.map((ev, i) => (
                    <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: 3 }}>
                        <div style={{ width: 14, height: 14, borderRadius: "50%",
                          border: `2px solid ${ev.done ? "#22c55e" : "#d1d5db"}`,
                          background: ev.done ? "#22c55e" : "#fff", flexShrink: 0 }} />
                        {i < events.length - 1 && (
                          <div style={{ width: 2, flex: 1, background: "#e5e7eb", minHeight: 20, margin: "2px 0" }} />
                        )}
                      </div>
                      <div style={{ paddingBottom: 16, flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: ev.done ? TXT : GRAY }}>{ev.event}</div>
                        <div style={{ fontSize: 11, color: GRAY, marginTop: 2 }}>
                          {ev.timestamp ? new Date(ev.timestamp).toLocaleString() : "Pending"}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
