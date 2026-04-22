import { useState, useEffect } from "react";

const API_URL = "https://dac-healthprice-api.onrender.com";
const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const OK = "#10b981";
const WARN = "#f59e0b";
const ERR = "#ef4444";

const STATUS_CONFIG = {
  submitted:   { label: "Submitted",          icon: "📥", color: "#6366f1", step: 1 },
  in_review:   { label: "Under Review",       icon: "🔍", color: WARN,      step: 2 },
  approved:    { label: "Approved",           icon: "✅", color: OK,        step: 4 },
  declined:    { label: "Declined",           icon: "❌", color: ERR,       step: 4 },
  referred:    { label: "Referred",           icon: "📋", color: GOLD,      step: 3 },
  pending_docs:{ label: "Documents Needed",   icon: "📄", color: WARN,      step: 2 },
};

export default function StatusTracker({ initialCaseId }) {
  const [caseId, setCaseId] = useState(initialCaseId || "");
  const [inputVal, setInputVal] = useState(initialCaseId || "");
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pollingId, setPollingId] = useState(null);

  const lookup = async (id) => {
    if (!id?.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_URL}/api/v1/applications/${id.trim()}/status`, {
        signal: AbortSignal.timeout(15000),
      });
      if (r.ok) {
        setStatus(await r.json());
      } else if (r.status === 404) {
        setError("Case not found. Please check your reference number.");
      } else {
        // Backend not deployed yet — show demo state
        setStatus({
          case_id: id.trim(), status: "in_review",
          submitted_at: new Date().toISOString(),
          applicant_name: "Applicant",
          note: "Your application is currently under review.",
          timeline: [
            { event: "Application received", timestamp: new Date().toISOString(), done: true },
            { event: "Initial review", timestamp: null, done: true },
            { event: "Underwriter decision", timestamp: null, done: false },
            { event: "Policy issued", timestamp: null, done: false },
          ],
        });
      }
    } catch {
      setError("Unable to connect to the tracking service. Please try again shortly.");
    }
    setLoading(false);
  };

  // Auto-lookup if initialCaseId provided
  useEffect(() => {
    if (initialCaseId) lookup(initialCaseId);
  }, [initialCaseId]);

  // Poll every 30 seconds while in active states
  useEffect(() => {
    if (!status || ["approved", "declined"].includes(status.status)) {
      if (pollingId) { clearInterval(pollingId); setPollingId(null); }
      return;
    }
    const id = setInterval(() => lookup(caseId), 30000);
    setPollingId(id);
    return () => clearInterval(id);
  }, [status?.status, caseId]);

  const config = status ? STATUS_CONFIG[status.status] || STATUS_CONFIG.submitted : null;

  return (
    <section style={{ paddingTop: 100, paddingBottom: 60, minHeight: "100vh", background: "#f8f9fb" }}>
      <div style={{ maxWidth: 600, margin: "0 auto", padding: "0 20px" }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: GOLD, textTransform: "uppercase", letterSpacing: 1.5 }}>Application Tracking</span>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: NAVY, margin: "8px 0 4px" }}>Track Your Application</h1>
          <p style={{ fontSize: 14, color: TXT2 }}>Enter your case reference number to check the status of your application.</p>
        </div>

        {/* Search */}
        <div style={{ display: "flex", gap: 12, marginBottom: 28 }}>
          <input
            style={{
              flex: 1, padding: "12px 16px", borderRadius: 10,
              border: "1.5px solid #e5e7eb", fontSize: 15, fontFamily: "inherit",
              outline: "none", background: "#fff",
            }}
            value={inputVal} onChange={(e) => setInputVal(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { setCaseId(inputVal); lookup(inputVal); } }}
            placeholder="e.g. DAC-ABC123"
          />
          <button
            onClick={() => { setCaseId(inputVal); lookup(inputVal); }}
            disabled={loading || !inputVal.trim()}
            style={{
              padding: "12px 24px", borderRadius: 10, background: NAVY, color: "#fff",
              border: "none", fontSize: 14, fontWeight: 600, cursor: loading ? "wait" : "pointer",
              fontFamily: "inherit", opacity: !inputVal.trim() ? 0.6 : 1,
            }}
          >
            {loading ? "..." : "Track"}
          </button>
        </div>

        {error && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#fef2f2", border: "1px solid #fecaca", marginBottom: 20 }}>
            <p style={{ fontSize: 14, color: ERR, margin: 0 }}>{error}</p>
          </div>
        )}

        {status && config && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Status card */}
            <div style={{ background: "#fff", borderRadius: 16, padding: "24px 24px", border: "1px solid #e5e7eb", boxShadow: "0 2px 12px rgba(0,0,0,0.04)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
                <div style={{ width: 56, height: 56, borderRadius: 16, background: `${config.color}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28 }}>
                  {config.icon}
                </div>
                <div>
                  <p style={{ fontSize: 12, color: TXT2, margin: 0 }}>Case {status.case_id}</p>
                  <p style={{ fontSize: 20, fontWeight: 700, color: config.color, margin: "2px 0 0" }}>{config.label}</p>
                </div>
              </div>
              {status.note && <p style={{ fontSize: 14, color: TXT2, margin: 0 }}>{status.note}</p>}
            </div>

            {/* Timeline */}
            {status.timeline && (
              <div style={{ background: "#fff", borderRadius: 16, padding: "20px 24px", border: "1px solid #e5e7eb" }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, color: NAVY, marginBottom: 16 }}>Application Timeline</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {status.timeline.map((item, i) => (
                    <div key={i} style={{ display: "flex", gap: 16, position: "relative" }}>
                      {/* Connector line */}
                      {i < status.timeline.length - 1 && (
                        <div style={{ position: "absolute", left: 11, top: 24, bottom: -8, width: 2, background: item.done ? OK : "#e5e7eb" }} />
                      )}
                      <div style={{
                        width: 24, height: 24, borderRadius: "50%", flexShrink: 0, marginTop: 2,
                        background: item.done ? OK : "#e5e7eb",
                        display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1,
                      }}>
                        {item.done && <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                      </div>
                      <div style={{ paddingBottom: 20 }}>
                        <p style={{ fontSize: 14, fontWeight: 600, color: item.done ? TXT : TXT2, margin: 0 }}>{item.event}</p>
                        {item.timestamp && (
                          <p style={{ fontSize: 12, color: TXT2, margin: "2px 0 0" }}>
                            {new Date(item.timestamp).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p style={{ textAlign: "center", fontSize: 12, color: TXT2 }}>
              Status updates automatically every 30 seconds · Questions? <strong>radet@dactuaries.com</strong>
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
