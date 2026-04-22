/**
 * LifeInsurancePricer.jsx
 *
 * Shell that renders the ActuarialWorkbench (GLM Workbench) directly.
 * Assumption version: v3.0-cambodia-2026-04-14
 */

import ActuarialWorkbench from "./ActuarialWorkbench";

const NAVY_D = "#091d5e";
const NAVY   = "#0d2b7a";
const GOLD   = "#f5a623";
const WHITE  = "#ffffff";
const LTGRAY = "#f8f9fb";

const ASSUMPTION_VERSION = "v3.0-cambodia-2026-04-14";

export default function LifeInsurancePricer() {
  return (
    <div style={{ paddingTop: 80, minHeight: "100vh", background: LTGRAY, fontFamily: "'DM Sans', sans-serif" }}>

      {/* Page header */}
      <div style={{ background: `linear-gradient(135deg, ${NAVY_D} 0%, ${NAVY} 100%)`, padding: "48px 24px 40px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{
            display: "inline-block", background: "rgba(245,166,35,0.15)",
            border: "1px solid rgba(245,166,35,0.3)", borderRadius: 20,
            padding: "4px 14px", fontSize: 12, color: GOLD, fontWeight: 600, marginBottom: 12,
          }}>
            INTERNAL — ACTUARY WORKBENCH
          </div>
          <h1 style={{ color: WHITE, fontSize: 32, fontWeight: 700, margin: "0 0 8px" }}>Life Insurance Pricer</h1>
          <p style={{ color: "rgba(255,255,255,0.65)", fontSize: 15, margin: 0 }}>
            Mortality Ratio Method · Cambodia WHO SEA tables · Assumption {ASSUMPTION_VERSION}
          </p>
        </div>
      </div>

      <ActuarialWorkbench />

      <style>{`
        @media (max-width: 768px) {
          div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
