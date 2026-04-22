/**
 * DAC HealthPrice — Model Retraining Dashboard
 *
 * Purpose: Transparent visualization of how models improve with claim data
 * For stakeholders to understand and validate the retraining process
 */

import { useState } from "react";

// ── Colors & Styles ──────────────────────────────────────────────────────────
const NAVY = "#1a1a2e";
const GOLD = "#f5c563";
const GREEN = "#10b981";
const RED = "#ef4444";
const GRAY = "#f5f5f5";
const GRAY_TEXT = "#4b5563";
const WHITE = "#ffffff";

const lbl = { display: "block", fontSize: 12, fontWeight: 600, color: GRAY_TEXT, marginBottom: 8 };
const inp = { width: "100%", padding: "10px 12px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit", outline: "none" };

// ── Mock Model Data ──────────────────────────────────────────────────────────
const INITIAL_MODEL = {
  version: "v2.3",
  deployed_date: "2026-03-15",
  metrics: {
    mape: 0.185,      // Mean Absolute Percentage Error: 18.5%
    r_squared: 0.742, // R²: 74.2%
    rmse: 245.67,     // Root Mean Squared Error: $245.67
  },
  training_records: 5420,
  claim_types: { IPD: 0.38, OPD: 0.42, Dental: 0.14, Maternity: 0.06 },
};

const RETRAINED_MODEL = {
  version: "v2.4",
  deployed_date: null,
  metrics: {
    mape: 0.162,      // Improved from 18.5% → 16.2%
    r_squared: 0.791, // Improved from 74.2% → 79.1%
    rmse: 198.34,     // Improved from $245.67 → $198.34
  },
  training_records: 5620,
  claim_types: { IPD: 0.36, OPD: 0.44, Dental: 0.15, Maternity: 0.05 },
};

// ── Helper Components ────────────────────────────────────────────────────────
function MetricBadge({ label, oldValue, newValue, unit = "", isPercent = false, isMoney = false }) {
  const improvement = oldValue > newValue ? ((oldValue - newValue) / oldValue * 100) : ((newValue - oldValue) / oldValue * 100);
  const isPositive = oldValue > newValue; // Lower is better for MAPE and RMSE

  const formatValue = (v) => {
    if (isMoney) return `$${v.toFixed(2)}`;
    if (isPercent) return `${(v * 100).toFixed(1)}%`;
    return v.toFixed(3);
  };

  return (
    <div style={{ background: WHITE, border: "1px solid #e5e7eb", borderRadius: 10, padding: 16, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <p style={{ fontSize: 11, color: GRAY_TEXT, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 6 }}>{label}</p>
          <div style={{ display: "flex", gap: 20 }}>
            <div>
              <p style={{ fontSize: 11, color: GRAY_TEXT, marginBottom: 2 }}>Current (v2.3)</p>
              <p style={{ fontSize: 20, fontWeight: 700, color: NAVY }}>{formatValue(oldValue)} {unit}</p>
            </div>
            <div style={{ borderLeft: "2px solid #e5e7eb", paddingLeft: 20 }}>
              <p style={{ fontSize: 11, color: GRAY_TEXT, marginBottom: 2 }}>Retrained (v2.4)</p>
              <p style={{ fontSize: 20, fontWeight: 700, color: isPositive ? GREEN : RED }}>{formatValue(newValue)} {unit}</p>
            </div>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <p style={{ fontSize: 12, fontWeight: 700, color: isPositive ? GREEN : RED, marginBottom: 4 }}>
            {isPositive ? "↓" : "↑"} {Math.abs(improvement).toFixed(1)}%
          </p>
          <p style={{ fontSize: 10, color: GRAY_TEXT }}>{isPositive ? "Improved" : "Degraded"}</p>
        </div>
      </div>
    </div>
  );
}

function SegmentRow({ label, oldRatio, newRatio, status }) {
  const statusColor = status === "improved" ? GREEN : status === "degraded" ? RED : GOLD;
  const delta = ((newRatio - oldRatio) / oldRatio * 100).toFixed(1);

  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 12px", borderBottom: "1px solid #f3f4f6" }}>
      <span style={{ fontSize: 13, color: NAVY, fontWeight: 500 }}>{label}</span>
      <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
        <div style={{ textAlign: "right" }}>
          <p style={{ fontSize: 12, color: GRAY_TEXT }}>
            {oldRatio.toFixed(3)} <span style={{ fontSize: 10, color: "#999" }}>→</span> {newRatio.toFixed(3)}
          </p>
          <p style={{ fontSize: 11, fontWeight: 600, color: statusColor }}>
            {delta > 0 ? "+" : ""}{delta}%
          </p>
        </div>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: statusColor }} />
      </div>
    </div>
  );
}

// ── Phase 1: Upload Claims ───────────────────────────────────────────────────
function PhaseUpload({ onPhaseChange }) {
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [errors, setErrors] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [datasetName, setDatasetName] = useState("Claims Data - Jan 2026");

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;

    setFile(f);
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const lines = ev.target.result.trim().split("\n");
        const headers = lines[0].split(",").map(h => h.trim());
        const parsed = lines.slice(1).map((l, i) => {
          const vals = l.split(",").map(v => v.trim());
          const row = {};
          headers.forEach((h, j) => { row[h] = vals[j]; });
          return row;
        });

        // Validate
        const validTypes = new Set(["IPD", "OPD", "Dental", "Maternity"]);
        const errs = [];
        parsed.forEach((r, i) => {
          if (!r.claim_id) errs.push({ row: i + 2, issue: "Missing claim_id" });
          if (!r.customer_age || isNaN(+r.customer_age)) errs.push({ row: i + 2, issue: "Invalid age" });
          if (!r.claim_amount || isNaN(+r.claim_amount)) errs.push({ row: i + 2, issue: "Invalid claim_amount" });
          if (!r.claim_type || !validTypes.has(r.claim_type)) errs.push({ row: i + 2, issue: "Invalid claim_type" });
          if (!r.claim_date) errs.push({ row: i + 2, issue: "Missing claim_date" });
        });

        setRows(parsed);
        setErrors(errs);
      } catch {
        setErrors([{ issue: "Failed to parse CSV" }]);
      }
    };
    reader.readAsText(f);
  };

  const handleUpload = () => {
    if (!rows.length || errors.length || !datasetName) return;
    setUploading(true);
    setTimeout(() => {
      onPhaseChange("analysis", { dataset_name: datasetName, record_count: rows.length });
      setUploading(false);
    }, 1500);
  };

  const downloadTemplate = () => {
    const csv = `claim_id,customer_age,customer_occupation,claim_type,claim_amount,claim_date
CLM001,35,Office/Desk,IPD,2800,2026-01-15
CLM002,42,Manual Labor,OPD,75,2026-01-18
CLM003,28,Retail/Service,Dental,145,2026-01-22
CLM004,55,Retired,IPD,4200,2026-02-03
CLM005,38,Healthcare,OPD,55,2026-02-10`;
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "claims_template.csv";
    a.click();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ background: WHITE, borderRadius: 12, padding: 24, border: "1px solid #e5e7eb" }}>
        <h3 style={{ fontSize: 18, fontWeight: 700, color: NAVY, marginBottom: 8 }}>Upload Claim Data</h3>
        <p style={{ fontSize: 13, color: GRAY_TEXT, marginBottom: 24, lineHeight: 1.6 }}>
          Upload historical claims data (customer age, occupation, claim type, amount, date).
          The system will retrain models and show you accuracy improvements before/after.
        </p>

        <div style={{ marginBottom: 20 }}>
          <label style={lbl}>Dataset Name</label>
          <input
            type="text"
            value={datasetName}
            onChange={e => setDatasetName(e.target.value)}
            placeholder="e.g., Claims Data - January 2026"
            style={inp}
          />
          <p style={{ fontSize: 11, color: GRAY_TEXT, marginTop: 4 }}>For your records and audit trail</p>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={lbl}>CSV File</label>
          <input type="file" accept=".csv" onChange={handleFile} style={inp} />
          <button
            onClick={downloadTemplate}
            style={{ marginTop: 8, padding: "6px 12px", borderRadius: 6, border: "1px solid " + NAVY, background: WHITE, color: NAVY, fontSize: 12, fontWeight: 600, cursor: "pointer" }}
          >
            Download Template
          </button>
        </div>

        {rows.length > 0 && (
          <div style={{ padding: 12, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, marginBottom: 20 }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: "#166534" }}>
              {rows.length} claims parsed successfully {errors.length > 0 && `(${errors.length} validation errors)`}
            </p>
          </div>
        )}

        {errors.length > 0 && (
          <div style={{ padding: 12, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, marginBottom: 20, maxHeight: 200, overflowY: "auto" }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: "#991b1b", marginBottom: 8 }}>Validation Errors:</p>
            {errors.map((e, i) => (
              <p key={i} style={{ fontSize: 11, color: "#7c2d12", marginBottom: 4 }}>
                Row {e.row || "?"}: {e.issue}
              </p>
            ))}
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!rows.length || errors.length > 0 || uploading || !datasetName}
          style={{
            width: "100%",
            padding: 12,
            background: rows.length && !errors.length && datasetName ? NAVY : "#d1d5db",
            color: WHITE,
            border: "none",
            borderRadius: 8,
            fontWeight: 600,
            fontSize: 14,
            cursor: (rows.length && !errors.length && datasetName && !uploading) ? "pointer" : "default",
            opacity: uploading ? 0.7 : 1,
          }}
        >
          {uploading ? "Uploading & Retraining..." : "Proceed to Analysis"}
        </button>
      </div>
    </div>
  );
}

// ── Phase 2: Show Improvements ───────────────────────────────────────────────
function PhaseAnalysis({ metadata, onPhaseChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div style={{ background: NAVY, color: WHITE, borderRadius: 12, padding: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>Model Retraining Results</h2>
        <p style={{ fontSize: 13, opacity: 0.9, marginBottom: 12 }}>{metadata.dataset_name} • {metadata.record_count} new claims</p>
        <div style={{ display: "flex", gap: 20, marginTop: 16 }}>
          <div>
            <p style={{ fontSize: 11, opacity: 0.8, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>Current Model</p>
            <p style={{ fontSize: 20, fontWeight: 700 }}>{INITIAL_MODEL.version}</p>
          </div>
          <div style={{ borderLeft: "2px solid rgba(255,255,255,0.2)", paddingLeft: 20 }}>
            <p style={{ fontSize: 11, opacity: 0.8, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>Retrained Model</p>
            <p style={{ fontSize: 20, fontWeight: 700 }}>{RETRAINED_MODEL.version}</p>
          </div>
        </div>
      </div>

      {/* Accuracy Metrics */}
      <div>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: NAVY, marginBottom: 16 }}>Model Accuracy Improvements</h3>

        <MetricBadge
          label="Prediction Error (MAPE)"
          oldValue={INITIAL_MODEL.metrics.mape}
          newValue={RETRAINED_MODEL.metrics.mape}
          isPercent={true}
        />

        <MetricBadge
          label="Model Fit (R²)"
          oldValue={INITIAL_MODEL.metrics.r_squared}
          newValue={RETRAINED_MODEL.metrics.r_squared}
          isPercent={true}
        />

        <MetricBadge
          label="Claim Amount RMSE"
          oldValue={INITIAL_MODEL.metrics.rmse}
          newValue={RETRAINED_MODEL.metrics.rmse}
          unit="USD"
          isMoney={true}
        />
      </div>

      {/* Segment-Level Analysis */}
      <div style={{ background: WHITE, borderRadius: 12, padding: 20, border: "1px solid #e5e7eb" }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: NAVY, marginBottom: 16 }}>Segment-Level Improvements</h3>
        <p style={{ fontSize: 12, color: GRAY_TEXT, marginBottom: 12 }}>O/E ratios (Observed vs Expected claims) by demographic segment</p>

        <div>
          <SegmentRow label="Age 18–24" oldRatio={0.92} newRatio={0.94} status="improved" />
          <SegmentRow label="Age 25–34" oldRatio={0.98} newRatio={0.98} status="stable" />
          <SegmentRow label="Age 35–44" oldRatio={1.18} newRatio={1.15} status="improved" />
          <SegmentRow label="Age 45–54" oldRatio={1.32} newRatio={1.28} status="improved" />
          <SegmentRow label="Age 55–64" oldRatio={1.45} newRatio={1.41} status="improved" />
          <SegmentRow label="Smoking: Current" oldRatio={1.42} newRatio={1.38} status="improved" />
          <SegmentRow label="Manual Labor" oldRatio={1.18} newRatio={1.12} status="improved" />
          <SegmentRow label="IPD (Hospital)" oldRatio={1.15} newRatio={1.12} status="improved" />
          <SegmentRow label="OPD (Outpatient)" oldRatio={0.98} newRatio={1.01} status="degraded" />
        </div>
      </div>

      {/* Premium Impact */}
      <div style={{ background: "linear-gradient(135deg, #fef3c7 0%, #fef08a 100%)", borderRadius: 12, padding: 20, border: "1px solid #fcd34d" }}>
        <h3 style={{ fontSize: 14, fontWeight: 700, color: NAVY, marginBottom: 8 }}>Expected Premium Impact</h3>
        <p style={{ fontSize: 24, fontWeight: 700, color: NAVY, marginBottom: 4 }}>+2.1% average</p>
        <p style={{ fontSize: 12, color: GRAY_TEXT }}>Typical customer will see ~2% premium increase due to more accurate risk assessment</p>
      </div>

      {/* Action Buttons */}
      <div style={{ display: "flex", gap: 12 }}>
        <button
          onClick={() => onPhaseChange("deploy")}
          style={{
            flex: 1,
            padding: 12,
            background: GREEN,
            color: WHITE,
            border: "none",
            borderRadius: 8,
            fontWeight: 600,
            fontSize: 14,
            cursor: "pointer",
          }}
        >
          Deploy v2.4 to Production
        </button>
        <button
          onClick={() => onPhaseChange("upload")}
          style={{
            flex: 1,
            padding: 12,
            background: GRAY,
            color: GRAY_TEXT,
            border: "none",
            borderRadius: 8,
            fontWeight: 600,
            fontSize: 14,
            cursor: "pointer",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Phase 3: Deployment Confirmation ─────────────────────────────────────────
function PhaseDeploy({ metadata, onPhaseChange }) {
  const [deploying, setDeploying] = useState(false);

  const handleDeploy = () => {
    setDeploying(true);
    setTimeout(() => {
      onPhaseChange("success");
    }, 2000);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ background: WHITE, borderRadius: 12, padding: 24, border: "1px solid #e5e7eb" }}>
        <h3 style={{ fontSize: 18, fontWeight: 700, color: NAVY, marginBottom: 16 }}>Deploy Retrained Model</h3>

        <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 16, marginBottom: 20 }}>
          <p style={{ fontSize: 13, fontWeight: 600, color: "#166534", marginBottom: 8 }}>Ready to deploy v2.4</p>
          <ul style={{ fontSize: 12, color: "#166534", lineHeight: 1.8, paddingLeft: 20 }}>
            <li>MAPE improved from 18.5% → 16.2%</li>
            <li>R² improved from 74.2% → 79.1%</li>
            <li>7 out of 9 segments show improvement</li>
            <li>{metadata.record_count} new claims integrated</li>
          </ul>
        </div>

        <div style={{ padding: 16, background: GRAY, borderRadius: 8, marginBottom: 20 }}>
          <p style={{ fontSize: 11, fontWeight: 600, color: GRAY_TEXT, letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 8 }}>Deployment Details</p>
          <div style={{ fontSize: 12, color: GRAY_TEXT, lineHeight: 2 }}>
            <p><strong>New Version:</strong> v2.4 ({new Date().toISOString().split('T')[0]})</p>
            <p><strong>Dataset:</strong> {metadata.dataset_name}</p>
            <p><strong>Training Records:</strong> 5,620</p>
            <p><strong>Deployed By:</strong> Admin</p>
          </div>
        </div>

        <div style={{ padding: 12, background: "#fef2f2", borderRadius: 8, marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: "#7c2d12" }}>
            <strong>Note:</strong> This will immediately replace the live model. All new quotes will use v2.4. Previous quotes remain auditable under v2.3.
          </p>
        </div>

        <div style={{ display: "flex", gap: 12 }}>
          <button
            onClick={handleDeploy}
            disabled={deploying}
            style={{
              flex: 1,
              padding: 12,
              background: GREEN,
              color: WHITE,
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 14,
              cursor: deploying ? "default" : "pointer",
              opacity: deploying ? 0.7 : 1,
            }}
          >
            {deploying ? "Deploying..." : "Confirm & Deploy"}
          </button>
          <button
            onClick={() => onPhaseChange("analysis")}
            disabled={deploying}
            style={{
              flex: 1,
              padding: 12,
              background: GRAY,
              color: GRAY_TEXT,
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Go Back
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Success Screen ───────────────────────────────────────────────────────────
function PhaseSuccess({ metadata, onPhaseChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ background: WHITE, borderRadius: 12, padding: 40, border: "1px solid #e5e7eb", textAlign: "center" }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>✓</div>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: GREEN, marginBottom: 8 }}>Model Deployed Successfully</h2>
        <p style={{ fontSize: 13, color: GRAY_TEXT, marginBottom: 24 }}>v2.4 is now live</p>

        <div style={{ padding: 16, background: GRAY, borderRadius: 8, marginBottom: 24, textAlign: "left" }}>
          <div style={{ fontSize: 12, color: GRAY_TEXT, lineHeight: 2 }}>
            <p><strong>Version:</strong> v2.4</p>
            <p><strong>Deployed:</strong> {new Date().toLocaleString()}</p>
            <p><strong>Dataset:</strong> {metadata.dataset_name}</p>
            <p><strong>Improvement:</strong> MAPE 18.5% → 16.2% (-12.4%)</p>
          </div>
        </div>

        <div style={{ display: "flex", gap: 12 }}>
          <button
            onClick={() => onPhaseChange("upload")}
            style={{
              flex: 1,
              padding: 12,
              background: NAVY,
              color: WHITE,
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Upload More Data
          </button>
          <button
            onClick={() => window.location.reload()}
            style={{
              flex: 1,
              padding: 12,
              background: GRAY,
              color: GRAY_TEXT,
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            View Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function ModelRetrainingDashboard() {
  const [phase, setPhase] = useState("upload");
  const [metadata, setMetadata] = useState(null);

  const handlePhaseChange = (newPhase, newMetadata = null) => {
    setPhase(newPhase);
    if (newMetadata) setMetadata(newMetadata);
  };

  return (
    <div style={{ minHeight: "100vh", background: GRAY, padding: 24, fontFamily: "inherit" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 32, fontWeight: 700, color: NAVY, marginBottom: 8 }}>Model Retraining Dashboard</h1>
          <p style={{ fontSize: 14, color: GRAY_TEXT }}>Upload claim data, see real-time accuracy improvements, deploy retrained models</p>
        </div>

        {/* Progress Indicator */}
        <div style={{ display: "flex", gap: 16, marginBottom: 32, justifyContent: "center" }}>
          {[
            { key: "upload", label: "Upload Data" },
            { key: "analysis", label: "View Improvements" },
            { key: "deploy", label: "Deploy Model" },
            { key: "success", label: "Complete" },
          ].map((step, i) => (
            <div key={step.key} style={{ display: "flex", alignItems: "center" }}>
              <div style={{
                width: 40,
                height: 40,
                borderRadius: "50%",
                background: phase === step.key ? GOLD : ["upload", "analysis", "deploy"].includes(phase) && ["upload", "analysis", "deploy"].indexOf(step.key) < ["upload", "analysis", "deploy"].indexOf(phase) ? GREEN : "#e5e7eb",
                color: WHITE,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: 700,
                fontSize: 14,
              }}>
                {i + 1}
              </div>
              {i < 3 && <div style={{ width: 40, height: 2, background: "#e5e7eb" }} />}
              <div style={{ fontSize: 12, fontWeight: 600, color: NAVY, marginLeft: 8, whiteSpace: "nowrap" }}>{step.label}</div>
            </div>
          ))}
        </div>

        {/* Phase Content */}
        {phase === "upload" && <PhaseUpload onPhaseChange={handlePhaseChange} />}
        {phase === "analysis" && metadata && <PhaseAnalysis metadata={metadata} onPhaseChange={handlePhaseChange} />}
        {phase === "deploy" && metadata && <PhaseDeploy metadata={metadata} onPhaseChange={handlePhaseChange} />}
        {phase === "success" && metadata && <PhaseSuccess metadata={metadata} onPhaseChange={handlePhaseChange} />}
      </div>
    </div>
  );
}
