const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const ERR = "#ef4444";

function ConsentItem({ id, checked, onChange, error, label, sublabel }) {
  return (
    <div style={{
      padding: "14px 16px", borderRadius: 10,
      border: `1.5px solid ${error ? ERR : checked ? NAVY : "#e5e7eb"}`,
      background: checked ? `rgba(13,43,122,0.03)` : "#fff",
      cursor: "pointer", transition: "all 0.15s",
    }} onClick={() => onChange(!checked)}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div style={{
          width: 20, height: 20, borderRadius: 4, border: `2px solid ${checked ? NAVY : "#d1d5db"}`,
          background: checked ? NAVY : "#fff", flexShrink: 0, marginTop: 1,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {checked && (
            <svg width="12" height="12" viewBox="0 0 12 12">
              <path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
        <div>
          <p style={{ fontSize: 14, fontWeight: 600, color: TXT, margin: 0 }}>{label}</p>
          {sublabel && <p style={{ fontSize: 13, color: TXT2, margin: "4px 0 0" }}>{sublabel}</p>}
        </div>
      </div>
      {error && <p style={{ fontSize: 12, color: ERR, margin: "8px 0 0" }}>{error}</p>}
    </div>
  );
}

export default function Step5_Consent({ data, onChange, errors }) {
  const set = (k) => (v) => onChange({ ...data, [k]: v });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, marginBottom: 4 }}>Declaration & Consent</h2>
        <p style={{ fontSize: 14, color: TXT2 }}>Please read and accept the following declarations before submitting your application.</p>
      </div>

      <ConsentItem
        checked={data.consentTerms}
        onChange={set("consentTerms")}
        error={errors.consentTerms}
        label="I agree to the Terms & Conditions"
        sublabel="I confirm that I have read and understood the terms governing this insurance application, including the policyholder obligations and exclusions."
      />

      <ConsentItem
        checked={data.consentPrivacy}
        onChange={set("consentPrivacy")}
        error={errors.consentPrivacy}
        label="I consent to the Privacy Policy"
        sublabel="I authorise DAC to collect, process, and store my personal and medical data for underwriting and insurance administration purposes, in compliance with Cambodia's data protection laws."
      />

      <ConsentItem
        checked={data.consentDataProcessing}
        onChange={set("consentDataProcessing")}
        error={errors.consentDataProcessing}
        label="I consent to AI-assisted underwriting"
        sublabel="I understand that an AI system will assist in processing my application. I acknowledge that a human underwriter will make the final decision and I can request a manual review at any time."
      />

      <ConsentItem
        checked={data.consentTruthfulness}
        onChange={set("consentTruthfulness")}
        error={errors.consentTruthfulness}
        label="I declare all information is true and complete"
        sublabel="I understand that providing false or incomplete information may result in policy cancellation, claim rejection, or legal consequences under Cambodian insurance regulations."
      />

      {/* Electronic signature */}
      <div>
        <label style={{ fontSize: 13, fontWeight: 600, color: TXT2, display: "block", marginBottom: 6 }}>
          Electronic Signature <span style={{ color: ERR }}>*</span>
        </label>
        <p style={{ fontSize: 13, color: TXT2, marginBottom: 8 }}>
          Type your full name as it appears on your ID to sign this application electronically.
        </p>
        <input
          style={{
            width: "100%", padding: "12px 16px", borderRadius: 8,
            border: `1.5px solid ${errors.signature ? ERR : "#e5e7eb"}`,
            fontSize: 16, fontFamily: "cursive", outline: "none",
            background: "#fff", boxSizing: "border-box",
          }}
          value={data.signature} onChange={(e) => onChange({ ...data, signature: e.target.value })}
          placeholder="Sign here with your full name"
        />
        {errors.signature && <p style={{ fontSize: 12, color: ERR, marginTop: 4 }}>{errors.signature}</p>}
      </div>

      {/* Summary banner */}
      <div style={{
        background: `linear-gradient(135deg, ${NAVY} 0%, #091d5e 100%)`,
        borderRadius: 12, padding: "16px 20px",
        display: "flex", alignItems: "center", gap: 16,
      }}>
        <div style={{ fontSize: 28 }}>🔒</div>
        <div>
          <p style={{ color: "#fff", fontWeight: 600, fontSize: 14, margin: 0 }}>Secure & Encrypted</p>
          <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 13, margin: "4px 0 0" }}>
            Your application is transmitted over TLS 1.3. Data is encrypted at rest and accessible only to authorised underwriting staff.
          </p>
        </div>
      </div>
    </div>
  );
}
