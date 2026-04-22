import { useState } from "react";

const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const ERR = "#ef4444";
const LTGRAY = "#f8f9fb";

const REGIONS = [
  "Phnom Penh", "Siem Reap", "Battambang", "Sihanoukville",
  "Kampong Cham", "Kampot", "Takeo", "Kandal", "Rural Areas",
];

const OCCUPATIONS = [
  "Office/Desk", "Healthcare Worker", "Teacher/Educator",
  "Manual Labour", "Retail/Service", "Agriculture",
  "Security/Military", "Transportation", "Self-employed", "Other",
];

const FIELD = {
  width: "100%", padding: "10px 14px", borderRadius: 8,
  border: "1.5px solid #e5e7eb", fontSize: 14, fontFamily: "inherit",
  outline: "none", background: "#fff", boxSizing: "border-box",
};

const ERR_FIELD = { ...FIELD, borderColor: ERR };

function Field({ label, required, error, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 13, fontWeight: 600, color: TXT2 }}>
        {label}{required && <span style={{ color: ERR }}> *</span>}
      </label>
      {children}
      {error && <span style={{ fontSize: 12, color: ERR }}>{error}</span>}
    </div>
  );
}

export default function Step1_PersonalInfo({ data, onChange, errors }) {
  const set = (k) => (e) => onChange({ ...data, [k]: e.target.value });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, marginBottom: 4 }}>Personal Information</h2>
        <p style={{ fontSize: 14, color: TXT2 }}>Basic details about the applicant.</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Field label="Full Name" required error={errors.fullName}>
          <input
            style={errors.fullName ? ERR_FIELD : FIELD}
            value={data.fullName} onChange={set("fullName")}
            placeholder="e.g. Chan Poly"
          />
        </Field>

        <Field label="Date of Birth" required error={errors.dateOfBirth}>
          <input
            type="date" style={errors.dateOfBirth ? ERR_FIELD : FIELD}
            value={data.dateOfBirth} onChange={set("dateOfBirth")}
            max={new Date().toISOString().split("T")[0]}
          />
        </Field>

        <Field label="Gender" required error={errors.gender}>
          <select style={errors.gender ? ERR_FIELD : FIELD} value={data.gender} onChange={set("gender")}>
            <option value="">Select gender</option>
            {["Male", "Female", "Other"].map(g => <option key={g} value={g}>{g}</option>)}
          </select>
        </Field>

        <Field label="National ID / Passport" error={errors.nationalId}>
          <input
            style={FIELD} value={data.nationalId} onChange={set("nationalId")}
            placeholder="Optional"
          />
        </Field>

        <Field label="Phone Number" required error={errors.phone}>
          <input
            type="tel" style={errors.phone ? ERR_FIELD : FIELD}
            value={data.phone} onChange={set("phone")}
            placeholder="+855 XX XXX XXX"
          />
        </Field>

        <Field label="Email Address" required error={errors.email}>
          <input
            type="email" style={errors.email ? ERR_FIELD : FIELD}
            value={data.email} onChange={set("email")}
            placeholder="name@email.com"
          />
        </Field>

        <Field label="Region" required error={errors.region}>
          <select style={errors.region ? ERR_FIELD : FIELD} value={data.region} onChange={set("region")}>
            <option value="">Select region</option>
            {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </Field>

        <Field label="Occupation" required error={errors.occupation}>
          <select style={errors.occupation ? ERR_FIELD : FIELD} value={data.occupation} onChange={set("occupation")}>
            <option value="">Select occupation</option>
            {OCCUPATIONS.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        </Field>
      </div>

      <div style={{ background: `rgba(245,166,35,0.08)`, borderRadius: 10, padding: "12px 16px", borderLeft: `3px solid ${GOLD}` }}>
        <p style={{ fontSize: 13, color: TXT2, margin: 0 }}>
          Your personal data is encrypted and used solely for underwriting purposes, in accordance with Cambodia's data protection regulations.
        </p>
      </div>
    </div>
  );
}
