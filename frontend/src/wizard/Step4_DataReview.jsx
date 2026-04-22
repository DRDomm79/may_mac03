const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const LTGRAY = "#f8f9fb";

function Section({ title, children }) {
  return (
    <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e5e7eb", overflow: "hidden" }}>
      <div style={{ background: NAVY, padding: "10px 16px" }}>
        <h3 style={{ color: "#fff", fontSize: 14, fontWeight: 600, margin: 0 }}>{title}</h3>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1px", background: "#e5e7eb" }}>
        {children}
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={{ padding: "10px 16px", background: "#fff" }}>
      <div style={{ fontSize: 11, color: TXT2, marginBottom: 2, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 600, color: value ? TXT : TXT2 }}>{value || "—"}</div>
    </div>
  );
}

export default function Step4_DataReview({ data, onGoToStep }) {
  const bmi = data.height && data.weight
    ? (parseFloat(data.weight) / ((parseFloat(data.height) / 100) ** 2)).toFixed(1)
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, marginBottom: 4 }}>Review Your Application</h2>
        <p style={{ fontSize: 14, color: TXT2 }}>Please verify all information before submitting. Click "Edit" to go back and correct any section.</p>
      </div>

      {/* Personal Info */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: TXT2, textTransform: "uppercase", letterSpacing: 0.5 }}>Personal Information</span>
          <button onClick={() => onGoToStep(1)} style={{
            padding: "4px 12px", borderRadius: 6, border: "1px solid #e5e7eb",
            background: "#fff", fontSize: 12, cursor: "pointer", fontFamily: "inherit", color: NAVY, fontWeight: 600,
          }}>Edit</button>
        </div>
        <Section title="">
          <Row label="Full Name" value={data.fullName} />
          <Row label="Date of Birth" value={data.dateOfBirth} />
          <Row label="Gender" value={data.gender} />
          <Row label="Phone" value={data.phone} />
          <Row label="Email" value={data.email} />
          <Row label="Region" value={data.region} />
          <Row label="Occupation" value={data.occupation} />
          <Row label="National ID" value={data.nationalId} />
        </Section>
      </div>

      {/* Medical History */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: TXT2, textTransform: "uppercase", letterSpacing: 0.5 }}>Medical History</span>
          <button onClick={() => onGoToStep(2)} style={{
            padding: "4px 12px", borderRadius: 6, border: "1px solid #e5e7eb",
            background: "#fff", fontSize: 12, cursor: "pointer", fontFamily: "inherit", color: NAVY, fontWeight: 600,
          }}>Edit</button>
        </div>
        <Section title="">
          <Row label="Height / Weight" value={data.height && data.weight ? `${data.height} cm / ${data.weight} kg` : null} />
          <Row label="BMI" value={bmi} />
          <Row label="Smoking Status" value={data.smokingStatus} />
          <Row label="Alcohol" value={data.alcoholConsumption} />
          <Row label="Exercise" value={data.exerciseFrequency} />
          <Row label="Blood Pressure" value={data.bloodPressure || "Not provided"} />
          <Row label="Pre-existing Conditions" value={data.preexistingConditions?.join(", ")} />
          <Row label="Family History" value={data.familyHistory?.join(", ") || "None"} />
        </Section>
        {data.currentMedications && (
          <div style={{ marginTop: 8, padding: "12px 16px", background: "#fff", borderRadius: 8, border: "1px solid #e5e7eb" }}>
            <div style={{ fontSize: 11, color: TXT2, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>Current Medications</div>
            <div style={{ fontSize: 14, color: TXT }}>{data.currentMedications}</div>
          </div>
        )}
      </div>

      {/* Documents */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: TXT2, textTransform: "uppercase", letterSpacing: 0.5 }}>Documents</span>
          <button onClick={() => onGoToStep(3)} style={{
            padding: "4px 12px", borderRadius: 6, border: "1px solid #e5e7eb",
            background: "#fff", fontSize: 12, cursor: "pointer", fontFamily: "inherit", color: NAVY, fontWeight: 600,
          }}>Edit</button>
        </div>
        <div style={{ padding: "12px 16px", background: "#fff", borderRadius: 8, border: "1px solid #e5e7eb" }}>
          {data.documentFile ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 20 }}>📄</span>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: TXT }}>{data.documentFile.name}</div>
                <div style={{ fontSize: 12, color: TXT2 }}>Uploaded</div>
              </div>
            </div>
          ) : (
            <p style={{ fontSize: 14, color: TXT2, margin: 0 }}>No document uploaded — underwriter may request documents later.</p>
          )}
        </div>
      </div>

      {/* Confirmation notice */}
      <div style={{ background: `rgba(245,166,35,0.08)`, borderRadius: 10, padding: "12px 16px", borderLeft: `3px solid ${GOLD}` }}>
        <p style={{ fontSize: 13, color: TXT, margin: 0 }}>
          By proceeding to the next step, you confirm that all information provided is accurate and complete. False declarations may result in policy cancellation.
        </p>
      </div>
    </div>
  );
}
