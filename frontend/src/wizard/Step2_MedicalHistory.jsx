const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const ERR = "#ef4444";
const OK = "#10b981";

const FIELD = {
  width: "100%", padding: "10px 14px", borderRadius: 8,
  border: "1.5px solid #e5e7eb", fontSize: 14, fontFamily: "inherit",
  outline: "none", background: "#fff", boxSizing: "border-box",
};
const ERR_FIELD = { ...FIELD, borderColor: ERR };

const CONDITIONS = [
  "Diabetes", "Hypertension", "Heart Disease", "Asthma/COPD",
  "Cancer (past/present)", "Kidney Disease", "Liver Disease",
  "HIV/AIDS", "Thyroid Disorder", "Stroke", "None",
];

const FAMILY_CONDITIONS = [
  "Diabetes", "Heart Disease", "Cancer", "Stroke", "Hypertension", "None",
];

function Field({ label, required, error, hint, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 13, fontWeight: 600, color: TXT2 }}>
        {label}{required && <span style={{ color: ERR }}> *</span>}
      </label>
      {hint && <span style={{ fontSize: 12, color: TXT2, marginTop: -2 }}>{hint}</span>}
      {children}
      {error && <span style={{ fontSize: 12, color: ERR }}>{error}</span>}
    </div>
  );
}

function CheckGroup({ options, selected, onChange, exclusive }) {
  const toggle = (val) => {
    if (exclusive && val === "None") {
      onChange(["None"]);
      return;
    }
    let next = selected.includes(val)
      ? selected.filter(v => v !== val)
      : [...selected.filter(v => v !== "None"), val];
    if (next.length === 0) next = ["None"];
    onChange(next);
  };

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {options.map(opt => {
        const active = selected.includes(opt);
        return (
          <button
            key={opt} type="button" onClick={() => toggle(opt)}
            style={{
              padding: "7px 14px", borderRadius: 20, fontSize: 13, cursor: "pointer",
              border: `1.5px solid ${active ? NAVY : "#e5e7eb"}`,
              background: active ? NAVY : "#fff",
              color: active ? "#fff" : TXT2,
              fontFamily: "inherit", transition: "all 0.15s",
            }}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

export default function Step2_MedicalHistory({ data, onChange, errors }) {
  const set = (k) => (e) => onChange({ ...data, [k]: e.target.value });
  const setVal = (k) => (v) => onChange({ ...data, [k]: v });

  const height = parseFloat(data.height);
  const weight = parseFloat(data.weight);
  const bmi = height > 0 && weight > 0 ? (weight / ((height / 100) ** 2)).toFixed(1) : null;

  const bmiCategory = bmi
    ? bmi < 18.5 ? { label: "Underweight", color: "#f59e0b" }
    : bmi < 25 ? { label: "Normal", color: OK }
    : bmi < 30 ? { label: "Overweight", color: "#f59e0b" }
    : { label: "Obese", color: ERR }
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, marginBottom: 4 }}>Medical History</h2>
        <p style={{ fontSize: 14, color: TXT2 }}>Health profile for underwriting risk assessment.</p>
      </div>

      {/* BMI */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, alignItems: "end" }}>
        <Field label="Height (cm)" required error={errors.height}>
          <input type="number" style={errors.height ? ERR_FIELD : FIELD}
            value={data.height} onChange={set("height")} placeholder="170" min={100} max={220} />
        </Field>
        <Field label="Weight (kg)" required error={errors.weight}>
          <input type="number" style={errors.weight ? ERR_FIELD : FIELD}
            value={data.weight} onChange={set("weight")} placeholder="70" min={30} max={250} />
        </Field>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: bmiCategory ? `${bmiCategory.color}15` : "#f1f3f5", border: `1.5px solid ${bmiCategory?.color || "#e5e7eb"}`, textAlign: "center" }}>
          <div style={{ fontSize: 11, color: TXT2, marginBottom: 2 }}>BMI</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: bmiCategory?.color || TXT2 }}>{bmi || "—"}</div>
          {bmiCategory && <div style={{ fontSize: 11, color: bmiCategory.color }}>{bmiCategory.label}</div>}
        </div>
      </div>

      {/* Lifestyle */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Field label="Smoking Status" required error={errors.smokingStatus}>
          <select style={errors.smokingStatus ? ERR_FIELD : FIELD} value={data.smokingStatus} onChange={set("smokingStatus")}>
            <option value="">Select</option>
            {["Never", "Former (quit >1yr)", "Current"].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Alcohol Consumption" required error={errors.alcoholConsumption}>
          <select style={errors.alcoholConsumption ? ERR_FIELD : FIELD} value={data.alcoholConsumption} onChange={set("alcoholConsumption")}>
            <option value="">Select</option>
            {["None", "Occasional (1-2/week)", "Moderate (3-4/week)", "Heavy (daily)"].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Exercise Frequency" required error={errors.exerciseFrequency}>
          <select style={errors.exerciseFrequency ? ERR_FIELD : FIELD} value={data.exerciseFrequency} onChange={set("exerciseFrequency")}>
            <option value="">Select</option>
            {["Sedentary", "Light (1-2x/week)", "Moderate (3-4x/week)", "Active (5+/week)"].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Blood Pressure" error={errors.bloodPressure} hint="If known">
          <select style={FIELD} value={data.bloodPressure} onChange={set("bloodPressure")}>
            <option value="">Select / Unknown</option>
            {["Normal (<120/80)", "Elevated (120-129/80)", "Stage 1 (130-139/80-89)", "Stage 2 (≥140/90)"].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
      </div>

      {/* Pre-existing conditions */}
      <Field label="Pre-existing Conditions" required error={errors.preexistingConditions}>
        <CheckGroup
          options={CONDITIONS}
          selected={data.preexistingConditions}
          onChange={setVal("preexistingConditions")}
          exclusive
        />
      </Field>

      {/* Family history */}
      <Field label="Family History of Illness" hint="First-degree relatives (parents, siblings)">
        <CheckGroup
          options={FAMILY_CONDITIONS}
          selected={data.familyHistory}
          onChange={setVal("familyHistory")}
          exclusive
        />
      </Field>

      {/* Medications */}
      <Field label="Current Medications" hint="List any ongoing prescriptions (optional)">
        <textarea
          rows={3} style={{ ...FIELD, resize: "vertical" }}
          value={data.currentMedications} onChange={set("currentMedications")}
          placeholder="e.g. Metformin 500mg for diabetes, Amlodipine 5mg for hypertension..."
        />
      </Field>
    </div>
  );
}
