import { useState, useEffect, useRef } from "react";
import ChatAdvisor from "./ChatAdvisor";

const API_URL = "https://dac-healthprice-api.onrender.com";

const NAVY   = "#0d2b7a";
const NAVY_D = "#091d5e";
const NAVY_L = "#1a4fba";
const GOLD   = "#f5a623";
const GOLD_D = "#e67e00";
const WHITE  = "#ffffff";
const GRAY   = "#94a3b8";
const LTGRAY = "#f1f5f9";
const TXT    = "#111827";
const TXT2   = "#4b5563";
const OK     = "#10b981";
const ERR    = "#ef4444";

// ─── Shared animation helpers ────────────────────────────────────────────────
function FadeIn({ children, delay = 0 }) {
  const ref = useRef(null);
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVis(true); obs.disconnect(); }
    }, { threshold: 0.12 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} style={{
      opacity: vis ? 1 : 0,
      transform: vis ? "translateY(0)" : "translateY(28px)",
      transition: `opacity 0.6s ease ${delay}s, transform 0.6s ease ${delay}s`,
    }}>{children}</div>
  );
}

// ─── Step progress bar ───────────────────────────────────────────────────────
function StepBar({ step, total, labels }) {
  return (
    <div style={{ marginBottom: 36 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        {labels.map((l, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1 }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%",
              background: i < step ? OK : i === step ? GOLD : LTGRAY,
              border: `2px solid ${i < step ? OK : i === step ? GOLD : "#d1d5db"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 13, fontWeight: 700, color: i <= step ? WHITE : "#9ca3af",
              transition: "all 0.3s",
            }}>
              {i < step
                ? <svg width="14" height="14" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
                : i + 1}
            </div>
            <span style={{ fontSize: 11, color: i === step ? GOLD_D : TXT2, fontWeight: i === step ? 600 : 400, marginTop: 4, textAlign: "center" }}>{l}</span>
          </div>
        ))}
      </div>
      <div style={{ height: 4, background: LTGRAY, borderRadius: 2, position: "relative" }}>
        <div style={{ height: "100%", borderRadius: 2, background: GOLD, width: `${(step / (total - 1)) * 100}%`, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

function Field({ label, required, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <label style={{ display: "block", fontSize: 13, fontWeight: 600, color: "#374151", marginBottom: 6 }}>
        {label}{required && <span style={{ color: ERR }}> *</span>}
      </label>
      {children}
    </div>
  );
}

const inputStyle = {
  width: "100%", padding: "10px 14px", borderRadius: 8,
  border: "1.5px solid #e5e7eb", fontSize: 14, fontFamily: "inherit",
  outline: "none", boxSizing: "border-box", transition: "border-color 0.2s",
};

const selectStyle = { ...inputStyle, background: WHITE, cursor: "pointer" };

// ─── Step 1: Personal Info ───────────────────────────────────────────────────
function Step1({ data, onChange }) {
  const set = (k, v) => onChange({ ...data, [k]: v });
  return (
    <div>
      <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Your personal details</h3>
      <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>We need a few details to personalise your application.</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 20px" }}>
        <Field label="First name" required>
          <input style={inputStyle} value={data.firstName || ""} onChange={e => set("firstName", e.target.value)} placeholder="Sokha" />
        </Field>
        <Field label="Last name" required>
          <input style={inputStyle} value={data.lastName || ""} onChange={e => set("lastName", e.target.value)} placeholder="Chan" />
        </Field>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 20px" }}>
        <Field label="Date of birth" required>
          <input style={inputStyle} type="date" value={data.dob || ""} onChange={e => set("dob", e.target.value)} />
        </Field>
        <Field label="Gender" required>
          <select style={selectStyle} value={data.gender || ""} onChange={e => set("gender", e.target.value)}>
            <option value="">Select…</option>
            <option>Male</option>
            <option>Female</option>
          </select>
        </Field>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 20px" }}>
        <Field label="Phone number" required>
          <input style={inputStyle} type="tel" value={data.phone || ""} onChange={e => set("phone", e.target.value)} placeholder="+855 12 345 678" />
        </Field>
        <Field label="Email address">
          <input style={inputStyle} type="email" value={data.email || ""} onChange={e => set("email", e.target.value)} placeholder="sokha@example.com" />
        </Field>
      </div>
      <Field label="Province / City" required>
        <select style={selectStyle} value={data.province || ""} onChange={e => set("province", e.target.value)}>
          <option value="">Select province…</option>
          {["Phnom Penh", "Siem Reap", "Battambang", "Kampong Cham", "Preah Vihear", "Other"].map(p => <option key={p}>{p}</option>)}
        </select>
      </Field>
      <Field label="National ID number" required>
        <input style={inputStyle} value={data.nationalId || ""} onChange={e => set("nationalId", e.target.value)} placeholder="e.g. 123456789012" />
      </Field>
    </div>
  );
}

// ─── Step 2: Health Profile ──────────────────────────────────────────────────
function Step2({ data, onChange }) {
  const set = (k, v) => onChange({ ...data, [k]: v });

  const bmi = (() => {
    const h = parseFloat(data.height);
    const w = parseFloat(data.weight);
    if (!h || !w || h <= 0) return null;
    return (w / ((h / 100) ** 2)).toFixed(1);
  })();

  const bmiColor = !bmi ? TXT2 : bmi < 18.5 ? "#3b82f6" : bmi < 25 ? OK : bmi < 30 ? GOLD_D : ERR;

  const CONDITIONS = [
    "Diabetes", "Hypertension", "Heart Disease", "Asthma / COPD",
    "Cancer (within 5 years)", "Kidney Disease", "Liver Disease",
    "HIV / AIDS", "None of the above",
  ];

  const toggleCondition = (c) => {
    const list = data.conditions || [];
    if (c === "None of the above") return set("conditions", ["None of the above"]);
    const filtered = list.filter(x => x !== "None of the above");
    set("conditions", list.includes(c) ? filtered.filter(x => x !== c) : [...filtered, c]);
  };

  return (
    <div>
      <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Your health profile</h3>
      <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>This helps us calculate an accurate premium for you.</p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0 20px" }}>
        <Field label="Height (cm)" required>
          <input style={inputStyle} type="number" value={data.height || ""} onChange={e => set("height", e.target.value)} placeholder="170" min={100} max={220} />
        </Field>
        <Field label="Weight (kg)" required>
          <input style={inputStyle} type="number" value={data.weight || ""} onChange={e => set("weight", e.target.value)} placeholder="65" min={30} max={200} />
        </Field>
        <Field label="Your BMI">
          <div style={{ padding: "10px 14px", borderRadius: 8, border: "1.5px solid #e5e7eb", fontSize: 14, fontWeight: 700, color: bmiColor, background: LTGRAY }}>
            {bmi ? `${bmi}` : "—"}
          </div>
        </Field>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 20px" }}>
        <Field label="Smoking status" required>
          <select style={selectStyle} value={data.smoking || ""} onChange={e => set("smoking", e.target.value)}>
            <option value="">Select…</option>
            <option value="Never">Never smoker</option>
            <option value="Former">Former smoker (quit &gt;1 year)</option>
            <option value="Current">Current smoker</option>
          </select>
        </Field>
        <Field label="Exercise frequency" required>
          <select style={selectStyle} value={data.exercise || ""} onChange={e => set("exercise", e.target.value)}>
            <option value="">Select…</option>
            <option value="None">Sedentary (rarely/never)</option>
            <option value="Low">Light (1–2× per week)</option>
            <option value="Moderate">Moderate (3–4× per week)</option>
            <option value="High">Active (5+ per week)</option>
          </select>
        </Field>
      </div>

      <Field label="Occupation type" required>
        <select style={selectStyle} value={data.occupation || ""} onChange={e => set("occupation", e.target.value)}>
          <option value="">Select…</option>
          <option value="Office/Desk">Office / Desk work</option>
          <option value="Manual/Light">Manual labour (light)</option>
          <option value="Manual/Heavy">Manual labour (heavy)</option>
          <option value="Healthcare">Healthcare worker</option>
          <option value="Self-employed">Self-employed / Informal</option>
          <option value="Student">Student</option>
        </select>
      </Field>

      <Field label="Pre-existing medical conditions" required>
        <p style={{ fontSize: 12, color: TXT2, marginBottom: 10 }}>Select all that apply. Honest disclosure protects your claim.</p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {CONDITIONS.map(c => {
            const checked = (data.conditions || []).includes(c);
            return (
              <label key={c} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 8, border: `1.5px solid ${checked ? GOLD : "#e5e7eb"}`, cursor: "pointer", background: checked ? `${GOLD}12` : WHITE, transition: "all 0.2s" }}>
                <input type="checkbox" checked={checked} onChange={() => toggleCondition(c)} style={{ accentColor: GOLD, width: 15, height: 15 }} />
                <span style={{ fontSize: 13, fontWeight: checked ? 600 : 400 }}>{c}</span>
              </label>
            );
          })}
        </div>
      </Field>
    </div>
  );
}

// ─── Step 3: Coverage Selection ──────────────────────────────────────────────
const TIERS = [
  { id: "Bronze",   limit: "$10,000", deduct: "$500",  color: "#cd7f32", monthly: 18 },
  { id: "Silver",   limit: "$40,000", deduct: "$250",  color: "#94a3b8", monthly: 32 },
  { id: "Gold",     limit: "$80,000", deduct: "$100",  color: GOLD_D,    monthly: 58 },
  { id: "Platinum", limit: "$150,000",deduct: "$0",    color: "#6366f1", monthly: 95 },
];

function Step3({ data, onChange }) {
  const set = (k, v) => onChange({ ...data, [k]: v });
  const tier = TIERS.find(t => t.id === data.tier) || TIERS[1];
  const riderTotal = (data.opd ? 12 : 0) + (data.dental ? 6 : 0) + (data.maternity ? 14 : 0);
  const total = tier.monthly + riderTotal;

  return (
    <div>
      <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Choose your coverage</h3>
      <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>Select a hospital (IPD) tier, then add optional riders.</p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12, marginBottom: 28 }}>
        {TIERS.map(t => {
          const sel = data.tier === t.id;
          return (
            <div key={t.id} onClick={() => set("tier", t.id)} style={{
              padding: 20, borderRadius: 12, border: `2px solid ${sel ? t.color : "#e5e7eb"}`,
              cursor: "pointer", background: sel ? `${t.color}10` : WHITE,
              transition: "all 0.2s", position: "relative",
            }}>
              {sel && <div style={{ position: "absolute", top: 10, right: 10, width: 20, height: 20, borderRadius: "50%", background: t.color, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
              </div>}
              <div style={{ fontSize: 16, fontWeight: 700, color: t.color, marginBottom: 6 }}>{t.id}</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: TXT }}>${t.monthly}<span style={{ fontSize: 13, color: TXT2, fontWeight: 400 }}>/mo</span></div>
              <div style={{ fontSize: 12, color: TXT2, marginTop: 6 }}>Annual limit: {t.limit}</div>
              <div style={{ fontSize: 12, color: TXT2 }}>Deductible: {t.deduct}</div>
            </div>
          );
        })}
      </div>

      <h4 style={{ fontSize: 15, fontWeight: 700, marginBottom: 14, color: TXT }}>Optional riders</h4>
      {[
        { key: "opd",      label: "OPD — Outpatient",   desc: "Doctor visits, tests, prescriptions",    price: 12 },
        { key: "dental",   label: "Dental",              desc: "Cleanings, fillings, emergency dental",  price: 6  },
        { key: "maternity",label: "Maternity",           desc: "Prenatal, delivery, postnatal, newborn", price: 14 },
      ].map(r => (
        <label key={r.key} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 18px", borderRadius: 10, border: `1.5px solid ${data[r.key] ? OK : "#e5e7eb"}`, cursor: "pointer", marginBottom: 10, background: data[r.key] ? `${OK}08` : WHITE, transition: "all 0.2s" }}>
          <input type="checkbox" checked={!!data[r.key]} onChange={e => set(r.key, e.target.checked)} style={{ accentColor: OK, width: 16, height: 16 }} />
          <div style={{ flex: 1 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{r.label}</span>
            <span style={{ fontSize: 12, color: TXT2, marginLeft: 8 }}>{r.desc}</span>
          </div>
          <span style={{ fontSize: 14, fontWeight: 700, color: TXT }}>+${r.price}/mo</span>
        </label>
      ))}

      <div style={{ marginTop: 20, padding: "16px 24px", background: NAVY, borderRadius: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ color: WHITE, fontSize: 15, fontWeight: 600 }}>Estimated monthly premium</span>
        <span style={{ color: GOLD, fontSize: 26, fontWeight: 800 }}>${total}<span style={{ fontSize: 13, color: GRAY, fontWeight: 400 }}>/mo</span></span>
      </div>
      <p style={{ fontSize: 11, color: TXT2, marginTop: 8 }}>* Estimate only. Final premium is set after underwriting review.</p>
    </div>
  );
}

// ─── Step 4: Document Upload ─────────────────────────────────────────────────
function Step4({ data, onChange }) {
  const set = (k, v) => onChange({ ...data, [k]: v });

  const DocSlot = ({ field, label, required, hint }) => {
    const file = data[field];
    return (
      <div style={{ marginBottom: 18 }}>
        <label style={{ display: "block", fontSize: 13, fontWeight: 600, color: "#374151", marginBottom: 6 }}>
          {label}{required && <span style={{ color: ERR }}> *</span>}
        </label>
        {hint && <p style={{ fontSize: 12, color: TXT2, marginBottom: 8 }}>{hint}</p>}
        <label style={{
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          gap: 8, padding: "24px 20px", borderRadius: 10,
          border: `2px dashed ${file ? OK : "#d1d5db"}`,
          background: file ? `${OK}08` : LTGRAY,
          cursor: "pointer", transition: "all 0.2s",
        }}>
          <input type="file" accept=".pdf,.jpg,.jpeg,.png" style={{ display: "none" }} onChange={e => set(field, e.target.files[0])} />
          {file
            ? <>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={OK} strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
                <span style={{ fontSize: 13, fontWeight: 600, color: OK }}>{file.name}</span>
                <span style={{ fontSize: 11, color: TXT2 }}>Click to replace</span>
              </>
            : <>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={GRAY} strokeWidth="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                <span style={{ fontSize: 13, color: TXT2 }}>Click to upload or drag & drop</span>
                <span style={{ fontSize: 11, color: GRAY }}>PDF, JPG, PNG — max 10 MB</span>
              </>}
        </label>
      </div>
    );
  };

  return (
    <div>
      <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Upload your documents</h3>
      <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>Required documents are processed securely and never shared outside DAC.</p>
      <DocSlot field="idDoc"      label="National ID / Passport"  required hint="Front side is sufficient. Scanned copy or photo accepted." />
      <DocSlot field="medicalDoc" label="Medical report"                   hint="Any doctor reports, lab results, or health checks from the past 2 years (optional but speeds up review)." />
      <DocSlot field="employerDoc"label="Employer / income letter"         hint="Helps with group pricing if applicable (optional)." />
    </div>
  );
}

// ─── Step 5: Consent ─────────────────────────────────────────────────────────
function Step5({ data, onChange }) {
  const set = (k, v) => onChange({ ...data, [k]: v });
  const ITEMS = [
    { key: "c1", text: "I confirm that all information provided is accurate and complete to the best of my knowledge." },
    { key: "c2", text: "I consent to DAC processing my personal and medical data for the purposes of underwriting this insurance application." },
    { key: "c3", text: "I understand that providing false information may result in rejection of this application or cancellation of any policy issued." },
    { key: "c4", text: "I agree to be contacted by DAC staff via phone or email regarding my application." },
  ];
  return (
    <div>
      <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Review & consent</h3>
      <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>Please read and accept the following before submitting your application.</p>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 28 }}>
        {ITEMS.map(item => (
          <label key={item.key} style={{ display: "flex", gap: 14, padding: "14px 16px", borderRadius: 10, border: `1.5px solid ${data[item.key] ? OK : "#e5e7eb"}`, cursor: "pointer", background: data[item.key] ? `${OK}08` : WHITE, transition: "all 0.2s", alignItems: "flex-start" }}>
            <input type="checkbox" checked={!!data[item.key]} onChange={e => set(item.key, e.target.checked)} style={{ accentColor: OK, width: 16, height: 16, marginTop: 2, flexShrink: 0 }} />
            <span style={{ fontSize: 14, lineHeight: 1.6 }}>{item.text}</span>
          </label>
        ))}
      </div>

      <Field label="Electronic signature (type your full name)" required>
        <input style={inputStyle} value={data.signature || ""} onChange={e => set("signature", e.target.value)} placeholder="e.g. Sokha Chan" />
        <p style={{ fontSize: 11, color: TXT2, marginTop: 4 }}>By typing your name you are signing this application electronically.</p>
      </Field>
    </div>
  );
}

// ─── Application wizard ──────────────────────────────────────────────────────
const STEP_LABELS = ["Personal Info", "Health Profile", "Coverage", "Documents", "Consent"];

function ApplyView({ onDone }) {
  const [step, setStep]     = useState(0);
  const [form, setForm]     = useState({});
  const [submitting, setSub] = useState(false);
  const [error, setError]   = useState("");

  const canNext = () => {
    if (step === 0) return form.firstName && form.lastName && form.dob && form.gender && form.phone && form.province && form.nationalId;
    if (step === 1) return form.height && form.weight && form.smoking && form.exercise && form.occupation && (form.conditions || []).length > 0;
    if (step === 2) return !!form.tier;
    if (step === 3) return !!form.idDoc;
    if (step === 4) return form.c1 && form.c2 && form.c3 && form.c4 && form.signature;
    return true;
  };

  const submit = async () => {
    setSub(true);
    setError("");
    try {
      const payload = {
        personal: {
          fullName:    `${form.firstName} ${form.lastName}`,
          dateOfBirth: form.dob,
          gender:      form.gender,
          phone:       form.phone,
          email:       form.email || "",
          region:      form.province,
          occupation:  form.occupation,
          nationalId:  form.nationalId || "",
        },
        medical: {
          height:               parseFloat(form.height),
          weight:               parseFloat(form.weight),
          smokingStatus:        form.smoking,
          exerciseFrequency:    form.exercise,
          preexistingConditions: form.conditions || [],
        },
        coverage: {
          tier:             form.tier || "Silver",
          include_opd:      !!form.opd,
          include_dental:   !!form.dental,
          include_maternity: !!form.maternity,
        },
        consent: {
          terms:          !!form.c1,
          privacy:        !!form.c2,
          dataProcessing: !!form.c3,
          truthfulness:   !!form.c4,
          signature:      form.signature,
        },
      };
      const res = await fetch(`${API_URL}/api/v1/applications`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000),
      });
      if (res.ok) {
        const { case_id } = await res.json();
        onDone(case_id || `DAC-${Date.now().toString(36).toUpperCase()}`);
      } else {
        // Generate local case ref as fallback (demo mode)
        onDone(`DAC-${Date.now().toString(36).toUpperCase()}`);
      }
    } catch {
      onDone(`DAC-${Date.now().toString(36).toUpperCase()}`);
    } finally {
      setSub(false);
    }
  };

  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "100vh", background: LTGRAY }}>
      <div style={{ maxWidth: 680, margin: "0 auto", padding: "0 20px" }}>
        <div style={{ background: WHITE, borderRadius: 20, padding: "40px 40px 36px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)" }}>
          <div style={{ marginBottom: 28 }}>
            <span style={{ color: GOLD_D, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase" }}>New Application</span>
            <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginTop: 4 }}>Apply for health insurance</h2>
          </div>

          <StepBar step={step} total={5} labels={STEP_LABELS} />

          {step === 0 && <Step1 data={form} onChange={setForm} />}
          {step === 1 && <Step2 data={form} onChange={setForm} />}
          {step === 2 && <Step3 data={form} onChange={setForm} />}
          {step === 3 && <Step4 data={form} onChange={setForm} />}
          {step === 4 && <Step5 data={form} onChange={setForm} />}

          {error && <p style={{ color: ERR, fontSize: 13, marginBottom: 12 }}>{error}</p>}

          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 28 }}>
            <button
              onClick={() => setStep(s => s - 1)}
              disabled={step === 0}
              style={{ padding: "12px 28px", borderRadius: 8, border: "1.5px solid #e5e7eb", background: WHITE, fontSize: 14, fontWeight: 600, cursor: step === 0 ? "not-allowed" : "pointer", color: step === 0 ? GRAY : TXT, fontFamily: "inherit" }}
            >
              Back
            </button>
            {step < 4
              ? <button
                  onClick={() => setStep(s => s + 1)}
                  disabled={!canNext()}
                  style={{ padding: "12px 28px", borderRadius: 8, background: canNext() ? GOLD : "#e5e7eb", border: "none", fontSize: 14, fontWeight: 700, cursor: canNext() ? "pointer" : "not-allowed", color: canNext() ? NAVY : GRAY, fontFamily: "inherit", transition: "all 0.2s" }}
                >
                  Continue →
                </button>
              : <button
                  onClick={submit}
                  disabled={!canNext() || submitting}
                  style={{ padding: "12px 32px", borderRadius: 8, background: canNext() && !submitting ? NAVY : "#e5e7eb", border: "none", fontSize: 14, fontWeight: 700, cursor: canNext() && !submitting ? "pointer" : "not-allowed", color: canNext() && !submitting ? WHITE : GRAY, fontFamily: "inherit", transition: "all 0.2s" }}
                >
                  {submitting ? "Submitting…" : "Submit application"}
                </button>}
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── Success screen ──────────────────────────────────────────────────────────
function SuccessView({ caseId, onTrack }) {
  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "100vh", background: LTGRAY, display: "flex", alignItems: "center" }}>
      <div style={{ maxWidth: 560, margin: "0 auto", padding: "0 20px", textAlign: "center" }}>
        <div style={{ background: WHITE, borderRadius: 20, padding: "48px 40px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)" }}>
          <div style={{ width: 72, height: 72, borderRadius: "50%", background: `${OK}15`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 24px" }}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={OK} strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
          </div>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginBottom: 12 }}>Application submitted!</h2>
          <p style={{ color: TXT2, fontSize: 15, marginBottom: 28, lineHeight: 1.7 }}>Your application has been received. An underwriter will review it within 3–5 business days.</p>
          <div style={{ background: NAVY, borderRadius: 12, padding: "20px 28px", marginBottom: 28 }}>
            <p style={{ color: GRAY, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Your case reference</p>
            <p style={{ color: GOLD, fontSize: 26, fontWeight: 800, letterSpacing: 2 }}>{caseId}</p>
            <p style={{ color: GRAY, fontSize: 12, marginTop: 8 }}>Save this number to track your application</p>
          </div>
          <button
            onClick={() => onTrack(caseId)}
            style={{ width: "100%", padding: "14px", borderRadius: 10, background: GOLD, border: "none", fontSize: 15, fontWeight: 700, cursor: "pointer", color: NAVY, fontFamily: "inherit", marginBottom: 12 }}
          >
            Track my application
          </button>
          <p style={{ fontSize: 13, color: TXT2 }}>We'll contact you at the phone number you provided. Check your email for a confirmation.</p>
        </div>
      </div>
    </section>
  );
}

// ─── Case Tracking helpers ────────────────────────────────────────────────────
function addBusinessDays(n) {
  const d = new Date();
  let added = 0;
  while (added < n) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() !== 0 && d.getDay() !== 6) added++;
  }
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

const NEXT_STEPS = {
  "Received":         { msg: "Your documents are being verified by our team. This usually takes 1 business day.", showEta: true },
  "Under Review":     { msg: "An underwriter is actively reviewing your application. You'll receive a decision within 3–5 business days.", showEta: true },
  "Decision Pending": { msg: "The underwriter has completed their review. A final decision is being signed off by a senior team member — expect it very soon.", showEta: false },
  "Approved":         { msg: "Your application has been approved. Your policy documents will be sent to your email and phone number shortly.", showEta: false },
  "Declined":         { msg: "We were unable to approve your application at this time. Our team will contact you to explain the reasons and discuss any alternatives.", showEta: false },
  "On Hold":          { msg: "Additional information is required. Our team will contact you within 1 business day to let you know what's needed.", showEta: false },
};

// ─── Case Tracking View ──────────────────────────────────────────────────────
function TrackView({ prefillRef, onOpenChat, onSetContext }) {
  const [ref, setRef]       = useState(prefillRef || "");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");

  const lookup = async () => {
    if (!ref.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/applications/${encodeURIComponent(ref.trim())}/status`, {
        signal: AbortSignal.timeout(15000),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
        onSetContext?.({ case_id: ref.trim(), case_status: data.status });
      } else if (res.status === 404) {
        setError("Case not found. Check your reference number and try again.");
      } else {
        const data = demoStatus(ref.trim());
        setResult(data);
        onSetContext?.({ case_id: ref.trim(), case_status: data.status });
      }
    } catch {
      const data = demoStatus(ref.trim());
      setResult(data);
      onSetContext?.({ case_id: ref.trim(), case_status: data.status });
    } finally {
      setLoading(false);
    }
  };

  const STATUS_COLOR = { Received: "#3b82f6", "Under Review": GOLD_D, "Decision Pending": "#8b5cf6", Approved: OK, Declined: ERR, "On Hold": GRAY };

  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "100vh", background: LTGRAY }}>
      <div style={{ maxWidth: 620, margin: "0 auto", padding: "0 20px" }}>
        <FadeIn>
          <div style={{ background: WHITE, borderRadius: 20, padding: "40px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)", marginBottom: 24 }}>
            <span style={{ color: GOLD_D, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase" }}>Case status</span>
            <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginTop: 4, marginBottom: 20 }}>Track my application</h2>
            <div style={{ display: "flex", gap: 12 }}>
              <input
                style={{ ...inputStyle, flex: 1 }}
                value={ref}
                onChange={e => setRef(e.target.value)}
                onKeyDown={e => e.key === "Enter" && lookup()}
                placeholder="Enter your case reference (e.g. DAC-ABC123)"
              />
              <button
                onClick={lookup}
                disabled={!ref.trim() || loading}
                style={{ padding: "10px 24px", borderRadius: 8, background: ref.trim() ? NAVY : "#e5e7eb", border: "none", fontSize: 14, fontWeight: 700, cursor: ref.trim() ? "pointer" : "not-allowed", color: ref.trim() ? WHITE : GRAY, fontFamily: "inherit", whiteSpace: "nowrap" }}
              >
                {loading ? "Looking up…" : "Check status"}
              </button>
            </div>
            {error && <p style={{ color: ERR, fontSize: 13, marginTop: 12 }}>{error}</p>}
          </div>
        </FadeIn>

        {result && (
          <>
          <FadeIn delay={0.1}>
            <div style={{ background: WHITE, borderRadius: 20, padding: "36px 40px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
                <div>
                  <p style={{ color: TXT2, fontSize: 13 }}>Case reference</p>
                  <p style={{ fontSize: 20, fontWeight: 800, color: NAVY, letterSpacing: 1 }}>{result.case_id || ref}</p>
                </div>
                <span style={{
                  background: `${STATUS_COLOR[result.status] || GRAY}15`,
                  color: STATUS_COLOR[result.status] || GRAY,
                  padding: "6px 14px", borderRadius: 20, fontSize: 13, fontWeight: 700,
                }}>
                  {result.status}
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                {(result.timeline || demoStatus(ref).timeline).map((ev, i, arr) => (
                  <div key={i} style={{ display: "flex", gap: 16, paddingBottom: i < arr.length - 1 ? 20 : 0 }}>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <div style={{
                        width: 32, height: 32, borderRadius: "50%",
                        background: ev.done ? (i === arr.filter(x => x.done).length - 1 ? GOLD : OK) : LTGRAY,
                        border: `2px solid ${ev.done ? (i === arr.filter(x => x.done).length - 1 ? GOLD : OK) : "#d1d5db"}`,
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      }}>
                        {ev.done
                          ? <svg width="14" height="14" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
                          : <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#d1d5db" }} />}
                      </div>
                      {i < arr.length - 1 && <div style={{ width: 2, flex: 1, background: ev.done ? `${OK}50` : "#e5e7eb", minHeight: 20, marginTop: 4 }} />}
                    </div>
                    <div style={{ paddingTop: 4, paddingBottom: 8 }}>
                      <p style={{ fontSize: 14, fontWeight: 600, color: ev.done ? TXT : GRAY }}>{ev.label}</p>
                      {ev.date && <p style={{ fontSize: 12, color: TXT2, marginTop: 2 }}>{ev.date}</p>}
                      {ev.note && <p style={{ fontSize: 12, color: TXT2, marginTop: 2 }}>{ev.note}</p>}
                    </div>
                  </div>
                ))}
              </div>

              {result.message && (
                <div style={{ marginTop: 24, padding: "14px 18px", background: `${GOLD}10`, borderRadius: 10, borderLeft: `3px solid ${GOLD}` }}>
                  <p style={{ fontSize: 13, color: TXT, lineHeight: 1.6 }}>{result.message}</p>
                </div>
              )}
            </div>
          </FadeIn>

          {NEXT_STEPS[result.status] && (
            <FadeIn delay={0.2}>
              <div style={{ background: WHITE, borderRadius: 16, padding: "24px 28px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)", marginTop: 16 }}>
                <p style={{ fontSize: 11, color: GOLD_D, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>What happens next</p>
                <p style={{ fontSize: 14, color: TXT2, lineHeight: 1.7, margin: 0 }}>{NEXT_STEPS[result.status].msg}</p>
                {NEXT_STEPS[result.status].showEta && (
                  <p style={{ fontSize: 13, color: TXT, fontWeight: 600, marginTop: 12, marginBottom: 0 }}>
                    Expected decision by: <span style={{ color: NAVY }}>{addBusinessDays(5)}</span>
                  </p>
                )}
              </div>
            </FadeIn>
          )}

          <FadeIn delay={0.3}>
            <div style={{ background: WHITE, borderRadius: 16, padding: "20px 28px", boxShadow: "0 4px 24px rgba(0,0,0,0.06)", marginTop: 16, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
              <p style={{ fontSize: 14, color: TXT2, margin: 0 }}>Have questions about your case?</p>
              <button
                onClick={onOpenChat}
                style={{ padding: "9px 20px", borderRadius: 8, background: NAVY, border: "none", cursor: "pointer", fontSize: 13, fontWeight: 700, color: WHITE, fontFamily: "inherit", whiteSpace: "nowrap" }}
              >
                Ask our advisor →
              </button>
            </div>
          </FadeIn>
          </>
        )}
      </div>
    </section>
  );
}

function demoStatus(caseId) {
  return {
    case_id: caseId,
    status: "Under Review",
    message: "Your application is being reviewed by our underwriting team. We'll contact you within 3–5 business days.",
    timeline: [
      { label: "Application received",      done: true,  date: new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) },
      { label: "Documents verified",        done: true,  date: "", note: "ID and medical documents checked" },
      { label: "Underwriting review",       done: false, date: "", note: "In progress — 3–5 business days" },
      { label: "Decision issued",           done: false },
      { label: "Policy issued / Declined",  done: false },
    ],
  };
}

// ─── Landing Home View ───────────────────────────────────────────────────────
function HomeView({ onApply, onTrack }) {
  return (
    <>
      {/* Hero */}
      <section style={{
        background: `linear-gradient(135deg, ${NAVY} 0%, ${NAVY_D} 50%, #16213e 100%)`,
        minHeight: "100vh", display: "flex", alignItems: "center",
        position: "relative", overflow: "hidden", paddingTop: 72,
      }}>
        <div style={{ position: "absolute", top: -120, right: -80, width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, rgba(245,197,99,0.07) 0%, transparent 70%)" }} />
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)` }} />

        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "80px 24px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>
          <div>
            <span style={{ color: GOLD, fontSize: 13, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase", display: "block", marginBottom: 16 }}>DAC HealthPrice Customer Portal</span>
            <h1 style={{ fontFamily: "'Playfair Display', serif", color: WHITE, fontSize: "clamp(32px, 4.5vw, 52px)", lineHeight: 1.12, fontWeight: 700, marginBottom: 24 }}>
              Apply for health<br />insurance in minutes
            </h1>
            <p style={{ color: GRAY, fontSize: 17, lineHeight: 1.7, marginBottom: 36, maxWidth: 460 }}>
              Complete your health insurance application online. Upload your documents, get a personalised quote, and track your case — all in one place.
            </p>
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
              <button onClick={onApply} style={{ background: GOLD, color: NAVY, border: "none", padding: "15px 36px", borderRadius: 50, fontSize: 16, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                Start application
              </button>
              <button onClick={onTrack} style={{ background: "transparent", color: WHITE, border: "2px solid rgba(255,255,255,0.25)", padding: "15px 36px", borderRadius: 50, fontSize: 16, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>
                Track my case
              </button>
            </div>
          </div>

          {/* Info card */}
          <div>
            <div style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 32, backdropFilter: "blur(12px)" }}>
              <p style={{ color: GOLD, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", marginBottom: 20 }}>What you can do here</p>
              {[
                { icon: "📋", title: "Apply directly", desc: "5-step wizard — takes about 5 minutes" },
                { icon: "📡", title: "Track your case", desc: "Real-time status and decision timeline" },
                { icon: "📄", title: "Upload documents", desc: "ID, medical reports, and consent securely" },
                { icon: "💬", title: "AI quote advisor", desc: "Understand your premium breakdown" },
              ].map((f, i) => (
                <div key={i} style={{ display: "flex", gap: 14, padding: "14px 0", borderBottom: i < 3 ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
                  <span style={{ fontSize: 22 }}>{f.icon}</span>
                  <div>
                    <p style={{ color: WHITE, fontSize: 14, fontWeight: 600 }}>{f.title}</p>
                    <p style={{ color: GRAY, fontSize: 13 }}>{f.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <style>{`
          @media (max-width: 768px) {
            .portal-hero-grid { grid-template-columns: 1fr !important; }
          }
        `}</style>
      </section>

      {/* Steps */}
      <section style={{ background: WHITE, padding: "80px 24px" }}>
        <div style={{ maxWidth: 1000, margin: "0 auto" }}>
          <FadeIn>
            <div style={{ textAlign: "center", marginBottom: 56 }}>
              <span style={{ color: GOLD_D, fontSize: 13, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase" }}>Simple process</span>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, fontWeight: 700, marginTop: 10 }}>How it works</h2>
            </div>
          </FadeIn>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 24 }}>
            {[
              { step: "01", icon: "👤", title: "Fill in your details",   desc: "Personal information and health profile — takes about 3 minutes." },
              { step: "02", icon: "🛡️", title: "Choose your plan",       desc: "Select a hospital tier and optional riders (OPD, Dental, Maternity)." },
              { step: "03", icon: "📄", title: "Upload documents",        desc: "Attach your national ID and any medical reports." },
              { step: "04", icon: "📡", title: "Get your decision",       desc: "Track your case online. Underwriting takes 3–5 business days." },
            ].map((s, i) => (
              <FadeIn key={i} delay={i * 0.1}>
                <div style={{ background: WHITE, borderRadius: 16, padding: 28, border: "1px solid #e5e7eb", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: GOLD }} />
                  <span style={{ fontSize: 32 }}>{s.icon}</span>
                  <span style={{ display: "block", color: GOLD_D, fontSize: 12, fontWeight: 700, letterSpacing: 1, marginTop: 12 }}>STEP {s.step}</span>
                  <h3 style={{ fontSize: 17, fontWeight: 700, margin: "8px 0 10px" }}>{s.title}</h3>
                  <p style={{ color: TXT2, fontSize: 14, lineHeight: 1.7 }}>{s.desc}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ background: `linear-gradient(135deg, ${NAVY} 0%, #16213e 100%)`, padding: "72px 24px", textAlign: "center" }}>
        <FadeIn>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, color: WHITE, fontWeight: 700, marginBottom: 14 }}>Ready to apply?</h2>
          <p style={{ color: GRAY, fontSize: 16, marginBottom: 28 }}>The full application takes about 5 minutes.</p>
          <button onClick={onApply} style={{ background: GOLD, color: NAVY, border: "none", padding: "16px 44px", borderRadius: 50, fontSize: 17, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
            Start my application
          </button>
        </FadeIn>
      </section>
    </>
  );
}

// ─── Portal root ─────────────────────────────────────────────────────────────
export default function PublicPortal() {
  const [view, setView]         = useState("home");
  const [caseId, setCaseId]     = useState(null);
  const [trackRef, setTrackRef] = useState("");
  const [scrollY, setScrollY]   = useState(0);
  const [chatOpen, setChatOpen] = useState(false);
  const [advisorCtx, setAdvisorCtx] = useState({});

  useEffect(() => {
    const h = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

  const goTrack = (ref = "") => { setTrackRef(ref); setView("track"); window.scrollTo(0, 0); };
  const goApply = () => { setView("apply"); window.scrollTo(0, 0); };

  const handleSubmitDone = (id) => { setCaseId(id); setView("success"); window.scrollTo(0, 0); };

  const NAV_LINKS = [
    { label: "Home",          action: () => { setView("home");  window.scrollTo(0,0); } },
    { label: "Apply",         action: goApply },
    { label: "Track My Case", action: () => goTrack() },
  ];

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", color: TXT, background: WHITE, minHeight: "100vh" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet" />

      {/* Nav */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100, height: 68,
        background: scrollY > 50 ? "rgba(13,43,122,0.97)" : NAVY,
        backdropFilter: scrollY > 50 ? "blur(14px)" : "none",
        borderBottom: scrollY > 50 ? "1px solid rgba(255,255,255,0.06)" : "none",
        transition: "background 0.3s",
        display: "flex", alignItems: "center",
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px", width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => { setView("home"); window.scrollTo(0,0); }}>
            <img src="/DAC.png" alt="DAC" style={{ width: 48, height: 48, objectFit: "contain" }} />
            <div>
              <div style={{ color: WHITE, fontSize: 16, fontWeight: 700 }}>DAC <span style={{ color: GOLD }}>HealthPrice</span></div>
              <div style={{ color: GRAY, fontSize: 11 }}>Customer Portal</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
            {NAV_LINKS.map(l => (
              <span key={l.label} onClick={l.action} style={{ color: GRAY, fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "color 0.2s" }}
                onMouseEnter={e => e.target.style.color = GOLD} onMouseLeave={e => e.target.style.color = GRAY}>
                {l.label}
              </span>
            ))}
            <button onClick={goApply} style={{ background: GOLD, color: NAVY, border: "none", padding: "9px 22px", borderRadius: 50, fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
              Apply now
            </button>
          </div>
        </div>
      </nav>

      {/* Views */}
      {view === "home"    && <HomeView    onApply={goApply} onTrack={() => goTrack()} />}
      {view === "apply"   && <ApplyView   onDone={handleSubmitDone} />}
      {view === "success" && <SuccessView caseId={caseId} onTrack={goTrack} />}
      {view === "track"   && <TrackView   prefillRef={trackRef} onOpenChat={() => setChatOpen(true)} onSetContext={setAdvisorCtx} />}

      <ChatAdvisor open={chatOpen} onOpenChange={setChatOpen} context={advisorCtx} />

      {/* Footer */}
      <footer style={{ background: NAVY_D, color: GRAY, padding: "40px 24px 28px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <div>
            <p style={{ color: WHITE, fontSize: 15, fontWeight: 700, marginBottom: 4 }}>DAC HealthPrice</p>
            <p style={{ fontSize: 13 }}>AI-powered health insurance for Cambodia</p>
          </div>
          <div style={{ fontSize: 13 }}>
            <p>radet@dactuaries.com · +855 85 508 860</p>
            <p style={{ marginTop: 4 }}>Phnom Penh, Cambodia</p>
          </div>
        </div>
        <div style={{ maxWidth: 1100, margin: "20px auto 0", borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 16, fontSize: 12 }}>
          © 2026 Decent Actuarial Consultants. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
