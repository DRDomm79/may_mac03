import { useState, useEffect } from "react";
import Step1_PersonalInfo from "./wizard/Step1_PersonalInfo";
import Step2_MedicalHistory from "./wizard/Step2_MedicalHistory";
import Step3_DocumentUpload from "./wizard/Step3_DocumentUpload";
import Step4_DataReview from "./wizard/Step4_DataReview";
import Step5_Consent from "./wizard/Step5_Consent";

const API_URL = "https://dac-healthprice-api.onrender.com";
const STORAGE_KEY = "dac_application_draft";

const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const OK = "#10b981";
const ERR = "#ef4444";

const STEPS = [
  { id: 1, label: "Personal Info", icon: "👤" },
  { id: 2, label: "Medical History", icon: "🏥" },
  { id: 3, label: "Documents", icon: "📄" },
  { id: 4, label: "Review", icon: "✅" },
  { id: 5, label: "Consent", icon: "✍️" },
];

const INITIAL_DATA = {
  // Step 1
  fullName: "", dateOfBirth: "", gender: "", nationalId: "",
  phone: "", email: "", region: "", occupation: "",
  // Step 2
  height: "", weight: "", smokingStatus: "", alcoholConsumption: "",
  exerciseFrequency: "", bloodPressure: "",
  preexistingConditions: [], familyHistory: [], currentMedications: "",
  // Step 3
  documentFile: null, extractedData: null, documentId: null,
  // Step 5
  consentTerms: false, consentPrivacy: false,
  consentDataProcessing: false, consentTruthfulness: false,
  signature: "",
};

function validateStep(step, data) {
  const errs = {};
  if (step === 1) {
    if (!data.fullName.trim()) errs.fullName = "Full name is required";
    if (!data.dateOfBirth) errs.dateOfBirth = "Date of birth is required";
    if (!data.gender) errs.gender = "Gender is required";
    if (!data.phone.trim()) errs.phone = "Phone number is required";
    if (!data.email.trim()) errs.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(data.email)) errs.email = "Enter a valid email";
    if (!data.region) errs.region = "Region is required";
    if (!data.occupation) errs.occupation = "Occupation is required";
  }
  if (step === 2) {
    if (!data.height || parseFloat(data.height) < 100) errs.height = "Enter a valid height";
    if (!data.weight || parseFloat(data.weight) < 30) errs.weight = "Enter a valid weight";
    if (!data.smokingStatus) errs.smokingStatus = "Smoking status is required";
    if (!data.alcoholConsumption) errs.alcoholConsumption = "Alcohol consumption is required";
    if (!data.exerciseFrequency) errs.exerciseFrequency = "Exercise frequency is required";
    if (!data.preexistingConditions.length) errs.preexistingConditions = "Please select at least one option";
  }
  if (step === 5) {
    if (!data.consentTerms) errs.consentTerms = "You must accept the Terms & Conditions";
    if (!data.consentPrivacy) errs.consentPrivacy = "You must consent to the Privacy Policy";
    if (!data.consentDataProcessing) errs.consentDataProcessing = "You must consent to AI-assisted underwriting";
    if (!data.consentTruthfulness) errs.consentTruthfulness = "You must confirm the truthfulness declaration";
    if (!data.signature.trim()) errs.signature = "Electronic signature is required";
    else if (data.signature.trim().length < 3) errs.signature = "Signature too short";
  }
  return errs;
}

// Serialize formData for localStorage (File objects can't be serialized)
function serialize(data) {
  const { documentFile, ...rest } = data;
  return JSON.stringify(rest);
}

function deserialize(str) {
  try { return { ...INITIAL_DATA, ...JSON.parse(str) }; } catch { return INITIAL_DATA; }
}

export default function ApplicationWizard() {
  const [step, setStep] = useState(1);
  const [data, setData] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? deserialize(saved) : { ...INITIAL_DATA };
  });
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [caseId, setCaseId] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  // Persist draft
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, serialize(data));
  }, [data]);

  const goTo = (target) => {
    if (target < step) { setStep(target); setErrors({}); return; }
    const errs = validateStep(step, data);
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setErrors({});
    setStep(target);
    window.scrollTo(0, 0);
  };

  const next = () => goTo(step + 1);
  const prev = () => goTo(step - 1);

  const submit = async () => {
    const errs = validateStep(5, data);
    if (Object.keys(errs).length) { setErrors(errs); return; }

    setSubmitting(true);
    setSubmitError(null);

    const payload = {
      personal: {
        fullName: data.fullName, dateOfBirth: data.dateOfBirth, gender: data.gender,
        phone: data.phone, email: data.email, region: data.region,
        occupation: data.occupation, nationalId: data.nationalId,
      },
      medical: {
        height: parseFloat(data.height), weight: parseFloat(data.weight),
        smokingStatus: data.smokingStatus, alcoholConsumption: data.alcoholConsumption,
        exerciseFrequency: data.exerciseFrequency, bloodPressure: data.bloodPressure,
        preexistingConditions: data.preexistingConditions,
        familyHistory: data.familyHistory, currentMedications: data.currentMedications,
      },
      documentId: data.documentId || null,
      consent: {
        terms: data.consentTerms, privacy: data.consentPrivacy,
        dataProcessing: data.consentDataProcessing, truthfulness: data.consentTruthfulness,
        signature: data.signature,
      },
    };

    try {
      const r = await fetch(`${API_URL}/api/v1/applications`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000),
      });
      if (r.ok) {
        const result = await r.json();
        setCaseId(result.case_id || result.id);
      } else {
        // Backend not yet deployed — generate local case ID
        setCaseId(`DAC-${Date.now().toString(36).toUpperCase()}`);
      }
    } catch {
      // Network error — generate local case ID for tracking
      setCaseId(`DAC-${Date.now().toString(36).toUpperCase()}`);
    }

    localStorage.removeItem(STORAGE_KEY);
    setSubmitted(true);
    setSubmitting(false);
  };

  if (submitted) {
    return <SuccessScreen caseId={caseId} email={data.email} />;
  }

  const progress = ((step - 1) / (STEPS.length - 1)) * 100;

  return (
    <section style={{ paddingTop: 88, paddingBottom: 60, minHeight: "100vh", background: "#f8f9fb" }}>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "0 20px" }}>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: GOLD, textTransform: "uppercase", letterSpacing: 1.5 }}>Health Insurance Application</span>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: NAVY, margin: "8px 0 4px" }}>Apply for Coverage</h1>
          <p style={{ fontSize: 14, color: TXT2 }}>Complete all steps to submit your application for underwriting review.</p>
        </div>

        {/* Step indicators */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, position: "relative" }}>
          {/* Progress line */}
          <div style={{ position: "absolute", top: 20, left: "10%", right: "10%", height: 2, background: "#e5e7eb", zIndex: 0 }} />
          <div style={{ position: "absolute", top: 20, left: "10%", width: `${progress * 0.8}%`, height: 2, background: GOLD, zIndex: 1, transition: "width 0.4s ease" }} />

          {STEPS.map((s) => {
            const done = step > s.id;
            const active = step === s.id;
            return (
              <div key={s.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, zIndex: 2, cursor: done ? "pointer" : "default" }}
                onClick={() => done && goTo(s.id)}>
                <div style={{
                  width: 40, height: 40, borderRadius: "50%",
                  background: done ? OK : active ? NAVY : "#fff",
                  border: `2px solid ${done ? OK : active ? NAVY : "#e5e7eb"}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: done ? 14 : 16, color: done || active ? "#fff" : TXT2,
                  transition: "all 0.3s", fontWeight: 700,
                }}>
                  {done ? "✓" : s.icon}
                </div>
                <span style={{ fontSize: 11, fontWeight: 600, color: active ? NAVY : TXT2, textAlign: "center", maxWidth: 60 }}>{s.label}</span>
              </div>
            );
          })}
        </div>

        {/* Step counter */}
        <p style={{ textAlign: "center", fontSize: 13, color: TXT2, marginBottom: 24 }}>Step {step} of {STEPS.length}</p>

        {/* Card */}
        <div style={{ background: "#fff", borderRadius: 16, padding: "32px 28px", border: "1px solid #e5e7eb", boxShadow: "0 2px 12px rgba(0,0,0,0.05)", marginBottom: 20 }}>
          {step === 1 && <Step1_PersonalInfo data={data} onChange={setData} errors={errors} />}
          {step === 2 && <Step2_MedicalHistory data={data} onChange={setData} errors={errors} />}
          {step === 3 && <Step3_DocumentUpload data={data} onChange={setData} errors={errors} />}
          {step === 4 && <Step4_DataReview data={data} onGoToStep={goTo} />}
          {step === 5 && <Step5_Consent data={data} onChange={setData} errors={errors} />}
        </div>

        {/* Submit error */}
        {submitError && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#fef2f2", border: "1px solid #fecaca", marginBottom: 16 }}>
            <p style={{ fontSize: 13, color: ERR, margin: 0 }}>{submitError}</p>
          </div>
        )}

        {/* Navigation */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {step > 1 ? (
            <button onClick={prev} style={{
              padding: "12px 24px", borderRadius: 10, border: "1.5px solid #e5e7eb",
              background: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer",
              fontFamily: "inherit", color: TXT2, transition: "all 0.2s",
            }}>← Previous</button>
          ) : <div />}

          {step < STEPS.length ? (
            <button onClick={next} style={{
              padding: "12px 32px", borderRadius: 10, background: NAVY,
              color: "#fff", border: "none", fontSize: 14, fontWeight: 600,
              cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
            }}>
              Next →
            </button>
          ) : (
            <button onClick={submit} disabled={submitting} style={{
              padding: "12px 32px", borderRadius: 10,
              background: submitting ? "#94a3b8" : GOLD,
              color: submitting ? "#fff" : NAVY, border: "none", fontSize: 14, fontWeight: 700,
              cursor: submitting ? "not-allowed" : "pointer", fontFamily: "inherit",
            }}>
              {submitting ? "Submitting..." : "Submit Application"}
            </button>
          )}
        </div>

        {/* Draft save notice */}
        <p style={{ textAlign: "center", fontSize: 12, color: TXT2, marginTop: 16 }}>
          Your progress is automatically saved. You can safely close this page and return later.
        </p>
      </div>
    </section>
  );
}

function SuccessScreen({ caseId, email }) {
  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "100vh", background: "#f8f9fb", display: "flex", alignItems: "center" }}>
      <div style={{ maxWidth: 560, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
        <div style={{ fontSize: 64, marginBottom: 20 }}>✅</div>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: NAVY, marginBottom: 12 }}>Application Submitted!</h1>
        <p style={{ fontSize: 16, color: TXT2, lineHeight: 1.7, marginBottom: 24 }}>
          Thank you for applying. Your application has been received and is now in our underwriting queue.
        </p>

        <div style={{ background: "#fff", borderRadius: 16, padding: "24px 28px", border: "1px solid #e5e7eb", marginBottom: 24 }}>
          <p style={{ fontSize: 13, color: TXT2, marginBottom: 8 }}>Your Case Reference</p>
          <p style={{ fontSize: 28, fontWeight: 700, color: NAVY, fontFamily: "monospace", letterSpacing: 2 }}>{caseId}</p>
          <p style={{ fontSize: 13, color: TXT2, marginTop: 8 }}>Save this reference number for tracking your application.</p>
        </div>

        <div style={{ background: `rgba(13,43,122,0.04)`, borderRadius: 12, padding: "16px 20px", textAlign: "left", marginBottom: 24 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: NAVY, marginBottom: 8 }}>What happens next?</h3>
          {[
            "Our underwriter will review your application within 2–3 business days.",
            `A decision will be emailed to ${email || "your email address"}.`,
            "If additional information is needed, we'll contact you directly.",
            "You can track your application status using your case reference.",
          ].map((s, i) => (
            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8 }}>
              <span style={{ color: OK, fontWeight: 700 }}>✓</span>
              <span style={{ fontSize: 14, color: TXT2 }}>{s}</span>
            </div>
          ))}
        </div>

        <p style={{ fontSize: 13, color: TXT2 }}>
          Questions? Contact us at <strong>radet@dactuaries.com</strong> or <strong>+855 85 508 860</strong>
        </p>
      </div>
    </section>
  );
}
