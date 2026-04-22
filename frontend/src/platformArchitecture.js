export const NODE_CATEGORIES = {
  "staff-page": { color: "#0d2b7a", text: "#ffffff", badge: "Staff Page" },
  "workbench":  { color: "#0d9488", text: "#ffffff", badge: "Workbench" },
  "admin-tab":  { color: "#7c3aed", text: "#ffffff", badge: "Admin Panel" },
  "backend":    { color: "#1e293b", text: "#f1f5f9", badge: "Backend" },
  "portal":     { color: "#059669", text: "#ffffff", badge: "Public Portal" },
};

// All x,y,w,h are in SVG viewBox coordinates (0 0 960 480)
export const PLATFORM_NODES = [
  // ─── Main staff pages ─────────────────────────────────────────
  {
    id: "home",
    label: "Home",
    category: "staff-page",
    x: 34, y: 90, w: 120, h: 40,
    description:
      "Staff portal landing page with animated platform statistics, a how-it-works walkthrough, and a quick-start CTA that routes to New Case. No API calls — purely presentational.",
    endpoints: [],
    tags: ["UI only", "animated stats"],
    status: "live",
  },
  {
    id: "new-case",
    label: "New Case",
    category: "staff-page",
    x: 188, y: 90, w: 120, h: 40,
    description:
      "Multi-step application wizard for health insurance underwriting. Collects applicant demographics, medical history, lifestyle factors, and coverage selection across 5 wizard steps. Supports document upload (National ID, medical reports). Submits to PostgreSQL via the applications API.",
    endpoints: [
      "POST /api/v1/applications",
      "POST /api/v1/applications/{id}/documents",
    ],
    tags: ["5-step wizard", "file upload", "JWT required"],
    status: "live",
  },
  {
    id: "case-pipeline",
    label: "Case Pipeline",
    category: "staff-page",
    x: 342, y: 90, w: 120, h: 40,
    description:
      "Underwriter review queue showing all submitted and in-review applications. Each case displays AI-computed risk score (0–100) from age, BMI, smoking, and conditions. Underwriters can approve, decline, or refer — every decision is logged to the audit trail.",
    endpoints: [
      "GET /api/v1/applications?status=submitted,in_review",
      "POST /api/v1/applications/{id}/decision",
    ],
    tags: ["review queue", "risk scoring", "audit trail"],
    status: "live",
  },
  {
    id: "life-insurance",
    label: "Life Insurance",
    category: "staff-page",
    x: 496, y: 90, w: 120, h: 40,
    description:
      "Interactive actuarial workbench for health and life insurance pricing. Contains four sub-tabs: Attribution (GLM factor breakdown), Sensitivity (one-at-a-time sweep), Assumptions Editor (live coefficient editing), and Vietnam ML (GLM vs XGBoost comparison with SHAP).",
    endpoints: [
      "POST /api/v1/price",
      "GET /api/v1/price/assumptions",
      "POST /api/vietnam/price",
    ],
    tags: ["GLM", "4 sub-tabs"],
    status: "live",
  },
  {
    id: "admin",
    label: "Admin Console",
    category: "staff-page",
    x: 650, y: 90, w: 120, h: 40,
    description:
      "Full platform administration interface. Four panels: Model Management (version history + retraining), Rules Engine (thresholds + exclusions), System Health (live API/DB status + PSI drift), and Audit Log (filterable full decision history).",
    endpoints: [
      "GET /api/v1/models",
      "POST /api/v1/models/retrain",
      "GET /api/v1/rules",
      "PUT /api/v1/rules/{id}",
      "GET /api/v1/health",
    ],
    tags: ["admin only", "4 panels"],
    status: "live",
  },
  {
    id: "public-portal",
    label: "Public Portal",
    category: "portal",
    x: 804, y: 90, w: 120, h: 40,
    description:
      "Customer-facing SPA at /portal. No staff login required. Applicants can submit a quote, upload documents, track application status by case reference, and chat with the AI advisor. Completely separate from the staff portal — different auth flow, different UX.",
    endpoints: [
      "POST /api/v1/applications",
      "POST /api/v1/applications/{id}/documents",
      "GET /api/v1/applications/{ref}/status",
      "POST /api/v1/advisor/chat",
    ],
    tags: ["self-apply", "no auth", "AI advisor"],
    status: "live",
  },

  // ─── Life Insurance sub-tabs ──────────────────────────────────
  {
    id: "attribution",
    label: "Attribution",
    category: "workbench",
    x: 352, y: 210, w: 96, h: 34,
    description:
      "GLM premium attribution chart. Shows how each underwriting factor — age, BMI, smoking, pre-existing conditions — contributes to the final premium as a percentage. Recomputes on every input change.",
    endpoints: ["POST /api/v1/price"],
    tags: ["bar chart", "GLM"],
    status: "live",
    parent: "life-insurance",
  },
  {
    id: "sensitivity",
    label: "Sensitivity",
    category: "workbench",
    x: 456, y: 210, w: 96, h: 34,
    description:
      "One-at-a-time sensitivity sweep. Select any input variable, define a range, and see how premium responds as a line chart. Useful for explaining pricing decisions to applicants or regulators.",
    endpoints: ["POST /api/v1/price"],
    tags: ["line chart", "OAT sweep"],
    status: "live",
    parent: "life-insurance",
  },
  {
    id: "assumptions",
    label: "Assumptions",
    category: "workbench",
    x: 560, y: 210, w: 96, h: 34,
    description:
      "Live GLM assumptions editor. Modify base rates, age coefficients, BMI loadings, and rider multipliers directly in the UI without retraining. Changes persist in session state; premium recalculates immediately. Reset button restores production defaults.",
    endpoints: ["GET /api/v1/price/assumptions", "POST /api/v1/price"],
    tags: ["editable", "coefficients", "no retrain"],
    status: "live",
    parent: "life-insurance",
  },
  {
    id: "vietnam-ml",
    label: "Vietnam ML",
    category: "workbench",
    x: 664, y: 210, w: 96, h: 34,
    description:
      "Vietnam market ML pricing demo. Runs the same profile through two models simultaneously — a GLM (numpy-based OLS health score + Gamma mortality) and XGBoost — and compares results side by side. SHAP waterfall chart explains the top 3 drivers for each prediction.",
    endpoints: ["POST /api/vietnam/price"],
    tags: ["XGBoost", "GLM", "SHAP", "Vietnam"],
    status: "live",
    parent: "life-insurance",
  },

  // ─── Admin Console sub-panels ─────────────────────────────────
  {
    id: "model-mgmt",
    label: "Model Mgmt",
    category: "admin-tab",
    x: 522, y: 340, w: 88, h: 34,
    description:
      "Model version history table showing R², RMSE, training sample count, and created date for every model version. Trigger retraining from the UI with custom hyperparameters. Deployed model is marked as active. Includes the Vietnam retraining pipeline with version diffs.",
    endpoints: ["GET /api/v1/models", "POST /api/v1/models/retrain"],
    tags: ["versioning", "retrain", "metrics"],
    status: "live",
    parent: "admin",
  },
  {
    id: "rules-engine",
    label: "Rules Engine",
    category: "admin-tab",
    x: 618, y: 340, w: 88, h: 34,
    description:
      "Business rules management panel. Edit referral thresholds (max age, BMI limits, risk score cutoffs), coverage exclusions, and loading factors directly from the UI. Rule changes apply immediately to all new quotes — no redeploy needed.",
    endpoints: ["GET /api/v1/rules", "PUT /api/v1/rules/{id}"],
    tags: ["rules", "thresholds", "live edit"],
    status: "live",
    parent: "admin",
  },
  {
    id: "system-health",
    label: "System Health",
    category: "admin-tab",
    x: 714, y: 340, w: 88, h: 34,
    description:
      "Real-time platform health dashboard. Shows API response time, model inference latency, PostgreSQL connection status, and PSI drift metrics (GREEN < 0.10, AMBER 0.10–0.25, RED > 0.25). Alerts visible to all admin users.",
    endpoints: ["GET /api/v1/health", "GET /metrics"],
    tags: ["monitoring", "PSI drift", "latency"],
    status: "live",
    parent: "admin",
  },
  {
    id: "audit-log",
    label: "Audit Log",
    category: "admin-tab",
    x: 810, y: 340, w: 88, h: 34,
    description:
      "Complete, immutable audit trail of all underwriter decisions, model predictions, rule changes, and admin actions. Filterable by date range, user, case ID, and action type. Required for regulatory compliance.",
    endpoints: ["GET /api/v1/audit"],
    tags: ["compliance", "immutable", "filter"],
    status: "live",
    parent: "admin",
  },

  // ─── Backend ──────────────────────────────────────────────────
  {
    id: "backend-api",
    label: "FastAPI  ·  Render",
    category: "backend",
    x: 380, y: 415, w: 180, h: 44,
    description:
      "FastAPI REST API deployed on Render (Python 3.11). PostgreSQL via SQLAlchemy with connection pooling. JWT authentication (HS256, 8-hour expiry). Serves 30+ endpoints across pricing, applications, models, rules, audit, and advisor chat. ANTHROPIC_API_KEY and JWT_SECRET injected via Render env vars.",
    endpoints: [
      "POST /auth/login",
      "POST /api/v1/applications",
      "POST /api/v1/price",
      "POST /api/vietnam/price",
      "GET  /api/v1/health",
    ],
    tags: ["FastAPI", "PostgreSQL", "JWT", "Render"],
    status: "live",
  },
];

// Hierarchy edges (solid) — parent to children
export const PLATFORM_EDGES = [
  { from: "life-insurance", to: "attribution" },
  { from: "life-insurance", to: "sensitivity" },
  { from: "life-insurance", to: "assumptions" },
  { from: "life-insurance", to: "vietnam-ml" },
  { from: "admin", to: "model-mgmt" },
  { from: "admin", to: "rules-engine" },
  { from: "admin", to: "system-health" },
  { from: "admin", to: "audit-log" },
  // API connections (dashed)
  { from: "new-case",       to: "backend-api", dashed: true },
  { from: "case-pipeline",  to: "backend-api", dashed: true },
  { from: "life-insurance", to: "backend-api", dashed: true },
  { from: "admin",          to: "backend-api", dashed: true },
  { from: "public-portal",  to: "backend-api", dashed: true },
  { from: "vietnam-ml",     to: "backend-api", dashed: true },
  { from: "model-mgmt",     to: "backend-api", dashed: true },
];
