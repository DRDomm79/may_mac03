import { useState, useEffect, useRef } from "react";
import LifeInsurancePricer from "./LifeInsurancePricer";
import ApplicationWizard from "./ApplicationWizard";
import StatusTracker from "./StatusTracker";
import UnderwriterDashboard from "./UnderwriterDashboard";
import AdminConsole from "./AdminConsole";
import PlatformExplorer from "./PlatformExplorer";
import ActuarialAILab from "./ActuarialAILab";
import PublicPortal from "./portal/PublicPortal";
import { API_URL, clearAuth, getRole, getUser } from "./auth";

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim().toLowerCase(), password }),
      });
      if (!res.ok) { setError("Invalid credentials."); setPassword(""); return; }
      const { access_token, role, username: user } = await res.json();
      sessionStorage.setItem("dac_token", access_token);
      sessionStorage.setItem("dac_role",  role);
      sessionStorage.setItem("dac_user",  user);
      onLogin(role);
    } catch {
      setError("Unable to reach server. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: `linear-gradient(135deg, #0d2b7a 0%, #091d5e 100%)`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Sans', sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
      <div style={{ background: "#fff", borderRadius: 16, padding: "48px 40px", width: "100%", maxWidth: 400, boxShadow: "0 24px 64px rgba(0,0,0,0.3)" }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <img src="/DAC.png" alt="DAC" style={{ width: 80, height: 80, objectFit: "contain", marginBottom: 12 }} />
          <div style={{ fontSize: 22, fontWeight: 700, color: "#0d2b7a" }}>DAC <span style={{ color: "#f5a623" }}>HealthPrice</span></div>
          <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>Staff portal — internal access only</div>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "#374151", marginBottom: 6 }}>Username</label>
            <input
              type="text" value={username} onChange={e => setUsername(e.target.value)} autoComplete="username"
              style={{ width: "100%", padding: "10px 14px", borderRadius: 8, border: "1.5px solid #e5e7eb", fontSize: 14, fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              placeholder="Enter username"
            />
          </div>
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "#374151", marginBottom: 6 }}>Password</label>
            <input
              type="password" value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password"
              style={{ width: "100%", padding: "10px 14px", borderRadius: 8, border: "1.5px solid #e5e7eb", fontSize: 14, fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              placeholder="Enter password"
            />
          </div>
          {error && <div style={{ fontSize: 13, color: "#ef4444", marginBottom: 16, textAlign: "center" }}>{error}</div>}
          <button type="submit" disabled={loading} style={{ width: "100%", padding: "12px", borderRadius: 8, background: "#f5a623", color: "#0d2b7a", border: "none", fontSize: 15, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer", fontFamily: "inherit", opacity: loading ? 0.7 : 1 }}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}

const NAVY = "#0d2b7a";
const NAVY_D = "#091d5e";
const NAVY_L = "#1a4fba";
const GOLD = "#f5a623";
const GOLD_D = "#e67e00";
const WHITE = "#ffffff";
const GRAY = "#94a3b8";
const LTGRAY = "#f1f3f5";
const TXT = "#111827";
const TXT2 = "#4b5563";
const OK = "#10b981";
const TEAL = "#0d9488";

// ─── Animated counter ───────────────────────────────────────────────────────
function Counter({ end, suffix = "", duration = 2000 }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        let start = 0;
        const step = end / (duration / 16);
        const id = setInterval(() => {
          start += step;
          if (start >= end) { setVal(end); clearInterval(id); }
          else setVal(Math.floor(start));
        }, 16);
        obs.disconnect();
      }
    }, { threshold: 0.3 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [end, duration]);
  return <span ref={ref}>{val}{suffix}</span>;
}

// ─── Fade-in on scroll ──────────────────────────────────────────────────────
function FadeIn({ children, delay = 0, className = "" }) {
  const ref = useRef(null);
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVis(true); obs.disconnect(); }
    }, { threshold: 0.15 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} className={className} style={{
      opacity: vis ? 1 : 0, transform: vis ? "translateY(0)" : "translateY(32px)",
      transition: `opacity 0.7s ease ${delay}s, transform 0.7s ease ${delay}s`
    }}>{children}</div>
  );
}

// ─── PAGES ──────────────────────────────────────────────────────────────────
const ALL_PAGES = ["Home", "New Case", "Case Pipeline", "Life Insurance", "AI Lab", "Admin", "Platform"];
const ROLE_PAGES = {
  admin:       ALL_PAGES,
  underwriter: ALL_PAGES,
  analyst:     ALL_PAGES,
};

export default function App() {
  // Public portal — no staff auth required
  if (window.location.pathname.startsWith("/portal")) return <PublicPortal />;

  const [page, setPage] = useState("Home");
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const [authed, setAuthed] = useState(() => !!sessionStorage.getItem("dac_token"));
  const [role,   setRole]   = useState(() => getRole() || "");

  useEffect(() => {
    const h = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(30000) }).catch(() => {});
  }, []);

  if (!authed) return <LoginPage onLogin={(r) => { setAuthed(true); setRole(r); }} />;

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", color: TXT, background: WHITE, minHeight: "100vh" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet" />
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { overflow-x: hidden; }
        @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
        @keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:0.8} }
        @keyframes slideUp { from{opacity:0;transform:translateY(40px)} to{opacity:1;transform:translateY(0)} }
        .hero-title { animation: slideUp 0.8s ease forwards; }
        .hero-sub { animation: slideUp 0.8s ease 0.15s forwards; opacity:0; }
        .hero-bullets { animation: slideUp 0.8s ease 0.3s forwards; opacity:0; }
        .hero-cta { animation: slideUp 0.8s ease 0.45s forwards; opacity:0; }
        .card-hover:hover { transform: translateY(-6px); box-shadow: 0 20px 40px rgba(0,0,0,0.12); }
        .gold-btn { background:${GOLD}; color:${NAVY}; border:none; padding:14px 36px; border-radius:50px; font-weight:600; font-size:16px; cursor:pointer; transition:all 0.3s; font-family:inherit; }
        .gold-btn:hover { background:${GOLD_D}; transform:translateY(-2px); box-shadow:0 8px 24px rgba(245,197,99,0.3); }
        .outline-btn { background:transparent; color:${WHITE}; border:2px solid rgba(255,255,255,0.3); padding:14px 36px; border-radius:50px; font-weight:600; font-size:16px; cursor:pointer; transition:all 0.3s; font-family:inherit; }
        .outline-btn:hover { border-color:${GOLD}; color:${GOLD}; }
        a { text-decoration:none; color:inherit; }
        .nav-link { color:${GRAY}; font-size:15px; font-weight:500; cursor:pointer; transition:color 0.2s; padding:8px 0; }
        .nav-link:hover, .nav-link.active { color:${GOLD}; }
        .hamburger { display:none; flex-direction:column; justify-content:center; gap:5px; width:40px; height:40px; background:transparent; border:none; cursor:pointer; padding:4px; border-radius:8px; transition:background 0.2s; }
        .hamburger:hover { background:rgba(255,255,255,0.08); }
        .hamburger span { display:block; width:22px; height:2px; background:#ffffff; border-radius:2px; transition:all 0.25s ease; }
        .hamburger.open span:nth-child(1) { transform:translateY(7px) rotate(45deg); }
        .hamburger.open span:nth-child(2) { opacity:0; transform:scaleX(0); }
        .hamburger.open span:nth-child(3) { transform:translateY(-7px) rotate(-45deg); }
        .desktop-nav { display:flex; align-items:center; gap:32px; }
        .mobile-menu { position:fixed; top:72px; left:0; right:0; background:rgba(15,15,30,0.97); backdrop-filter:blur(16px); border-bottom:1px solid rgba(255,255,255,0.08); z-index:99; padding:16px 24px 24px; display:flex; flex-direction:column; gap:4px; }
        .mobile-menu .nav-link { font-size:17px; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.06); display:block; }
        .mobile-menu .gold-btn { margin-top:12px; width:100%; text-align:center; display:block; }
        .hero-grid { display:grid; grid-template-columns:1fr 1fr; gap:64px; align-items:center; position:relative; z-index:1; }
        .hero-inner { padding:80px 24px; }
        .grid-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:32px; text-align:center; }
        .grid-how { display:grid; grid-template-columns:repeat(4,1fr); gap:32px; }
        .grid-features { display:grid; grid-template-columns:1fr 1fr; gap:80px; align-items:center; }
        @media (max-width:768px) {
          .hamburger { display:flex; }
          .desktop-nav { display:none; }
          .hero-grid { grid-template-columns:1fr; gap:40px; }
          .hero-inner { padding:48px 20px 56px; }
          .grid-stats { grid-template-columns:repeat(2,1fr); gap:24px; }
          .grid-how { grid-template-columns:1fr; gap:24px; }
          .grid-features { grid-template-columns:1fr; gap:40px; }
        }
        @media (max-width:480px) {
          .hero-inner { padding:36px 16px 48px; }
        }
      `}</style>

      {/* ═══ NAVBAR ═══ */}
      <nav style={{ position:"fixed", top:0, left:0, right:0, zIndex:100,
        background: scrollY>60 ? "rgba(15,15,30,0.95)" : "transparent",
        backdropFilter: scrollY>60 ? "blur(16px)" : "none",
        transition:"background 0.3s, backdrop-filter 0.3s",
        borderBottom: scrollY>60 ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
        <div style={{ maxWidth:1200, margin:"0 auto", padding:"0 24px", display:"flex", alignItems:"center", justifyContent:"space-between", height:72 }}>
          {/* Logo */}
          <div style={{ display:"flex", alignItems:"center", gap:12, cursor:"pointer" }} onClick={() => setPage("Home")}>
            <img src="/DAC.png" alt="DAC" style={{ width:120, height:120, objectFit:"contain" }} />
            <span style={{ color:WHITE, fontSize:20, fontWeight:700, letterSpacing:-0.5 }}>DAC <span style={{ color:GOLD }}>HealthPrice</span></span>
          </div>
          {/* Desktop nav */}
          <div className="desktop-nav">
            <div style={{ display:"flex", gap:28 }}>
              {(ROLE_PAGES[role] || ALL_PAGES).map(p => <span key={p} className={`nav-link ${page===p?"active":""}`} onClick={() => { setPage(p); setMenuOpen(false); window.scrollTo(0,0); }}>{p}</span>)}
            </div>
            <button className="gold-btn" style={{ padding:"10px 28px", fontSize:14 }} onClick={() => { setPage("New Case"); window.scrollTo(0,0); }}>New Case</button>
            <button className="outline-btn" style={{ padding:"8px 20px", fontSize:13 }} onClick={() => { clearAuth(); setAuthed(false); setRole(""); }}>Sign out</button>
          </div>
          {/* Hamburger */}
          <button className={`hamburger ${menuOpen?"open":""}`} onClick={() => setMenuOpen(o => !o)} aria-label="Toggle menu">
            <span /><span /><span />
          </button>
        </div>
      </nav>
      {menuOpen && (
        <div className="mobile-menu">
          {(ROLE_PAGES[role] || ALL_PAGES).map(p => <span key={p} className={`nav-link ${page===p?"active":""}`} onClick={() => { setPage(p); setMenuOpen(false); window.scrollTo(0,0); }}>{p}</span>)}
          <button className="gold-btn" onClick={() => { setPage("New Case"); setMenuOpen(false); window.scrollTo(0,0); }}>New Case</button>
        </div>
      )}

      {page === "Home" && <HomePage onGetQuote={() => { setPage("New Case"); window.scrollTo(0, 0); }} />}
      {page === "New Case" && <ApplicationWizard />}
      {page === "Case Pipeline" && <UnderwriterDashboard />}
      {page === "Life Insurance" && <LifeInsurancePricer />}
      {page === "AI Lab" && <ActuarialAILab />}
      {page === "Admin" && <AdminConsole />}
      {page === "Platform" && <PlatformExplorer />}

      {/* ═══ FOOTER ═══ */}
      <footer style={{ background: NAVY_D, color: GRAY, padding: "64px 24px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 40, marginBottom: 48 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <img src="/DAC.png" alt="DAC" style={{ width: 96, height: 96, objectFit: "contain" }} />
                <span style={{ color: WHITE, fontSize: 18, fontWeight: 700 }}>DAC HealthPrice</span>
              </div>
              <p style={{ fontSize: 14, lineHeight: 1.7 }}>AI-powered health insurance pricing for Cambodia's emerging market.</p>
            </div>
            <div>
              <h4 style={{ color: WHITE, fontSize: 14, fontWeight: 600, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Coverage</h4>
              {["IPD Hospital", "OPD Outpatient", "Dental Care", "Maternity"].map(s => <p key={s} style={{ fontSize: 14, marginBottom: 10, cursor: "pointer" }}>{s}</p>)}
            </div>
            <div>
              <h4 style={{ color: WHITE, fontSize: 14, fontWeight: 600, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Company</h4>
              {[
                { label: "About DAC", go: "About" },
                { label: "Contact Us", go: "Contact" },
                { label: "New Case", go: "New Case" },
              ].map(s => <p key={s.label} onClick={() => { setPage(s.go); window.scrollTo(0,0); }} style={{ fontSize: 14, marginBottom: 10, cursor: "pointer" }}>{s.label}</p>)}
            </div>
            <div>
              <h4 style={{ color: WHITE, fontSize: 14, fontWeight: 600, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Contact</h4>
              <p style={{ fontSize: 14, marginBottom: 10 }}>radet@dactuaries.com</p>
              <p style={{ fontSize: 14, marginBottom: 10 }}>+855 85 508 860</p>
              <p style={{ fontSize: 14 }}>Phnom Penh, Cambodia</p>
            </div>
          </div>
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 24, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <p style={{ fontSize: 13 }}>© 2026 Decent Actuarial Consultants. All rights reserved.</p>
            <p style={{ fontSize: 13 }}>Powered by frequency-severity ML models</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOME PAGE
// ═══════════════════════════════════════════════════════════════════════════════
function HomePage({ onGetQuote }) {
  const [sampleQuote, setSampleQuote] = useState(null);

  useEffect(() => {
    const run = async () => {
      try {
        const sRes = await fetch(`${API_URL}/api/v2/session`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: "demo@dactuaries.com", browser_id: "homepage-demo" }),
          signal: AbortSignal.timeout(10000),
        });
        if (!sRes.ok) return;
        const { token } = await sRes.json();
        const pRes = await fetch(`${API_URL}/api/v2/price`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Session-Token": token },
          body: JSON.stringify({
            age: 35, gender: "Male", country: "cambodia", region: "Phnom Penh",
            smoking_status: "Never", exercise_frequency: "Moderate",
            occupation_type: "Office/Desk", preexist_conditions: ["None"],
            ipd_tier: "Silver", family_size: 1,
            include_opd: false, include_dental: false, include_maternity: false,
            browser_id: "homepage-demo", email: "demo@dactuaries.com",
          }),
          signal: AbortSignal.timeout(20000),
        });
        if (!pRes.ok) return;
        setSampleQuote(await pRes.json());
      } catch { /* keep static fallback */ }
    };
    run();
  }, []);

  const monthly = sampleQuote ? Math.round(sampleQuote.total_monthly_premium) : null;
  const annual  = sampleQuote ? sampleQuote.total_annual_premium : null;

  return (
    <>
      {/* ═══ HERO ═══ */}
      <section style={{
        background: `linear-gradient(135deg, ${NAVY} 0%, ${NAVY_D} 50%, #16213e 100%)`,
        minHeight: "100vh", display: "flex", alignItems: "center", position: "relative", overflow: "hidden", paddingTop: 72,
      }}>
        {/* Decorative orbs */}
        <div style={{ position: "absolute", top: -120, right: -80, width: 400, height: 400, borderRadius: "50%", background: `radial-gradient(circle, rgba(245,197,99,0.08) 0%, transparent 70%)`, animation: "float 6s ease-in-out infinite" }} />
        <div style={{ position: "absolute", bottom: -60, left: -100, width: 300, height: 300, borderRadius: "50%", background: `radial-gradient(circle, rgba(245,197,99,0.05) 0%, transparent 70%)`, animation: "float 8s ease-in-out infinite 1s" }} />
        {/* Gold line accent */}
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)` }} />

        <div className="hero-grid hero-inner" style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div>
            <div className="hero-title">
              <span style={{ color: GOLD, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase", marginBottom: 16, display: "block" }}>AI-Powered Insurance Pricing</span>
              <h1 style={{ fontFamily: "'Playfair Display', serif", color: WHITE, fontSize: "clamp(36px, 5vw, 56px)", lineHeight: 1.1, fontWeight: 700, marginBottom: 24 }}>
                Health insurance<br />priced by <span style={{ color: GOLD }}>machine learning</span>
              </h1>
            </div>
            <p className="hero-sub" style={{ color: GRAY, fontSize: 18, lineHeight: 1.7, marginBottom: 32, maxWidth: 480 }}>
              Personalized hospital reimbursement premiums for Cambodia. Get a quote in minutes using our frequency-severity ML models.
            </p>
            <div className="hero-bullets" style={{ display: "flex", flexDirection: "column", gap: 14, marginBottom: 40 }}>
              {[
                "IPD coverage with 4 tier options",
                "Optional OPD, Dental & Maternity riders",
                "AI advisor recommends your ideal plan",
              ].map((t, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ width: 22, height: 22, borderRadius: "50%", background: GOLD, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke={NAVY} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </div>
                  <span style={{ color: "rgba(255,255,255,0.85)", fontSize: 16 }}>{t}</span>
                </div>
              ))}
            </div>
            <div className="hero-cta" style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <button className="gold-btn" style={{ fontSize: 17, padding: "16px 40px" }} onClick={onGetQuote}>New Case</button>
              <button className="outline-btn" onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
            </div>
          </div>

          {/* Hero right — Stats cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20, alignItems: "center" }}>
            <div style={{ background: "rgba(255,255,255,0.06)", borderRadius: 20, border: "1px solid rgba(255,255,255,0.08)", padding: 32, width: "100%", maxWidth: 420, backdropFilter: "blur(12px)" }}>
              <p style={{ color: GRAY, fontSize: 13, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
                Sample quote — 35M, Non-smoker{sampleQuote ? " · Live" : ""}
              </p>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 16 }}>
                <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 48, fontWeight: 700, color: GOLD }}>
                  {monthly !== null ? `$${monthly}` : "$—"}
                </span>
                <span style={{ color: GRAY, fontSize: 16 }}>/month</span>
                {annual && <span style={{ color: GRAY, fontSize: 12, opacity: 0.7 }}>(${annual.toLocaleString()}/yr)</span>}
              </div>
              <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                {["Bronze", "Silver", "Gold", "Platinum"].map((t, i) => (
                  <div key={t} style={{ flex: 1, padding: "8px 0", borderRadius: 8, background: i === 1 ? GOLD : "rgba(255,255,255,0.06)", color: i === 1 ? NAVY : GRAY, textAlign: "center", fontSize: 12, fontWeight: 600 }}>{t}</div>
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 0", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ color: GRAY, fontSize: 13 }}>Annual limit</span>
                <span style={{ color: WHITE, fontSize: 13, fontWeight: 600 }}>$40,000</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 0", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ color: GRAY, fontSize: 13 }}>Deductible</span>
                <span style={{ color: WHITE, fontSize: 13, fontWeight: 600 }}>$250</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 0", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ color: GRAY, fontSize: 13 }}>AI recommendation</span>
                <span style={{ color: OK, fontSize: 13, fontWeight: 600 }}>Best fit ✓</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ TRUST BADGES ═══ */}
      <section style={{ background: WHITE, padding: "48px 24px", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", textAlign: "center" }}>
          <p style={{ color: TXT2, fontSize: 14, marginBottom: 24, textTransform: "uppercase", letterSpacing: 1 }}>Backed by actuarial expertise</p>
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 48, flexWrap: "wrap", opacity: 0.7 }}>
            {["Decent Actuarial Consultants", "RNA Analytics", "R3S Modeler", "Society of Actuaries"].map(t => (
              <span key={t} style={{ fontSize: 16, fontWeight: 700, color: TXT, letterSpacing: -0.5 }}>{t}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ STATS ═══ */}
      <section style={{ background: NAVY, padding: "64px 24px" }}>
        <div className="grid-stats" style={{ maxWidth: 1000, margin: "0 auto" }}>
          {[
            { val: 17, suf: "M", label: "Population" },
            { val: 40, suf: "+", label: "Insurance companies" },
            { val: 5, suf: "%", label: "Insurance penetration" },
            { val: 4, suf: "", label: "ML models compared" },
          ].map((s, i) => (
            <FadeIn key={i} delay={i * 0.1}>
              <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 42, fontWeight: 700, color: GOLD, marginBottom: 8 }}>
                <Counter end={s.val} suffix={s.suf} />
              </div>
              <p style={{ color: GRAY, fontSize: 14 }}>{s.label}</p>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* ═══ HOW IT WORKS ═══ */}
      <section id="how-it-works" style={{ background: WHITE, padding: "96px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <FadeIn>
            <div style={{ textAlign: "center", marginBottom: 64 }}>
              <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Simple process</span>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, fontWeight: 700, color: TXT, marginTop: 12 }}>Insurance made easy</h2>
              <p style={{ color: TXT2, fontSize: 18, marginTop: 12, maxWidth: 560, margin: "12px auto 0" }}>Health insurance pricing doesn't have to be complicated. Our ML models keep things simple.</p>
            </div>
          </FadeIn>
          <div className="grid-how">
            {[
              { step: "01", title: "Tell us about you", desc: "Enter your age, health history, lifestyle, and region across three short screens. The guided wizard takes under 2 minutes.", icon: "👤" },
              { step: "02", title: "Choose your cover", desc: "Pick your IPD tier — Bronze, Silver, Gold, or Platinum — and add optional OPD, Dental, or Maternity riders to suit your needs.", icon: "🛡️" },
              { step: "03", title: "Underwriting review", desc: "Our rule engine instantly assesses your risk profile. Pre-existing conditions and lifestyle factors are flagged transparently before you commit.", icon: "🔍" },
              { step: "04", title: "Your personalised quote", desc: "Receive a live ML-priced premium breakdown — frequency, severity, and loading shown clearly. Ask the AI chatbot anything about your plan.", icon: "💰" },
            ].map((s, i) => (
              <FadeIn key={i} delay={i * 0.15}>
                <div className="card-hover" style={{ background: WHITE, borderRadius: 16, padding: 36, border: "1px solid #e5e7eb", transition: "all 0.3s", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: GOLD }} />
                  <div style={{ fontSize: 40, marginBottom: 16 }}>{s.icon}</div>
                  <span style={{ color: GOLD_D, fontSize: 13, fontWeight: 700, letterSpacing: 1 }}>STEP {s.step}</span>
                  <h3 style={{ fontSize: 22, fontWeight: 700, marginTop: 8, marginBottom: 12, color: TXT }}>{s.title}</h3>
                  <p style={{ color: TXT2, fontSize: 15, lineHeight: 1.7 }}>{s.desc}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ COVERAGE CARDS — dark section ═══ */}
      <section style={{ background: NAVY, padding: "96px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <FadeIn>
            <div style={{ textAlign: "center", marginBottom: 64 }}>
              <span style={{ color: GOLD, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Coverage options</span>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, fontWeight: 700, color: WHITE, marginTop: 12 }}>Choose your protection</h2>
            </div>
          </FadeIn>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 24 }}>
            {[
              { title: "IPD Hospital", desc: "Inpatient department coverage for hospital stays, surgeries, and intensive care. Our core product with 4 tier levels.", tag: "Core", color: GOLD },
              { title: "OPD Outpatient", desc: "Doctor visits, consultations, diagnostic tests, and outpatient procedures. Add to any IPD plan.", tag: "Rider", color: TEAL },
              { title: "Dental Care", desc: "Routine dental checkups, cleanings, fillings, and emergency dental procedures.", tag: "Rider", color: "#8b5cf6" },
              { title: "Maternity", desc: "Prenatal care, delivery, postnatal care, and newborn coverage for the first 30 days.", tag: "Rider", color: "#ec4899" },
            ].map((c, i) => (
              <FadeIn key={i} delay={i * 0.1}>
                <div className="card-hover" style={{ background: WHITE, borderRadius: 16, padding: 32, transition: "all 0.3s", cursor: "pointer" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 16 }}>
                    <div style={{ width: 48, height: 48, borderRadius: 12, background: `${c.color}15`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <div style={{ width: 24, height: 24, borderRadius: "50%", background: c.color }} />
                    </div>
                    <span style={{ background: `${c.color}15`, color: c.color, padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600 }}>{c.tag}</span>
                  </div>
                  <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 10 }}>{c.title}</h3>
                  <p style={{ color: TXT2, fontSize: 14, lineHeight: 1.7, marginBottom: 20 }}>{c.desc}</p>
                  <button className="gold-btn" style={{ padding: "10px 24px", fontSize: 13, width: "100%" }} onClick={onGetQuote}>Get a quote</button>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ FEATURES — split layout ═══ */}
      <section style={{ background: WHITE, padding: "96px 24px" }}>
        <div className="grid-features" style={{ maxWidth: 1200, margin: "0 auto" }}>
          <FadeIn>
            <div style={{ background: NAVY, borderRadius: 24, padding: 40, position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: -40, right: -40, width: 160, height: 160, borderRadius: "50%", background: `rgba(245,197,99,0.08)` }} />
              <p style={{ color: GOLD, fontSize: 13, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase", marginBottom: 20 }}>AI-Powered Advisor</p>
              <div style={{ background: "rgba(255,255,255,0.06)", borderRadius: 12, padding: 16, marginBottom: 12 }}>
                <p style={{ color: GRAY, fontSize: 13, marginBottom: 8 }}>You asked:</p>
                <p style={{ color: WHITE, fontSize: 14 }}>"Why is my premium this amount?"</p>
              </div>
              <div style={{ background: `rgba(245,197,99,0.08)`, borderRadius: 12, padding: 16, borderLeft: `3px solid ${GOLD}` }}>
                <p style={{ color: GOLD, fontSize: 13, marginBottom: 8 }}>AI Advisor:</p>
                <p style={{ color: "rgba(255,255,255,0.8)", fontSize: 14, lineHeight: 1.6 }}>"Your premium of $301/year reflects a claim frequency of 0.16/year and average severity of $3,521. Smoking status is the largest contributor — quitting could save you $213/year."</p>
              </div>
            </div>
          </FadeIn>
          <FadeIn delay={0.2}>
            <div>
              <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Smart features</span>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, fontWeight: 700, marginTop: 12, marginBottom: 24 }}>Insurance that understands you</h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {[
                  { title: "Plan advisor", desc: "AI recommends the optimal tier based on your unique risk profile." },
                  { title: "Risk explainer", desc: "See exactly which factors drive your premium and by how much." },
                  { title: "Cost optimizer", desc: "Get personalized suggestions to reduce your premium — like exercising more or managing conditions." },
                  { title: "Transparent breakdown", desc: "Full actuarial breakdown: frequency, severity, loading, tier factor, and deductible credit." },
                ].map((f, i) => (
                  <div key={i} style={{ display: "flex", gap: 16 }}>
                    <div style={{ width: 32, height: 32, borderRadius: 8, background: `${GOLD}15`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2 }}>
                      <svg width="16" height="16" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke={GOLD_D} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </div>
                    <div>
                      <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>{f.title}</h4>
                      <p style={{ color: TXT2, fontSize: 14, lineHeight: 1.6 }}>{f.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ═══ CTA BANNER ═══ */}
      <section style={{ background: `linear-gradient(135deg, ${NAVY} 0%, #16213e 100%)`, padding: "80px 24px", textAlign: "center", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)` }} />
        <FadeIn>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, color: WHITE, fontWeight: 700, marginBottom: 16 }}>Ready to see your premium?</h2>
          <p style={{ color: GRAY, fontSize: 18, marginBottom: 32, maxWidth: 500, margin: "0 auto 32px" }}>Get a personalized quote in under 2 minutes. No commitment required.</p>
          <button className="gold-btn" style={{ fontSize: 18, padding: "18px 48px" }} onClick={onGetQuote}>Open New Case</button>
        </FadeIn>
      </section>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ABOUT PAGE
// ═══════════════════════════════════════════════════════════════════════════════
function AboutPage() {
  return (
    <section style={{ paddingTop: 120, paddingBottom: 80 }}>
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px" }}>
        <FadeIn>
          <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>About us</span>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, fontWeight: 700, marginTop: 12, marginBottom: 32 }}>Decent Actuarial Consultants</h2>
        </FadeIn>
        <FadeIn delay={0.1}>
          <p style={{ color: TXT2, fontSize: 17, lineHeight: 1.8, marginBottom: 24 }}>
            DAC is an independent actuarial consulting firm founded by Dr. Chris Chih-Ching Chan (PhD, FSA, FAIRC, FCAA) with offices in Taipei (2013), Phnom Penh (2015), and Hanoi (2017). We support insurance companies throughout their entire lifecycle — from pre-establishment market research through mature-stage risk management to M&A advisory.
          </p>
        </FadeIn>
        <FadeIn delay={0.2}>
          <p style={{ color: TXT2, fontSize: 17, lineHeight: 1.8, marginBottom: 40 }}>
            As the Head of Business in Taiwan and Cambodia for RNA Analytics, DAC brings global actuarial modeling standards (R3S Modeler, used in 50+ countries) to Southeast Asia's emerging insurance markets. Cambodia — with 17 million people, an average age of 27, and insurance penetration of only ~1% — represents enormous growth potential.
          </p>
        </FadeIn>
        <FadeIn delay={0.3}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24 }}>
            {[
              { city: "Taipei", year: "2013", role: "Headquarters" },
              { city: "Phnom Penh", year: "2015", role: "Cambodia office" },
              { city: "Hanoi", year: "2017", role: "Vietnam office" },
            ].map((o, i) => (
              <div key={i} style={{ background: NAVY, borderRadius: 16, padding: 28, textAlign: "center" }}>
                <p style={{ color: GOLD, fontFamily: "'Playfair Display', serif", fontSize: 32, fontWeight: 700 }}>{o.year}</p>
                <p style={{ color: WHITE, fontSize: 18, fontWeight: 600, marginTop: 8 }}>{o.city}</p>
                <p style={{ color: GRAY, fontSize: 14, marginTop: 4 }}>{o.role}</p>
              </div>
            ))}
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC PORTAL PAGE (coming soon)
// ═══════════════════════════════════════════════════════════════════════════════
function PublicPortalPage() {
  const features = [
    {
      icon: "📋",
      title: "Self-Apply",
      desc: "Customers fill the 5-step application wizard directly — no staff intermediary needed.",
      status: "Planned · Week 6",
      color: TEAL,
    },
    {
      icon: "📡",
      title: "Live Case Tracking",
      desc: "Customers see their own case status, documents required, and decision timeline in real time.",
      status: "Planned · Week 6",
      color: "#8b5cf6",
    },
    {
      icon: "🔐",
      title: "Secure Auth (OTP / JWT)",
      desc: "Phone-number OTP login so customers authenticate without creating a staff account.",
      status: "Planned · Week 7",
      color: GOLD_D,
    },
    {
      icon: "📄",
      title: "Document Upload",
      desc: "Customers upload ID, medical reports, and consent forms directly to their case file.",
      status: "Planned · Week 6",
      color: OK,
    },
    {
      icon: "💬",
      title: "AI Quote Advisor",
      desc: "Embedded chatbot explains premium breakdown, suggests plan optimizations, and answers coverage questions.",
      status: "Planned · Week 7",
      color: "#ec4899",
    },
    {
      icon: "🔔",
      title: "Push Notifications",
      desc: "Email / SMS alerts when case status changes — no polling required from the customer.",
      status: "Planned · Week 8",
      color: NAVY_L,
    },
  ];

  const techStack = [
    { label: "Frontend", value: "React SPA (separate Vercel deployment)" },
    { label: "Auth", value: "Phone OTP → JWT (Twilio / Firebase)" },
    { label: "API", value: "Shared FastAPI backend (dac-healthprice-api.onrender.com)" },
    { label: "Storage", value: "Supabase (PostgreSQL + object storage)" },
    { label: "Notifications", value: "Resend (email) + Twilio (SMS)" },
    { label: "Deploy target", value: "portal.dactuaries.com (Week 8)" },
  ];

  return (
    <section style={{ paddingTop: 120, paddingBottom: 80, background: WHITE, minHeight: "100vh" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px" }}>

        {/* Header */}
        <FadeIn>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
            <span style={{ background: `${GOLD}20`, color: GOLD_D, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", padding: "4px 12px", borderRadius: 20 }}>
              Coming Soon · Week 6–8
            </span>
          </div>
          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 44, fontWeight: 700, color: TXT, marginBottom: 16 }}>
            Customer Public Portal
          </h1>
          <p style={{ color: TXT2, fontSize: 18, lineHeight: 1.7, maxWidth: 680, marginBottom: 48 }}>
            A separate customer-facing SPA so applicants can self-apply, upload documents, and track their own case — without staff involvement at every step. Shares the same FastAPI backend as this internal tool.
          </p>
        </FadeIn>

        {/* Architecture split callout */}
        <FadeIn delay={0.1}>
          <div style={{ background: NAVY, borderRadius: 20, padding: "32px 40px", marginBottom: 64, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
            <div>
              <p style={{ color: GOLD, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", marginBottom: 12 }}>This app — Internal Tool</p>
              <p style={{ color: WHITE, fontSize: 20, fontWeight: 700, marginBottom: 8 }}>DAC HealthPrice (Staff)</p>
              <p style={{ color: GRAY, fontSize: 14, lineHeight: 1.6 }}>New Case · Case Pipeline · GLM Workbench · UW Review · Admin Console</p>
              <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: OK }} />
                <span style={{ color: OK, fontSize: 13, fontWeight: 600 }}>Live — dac-healthprice-frontend.vercel.app</span>
              </div>
            </div>
            <div style={{ borderLeft: `1px solid rgba(255,255,255,0.1)`, paddingLeft: 32 }}>
              <p style={{ color: GOLD, fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", marginBottom: 12 }}>New app — Public Portal</p>
              <p style={{ color: WHITE, fontSize: 20, fontWeight: 700, marginBottom: 8 }}>DAC HealthPrice (Customer)</p>
              <p style={{ color: GRAY, fontSize: 14, lineHeight: 1.6 }}>Self-Apply · Track My Case · Upload Documents · AI Advisor · OTP Login</p>
              <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: GOLD, animation: "pulse 2s infinite" }} />
                <span style={{ color: GRAY, fontSize: 13, fontWeight: 600 }}>Building · portal.dactuaries.com</span>
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Feature grid */}
        <FadeIn delay={0.15}>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginBottom: 32 }}>Planned Features</h2>
        </FadeIn>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 24, marginBottom: 64 }}>
          {features.map((f, i) => (
            <FadeIn key={i} delay={0.1 + i * 0.08}>
              <div className="card-hover" style={{ background: WHITE, border: "1px solid #e5e7eb", borderRadius: 16, padding: 28, transition: "all 0.3s", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: f.color }} />
                <div style={{ fontSize: 32, marginBottom: 12 }}>{f.icon}</div>
                <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, color: TXT }}>{f.title}</h3>
                <p style={{ color: TXT2, fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>{f.desc}</p>
                <span style={{ background: `${f.color}15`, color: f.color, fontSize: 11, fontWeight: 700, letterSpacing: 0.5, padding: "4px 10px", borderRadius: 20 }}>{f.status}</span>
              </div>
            </FadeIn>
          ))}
        </div>

        {/* Tech stack */}
        <FadeIn delay={0.2}>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginBottom: 24 }}>Tech Stack</h2>
          <div style={{ border: "1px solid #e5e7eb", borderRadius: 16, overflow: "hidden" }}>
            {techStack.map((r, i) => (
              <div key={i} style={{ display: "flex", gap: 24, padding: "16px 28px", background: i % 2 === 0 ? WHITE : LTGRAY, borderBottom: i < techStack.length - 1 ? "1px solid #e5e7eb" : "none" }}>
                <span style={{ minWidth: 120, fontSize: 14, fontWeight: 600, color: NAVY }}>{r.label}</span>
                <span style={{ fontSize: 14, color: TXT2 }}>{r.value}</span>
              </div>
            ))}
          </div>
        </FadeIn>

        {/* Week timeline */}
        <FadeIn delay={0.25}>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, margin: "48px 0 24px" }}>Build Timeline</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {[
              { week: "Week 6", focus: "Public portal scaffold — self-apply wizard + document upload", done: false },
              { week: "Week 7", focus: "OTP auth (JWT) + case tracking + AI advisor chatbot", done: false },
              { week: "Week 8", focus: "Email/SMS notifications + load test + custom domain launch", done: false },
            ].map((w, i) => (
              <div key={i} style={{ display: "flex", gap: 20, alignItems: "flex-start", padding: "20px 0", borderBottom: "1px solid #f3f4f6" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                  <div style={{ width: 36, height: 36, borderRadius: "50%", background: w.done ? OK : LTGRAY, border: `2px solid ${w.done ? OK : "#d1d5db"}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    {w.done
                      ? <svg width="16" height="16" viewBox="0 0 12 12"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      : <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#d1d5db" }} />}
                  </div>
                  {i < 2 && <div style={{ width: 2, height: 40, background: "#e5e7eb", marginTop: 4 }} />}
                </div>
                <div style={{ paddingTop: 6 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: GOLD_D, textTransform: "uppercase", letterSpacing: 0.5 }}>{w.week}</span>
                  <p style={{ color: TXT, fontSize: 15, marginTop: 4, lineHeight: 1.6 }}>{w.focus}</p>
                </div>
              </div>
            ))}
          </div>
        </FadeIn>

      </div>
    </section>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTACT PAGE
// ═══════════════════════════════════════════════════════════════════════════════
function ContactPage() {
  return (
    <section style={{ paddingTop: 120, paddingBottom: 80 }}>
      <div style={{ maxWidth: 600, margin: "0 auto", padding: "0 24px" }}>
        <FadeIn>
          <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Get in touch</span>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, fontWeight: 700, marginTop: 12, marginBottom: 32 }}>Contact us</h2>
        </FadeIn>
        <FadeIn delay={0.1}>
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <input placeholder="Full name" style={{ padding: "14px 18px", borderRadius: 10, border: "1px solid #d1d5db", fontSize: 15, fontFamily: "inherit", outline: "none" }} />
            <input placeholder="Email address" type="email" style={{ padding: "14px 18px", borderRadius: 10, border: "1px solid #d1d5db", fontSize: 15, fontFamily: "inherit", outline: "none" }} />
            <input placeholder="Phone number" style={{ padding: "14px 18px", borderRadius: 10, border: "1px solid #d1d5db", fontSize: 15, fontFamily: "inherit", outline: "none" }} />
            <textarea placeholder="Your message" rows={5} style={{ padding: "14px 18px", borderRadius: 10, border: "1px solid #d1d5db", fontSize: 15, fontFamily: "inherit", resize: "vertical", outline: "none" }} />
            <button className="gold-btn" style={{ width: "100%" }}>Send message</button>
          </div>
        </FadeIn>
        <FadeIn delay={0.2}>
          <div style={{ marginTop: 40, padding: 24, background: LTGRAY, borderRadius: 12 }}>
            <p style={{ fontWeight: 600, marginBottom: 8 }}>DAC Phnom Penh Office</p>
            <p style={{ color: TXT2, fontSize: 14, lineHeight: 1.8 }}>Email: radet@dactuaries.com<br />Phone: +855 85 508 860 <br />Phnom Penh, Cambodia</p>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

