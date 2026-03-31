import { useState, useEffect, useRef } from "react";
import PricingWizard from "./PricingWizard";


const NAVY = "#1a1a2e";
const NAVY_D = "#0f0f1e";
const NAVY_L = "#2d2d44";
const GOLD = "#f5c563";
const GOLD_D = "#d4a843";
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
const PAGES = ["Home", "Pricing", "About", "Contact"];

export default function App() {
  const [page, setPage] = useState("Home");
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const h = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

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
        .grid-how { display:grid; grid-template-columns:repeat(3,1fr); gap:40px; }
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
            <div style={{ width:40, height:40, background:GOLD, borderRadius:10, display:"flex", alignItems:"center", justifyContent:"center", fontWeight:700, color:NAVY, fontSize:16 }}>D</div>
            <span style={{ color:WHITE, fontSize:20, fontWeight:700, letterSpacing:-0.5 }}>DAC <span style={{ color:GOLD }}>HealthPrice</span></span>
          </div>
          {/* Desktop nav */}
          <div className="desktop-nav">
            <div style={{ display:"flex", gap:28 }}>
              {PAGES.map(p => <span key={p} className={`nav-link ${page===p?"active":""}`} onClick={() => { setPage(p); setMenuOpen(false); window.scrollTo(0,0); }}>{p}</span>)}
            </div>
            <button className="gold-btn" style={{ padding:"10px 28px", fontSize:14 }} onClick={() => { setPage("Pricing"); window.scrollTo(0,0); }}>Get a quote</button>
          </div>
          {/* Hamburger */}
          <button className={`hamburger ${menuOpen?"open":""}`} onClick={() => setMenuOpen(o => !o)} aria-label="Toggle menu">
            <span /><span /><span />
          </button>
        </div>
      </nav>
      {menuOpen && (
        <div className="mobile-menu">
          {PAGES.map(p => <span key={p} className={`nav-link ${page===p?"active":""}`} onClick={() => { setPage(p); setMenuOpen(false); window.scrollTo(0,0); }}>{p}</span>)}
          <button className="gold-btn" onClick={() => { setPage("Pricing"); setMenuOpen(false); window.scrollTo(0,0); }}>Get a quote</button>
        </div>
      )}

      {page === "Home" && <HomePage onGetQuote={() => { setPage("Pricing"); window.scrollTo(0, 0); }} />}
      {page === "Pricing" && <PricingPage />}
      {page === "About" && <AboutPage />}
      {page === "Contact" && <ContactPage />}
      {page === "Admin" && <AdminPage />}

      {/* ═══ FOOTER ═══ */}
      <footer style={{ background: NAVY_D, color: GRAY, padding: "64px 24px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 40, marginBottom: 48 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <div style={{ width: 36, height: 36, background: GOLD, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, color: NAVY, fontSize: 14 }}>D</div>
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
                { label: "Get a Quote", go: "Pricing" },
              ].map(s => <p key={s.label} onClick={() => { setPage(s.go); window.scrollTo(0,0); }} style={{ fontSize: 14, marginBottom: 10, cursor: "pointer" }}>{s.label}</p>)}
              <p onClick={() => { setPage("Admin"); window.scrollTo(0,0); }} style={{ fontSize: 12, marginTop: 16, cursor: "pointer", opacity: 0.4 }}>Admin portal</p>
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
              <button className="gold-btn" style={{ fontSize: 17, padding: "16px 40px" }} onClick={onGetQuote}>Get a quote</button>
              <button className="outline-btn" onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
            </div>
          </div>

          {/* Hero right — Stats cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20, alignItems: "center" }}>
            <div style={{ background: "rgba(255,255,255,0.06)", borderRadius: 20, border: "1px solid rgba(255,255,255,0.08)", padding: 32, width: "100%", maxWidth: 420, backdropFilter: "blur(12px)" }}>
              <p style={{ color: GRAY, fontSize: 13, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Sample quote — 35M, Non-smoker</p>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 16 }}>
                <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 48, fontWeight: 700, color: GOLD }}>$25</span>
                <span style={{ color: GRAY, fontSize: 16 }}>/month</span>
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
              { step: "01", title: "Tell us about you", desc: "Enter your age, health habits, occupation, and region in our guided step-by-step wizard. Takes under 2 minutes.", icon: "👤" },
              { step: "02", title: "AI picks your plan", desc: "Our frequency-severity models calculate your risk profile. The AI recommends your optimal tier — Bronze, Silver, Gold, or Platinum.", icon: "🤖" },
              { step: "03", title: "See your quote", desc: "Get a transparent premium breakdown showing claim frequency, severity, loading, and deductible credit. Ask the AI chatbot for advice.", icon: "💰" },
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
          <button className="gold-btn" style={{ fontSize: 18, padding: "18px 48px" }} onClick={onGetQuote}>Get your free quote</button>
        </FadeIn>
      </section>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRICING PAGE (placeholder — your existing wizard goes here)
// ═══════════════════════════════════════════════════════════════════════════════
function PricingPage() {  
    return (
      <section style={{ paddingTop: 80 }}>
        <PricingWizard />
      </section>
    );
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <section style={{ paddingTop: 100, paddingBottom: 80, minHeight: "100vh", background: "#f8f9fa" }}>
      <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
        <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Pricing Engine</span>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, fontWeight: 700, marginTop: 12, marginBottom: 16, color: TXT }}>Get your personalized quote</h2>
        <p style={{ color: TXT2, fontSize: 16, marginBottom: 40 }}>Your 4-step wizard (Profile → Health → Plan → Quote) will render here.</p>
        <div style={{ background: WHITE, borderRadius: 16, padding: 40, border: "1px solid #e5e7eb", textAlign: "left" }}>
          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            {["Profile", "Health", "Plan", "Quote"].map((s, i) => (
              <div key={s} style={{ flex: 1, textAlign: "center", padding: "10px 0", borderRadius: 8, background: i === 0 ? NAVY : "#f1f3f5", color: i === 0 ? GOLD : TXT2, fontSize: 13, fontWeight: 600 }}>{s}</div>
            ))}
          </div>
          <p style={{ color: TXT2, fontSize: 15, lineHeight: 1.8, marginBottom: 20 }}>
            To integrate your existing pricing wizard into this landing page:
          </p>
          <div style={{ background: "#f8f9fa", borderRadius: 8, padding: 16, fontFamily: "monospace", fontSize: 13, lineHeight: 1.8, color: TXT }}>
            <div>1. Rename <b>dac-healthprice-v2.jsx</b> → <b>PricingWizard.jsx</b></div>
            <div>2. Change its export: <code>export default function PricingWizard()</code></div>
            <div>3. Remove the navbar from PricingWizard (it's in the landing page now)</div>
            <div>4. Import in this file: <code>import PricingWizard from "./PricingWizard"</code></div>
            <div>5. Replace PricingPage body with: <code>{"<PricingWizard />"}</code></div>
          </div>
          <div style={{ marginTop: 20, padding: 16, background: `rgba(245,197,99,0.1)`, borderRadius: 12, borderLeft: `3px solid ${GOLD}` }}>
            <p style={{ fontSize: 14, color: TXT, margin: 0 }}>Both files are in your outputs. The wizard connects to your Render backend at <code>https://dac-healthprice-api.onrender.com</code></p>
          </div>
        </div>
      </div>
    </section>
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

// ═══════════════════════════════════════════════════════════════════════════════
// ADMIN PAGE (password-gated, accessed via footer link)
// ═══════════════════════════════════════════════════════════════════════════════
function AdminPage() {
  const [key, setKey] = useState("");
  const [authed, setAuthed] = useState(false);
  const [status, setStatus] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState(null);
  const [covType, setCovType] = useState("ipd");
  const [autoRetrain, setAutoRetrain] = useState(false);
 
  const API = "https://dac-healthprice-api.onrender.com";
 
  const checkHealth = async () => {
    try {
      const r = await fetch(`${API}/health`);
      setStatus(await r.json());
    } catch { setStatus({ error: "Cannot reach API" }); }
  };
 
  const handleUpload = async () => {
    if (!file || !key) return;
    setUploading(true); setUploadResult(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("coverage_type", covType);
    fd.append("auto_retrain", autoRetrain);
    try {
      const r = await fetch(`${API}/api/v2/admin/upload-dataset`, {
        method: "POST", headers: { "X-API-Key": key }, body: fd
      });
      setUploadResult(await r.json());
    } catch (e) { setUploadResult({ status: "error", detail: e.message }); }
    setUploading(false);
  };
 
  if (!authed) {
    return (
      <section style={{ paddingTop: 120, paddingBottom: 80, minHeight: "80vh" }}>
        <div style={{ maxWidth: 400, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
          <div style={{ width: 56, height: 56, borderRadius: 16, background: NAVY, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 20px" }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2C9.24 2 7 4.24 7 7v3H5v12h14V10h-2V7c0-2.76-2.24-5-5-5zm0 2c1.66 0 3 1.34 3 3v3H9V7c0-1.66 1.34-3 3-3z" fill={GOLD}/></svg>
          </div>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Admin portal</h2>
          <p style={{ color: TXT2, fontSize: 14, marginBottom: 24 }}>Enter your API key to access the dashboard</p>
          <input type="password" placeholder="Enter admin API key" value={key}
            onChange={e => setKey(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && key) setAuthed(true); }}
            style={{ width: "100%", padding: "12px 16px", borderRadius: 10, border: "1px solid #d1d5db", fontSize: 15, fontFamily: "inherit", marginBottom: 12, outline: "none" }} />
          <button className="gold-btn" style={{ width: "100%" }} onClick={() => { if (key) setAuthed(true); }}>Authenticate</button>
        </div>
      </section>
    );
  }
 
  return (
    <section style={{ paddingTop: 100, paddingBottom: 80 }}>
      <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
        <FadeIn>
          <span style={{ color: GOLD_D, fontSize: 14, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>Admin dashboard</span>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, fontWeight: 700, marginTop: 12, marginBottom: 32 }}>System management</h2>
        </FadeIn>
 
        <FadeIn delay={0.1}>
          <div style={{ background: WHITE, borderRadius: 16, padding: 28, border: "1px solid #e5e7eb", marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: 18, fontWeight: 600 }}>System status</h3>
              <button className="gold-btn" style={{ padding: "8px 20px", fontSize: 13 }} onClick={checkHealth}>Refresh</button>
            </div>
            {status ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
                {[
                  { label: "Status", value: status.status || "Error", ok: status.status === "healthy" },
                  { label: "Models", value: status.models_loaded?.length || 0, ok: (status.models_loaded?.length || 0) >= 8 },
                  { label: "Database", value: status.database_connected ? "Connected" : "Offline", ok: status.database_connected },
                  { label: "Version", value: status.model_version || "N/A" },
                ].map((s, i) => (
                  <div key={i} style={{ background: LTGRAY, borderRadius: 10, padding: 16 }}>
                    <p style={{ fontSize: 12, color: TXT2, marginBottom: 4 }}>{s.label}</p>
                    <p style={{ fontSize: 16, fontWeight: 600, color: s.ok === false ? "#ef4444" : s.ok ? "#10b981" : TXT }}>{s.value}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: TXT2, fontSize: 14 }}>Click "Refresh" to check system status</p>
            )}
          </div>
        </FadeIn>
 
        <FadeIn delay={0.2}>
          <div style={{ background: WHITE, borderRadius: 16, padding: 28, border: "1px solid #e5e7eb" }}>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>Upload claims dataset</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label style={{ fontSize: 13, fontWeight: 500, color: TXT2, display: "block", marginBottom: 6 }}>Coverage type</label>
                <select value={covType} onChange={e => setCovType(e.target.value)}
                  style={{ width: "100%", padding: "10px 14px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 14, fontFamily: "inherit" }}>
                  {["ipd", "opd", "dental", "maternity"].map(c => <option key={c} value={c}>{c.toUpperCase()}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: 13, fontWeight: 500, color: TXT2, display: "block", marginBottom: 6 }}>CSV file</label>
                <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)}
                  style={{ width: "100%", padding: "10px 14px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 14 }} />
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={autoRetrain} onChange={e => setAutoRetrain(e.target.checked)} />
                <span style={{ fontSize: 14, color: TXT }}>Auto-retrain model after upload</span>
              </label>
              <button className="gold-btn" onClick={handleUpload} disabled={!file || uploading}
                style={{ opacity: !file || uploading ? 0.5 : 1 }}>
                {uploading ? "Uploading..." : "Upload dataset"}
              </button>
            </div>
            {uploadResult && (
              <div style={{ marginTop: 16, padding: 16, borderRadius: 10, background: uploadResult.status === "accepted" ? "#ecfdf5" : "#fef2f2", border: `1px solid ${uploadResult.status === "accepted" ? "#a7f3d0" : "#fecaca"}` }}>
                <pre style={{ fontSize: 12, whiteSpace: "pre-wrap", margin: 0, color: TXT }}>{JSON.stringify(uploadResult, null, 2)}</pre>
              </div>
            )}
          </div>
        </FadeIn>
 
        {/* User behavior analytics */}
        <FadeIn delay={0.3}>
          <div style={{ background: WHITE, borderRadius: 16, padding: 28, border: "1px solid #e5e7eb", marginTop: 24 }}>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>User quote data</h3>
            <AdminUserData apiKey={key} apiUrl={API} />
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
 
function AdminUserData({ apiKey, apiUrl }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);
 
  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${apiUrl}/api/v2/admin/user-behavior`, { headers: { "X-API-Key": apiKey } });
      setData(await r.json());
    } catch { setData({ status: "error" }); }
    finally { setLoading(false); }
  };
 
  useEffect(() => { load(); }, []);
 
  if (!data) return <p style={{ fontSize: 13, color: TXT2, textAlign: "center", padding: 16 }}>{loading ? "Loading..." : "No data"}</p>;
  if (data.status === "no_db") return <p style={{ fontSize: 13, color: TXT2, textAlign: "center", padding: 16 }}>Database not connected</p>;
  if (data.status === "error") return <p style={{ fontSize: 13, color: "#ef4444", textAlign: "center", padding: 16 }}>Failed to load user data</p>;
 
  const { summary = {}, records = [] } = data;
 
  if (records.length === 0) return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <p style={{ fontSize: 13, color: TXT2 }}>No user quotes recorded yet</p>
      <p style={{ fontSize: 12, color: TXT2, marginTop: 4 }}>Quotes will appear here as users calculate premiums</p>
    </div>
  );
 
  const tierColors = { Bronze: "#92400e", Silver: "#475569", Gold: "#b07a0a", Platinum: "#1e40af" };
  const tierBg = { Bronze: "#fffbeb", Silver: "#f1f3f5", Gold: "#fef9ec", Platinum: "#eff6ff" };
 
  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
        {[
          { label: "Total quotes", value: summary.total_quotes || 0 },
          { label: "Avg age", value: summary.avg_age || "—" },
          { label: "OPD rider %", value: `${summary.rider_rates?.opd || 0}%` },
          { label: "Dental rider %", value: `${summary.rider_rates?.dental || 0}%` },
        ].map((s, i) => (
          <div key={i} style={{ background: LTGRAY, borderRadius: 10, padding: 14, textAlign: "center" }}>
            <p style={{ fontSize: 11, color: TXT2, marginBottom: 4 }}>{s.label}</p>
            <p style={{ fontSize: 22, fontWeight: 600, color: TXT }}>{s.value}</p>
          </div>
        ))}
      </div>
 
      {/* Tier distribution */}
      {summary.tier_distribution && Object.keys(summary.tier_distribution).length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: TXT2, marginBottom: 8, fontWeight: 600 }}>Tier distribution</p>
          <div style={{ display: "flex", gap: 8 }}>
            {Object.entries(summary.tier_distribution).map(([tier, count]) => {
              const pct = Math.round(count / (summary.total_quotes || 1) * 100);
              return (
                <div key={tier} style={{ flex: 1, background: tierBg[tier] || LTGRAY, borderRadius: 10, padding: "12px 8px", textAlign: "center" }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: tierColors[tier] || TXT }}>{tier}</p>
                  <p style={{ fontSize: 20, fontWeight: 700, color: tierColors[tier] || TXT }}>{pct}%</p>
                  <p style={{ fontSize: 10, color: TXT2 }}>{count} quotes</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
 
      {/* Smoking distribution */}
      {summary.smoking_distribution && Object.keys(summary.smoking_distribution).length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: TXT2, marginBottom: 8, fontWeight: 600 }}>Smoking status</p>
          <div style={{ display: "flex", gap: 8 }}>
            {Object.entries(summary.smoking_distribution).map(([s, count]) => {
              const pct = Math.round(count / (summary.total_quotes || 1) * 100);
              const colors = { Never: "#059669", Former: "#b07a0a", Current: "#dc2626" };
              const bgs = { Never: "#e1f5ee", Former: "#fffbeb", Current: "#fef2f2" };
              return (
                <div key={s} style={{ flex: 1, background: bgs[s] || LTGRAY, borderRadius: 10, padding: "12px 8px", textAlign: "center" }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: colors[s] || TXT }}>{s}</p>
                  <p style={{ fontSize: 20, fontWeight: 700, color: colors[s] || TXT }}>{pct}%</p>
                  <p style={{ fontSize: 10, color: TXT2 }}>{count} users</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
 
      {/* Toggle table */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <button onClick={() => setShowTable(!showTable)} className="gold-btn" style={{ padding: "8px 16px", fontSize: 12, background: showTable ? NAVY : GOLD, color: showTable ? WHITE : NAVY }}>
          {showTable ? "Hide" : "Show"} recent quotes ({records.length})
        </button>
        <button onClick={load} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #d1d5db", background: WHITE, fontSize: 12, cursor: "pointer", fontFamily: "inherit", color: TXT2 }}>
          Refresh
        </button>
      </div>
 
      {/* Records table */}
      {showTable && (
        <div style={{ overflowX: "auto", marginTop: 8 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                {["Time", "Age", "Gender", "Region", "Smoking", "Occupation", "Conditions", "Tier", "Riders", "Family"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "8px 6px", fontSize: 11, color: TXT2, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.slice(0, 30).map((r, i) => {
                const riders = [r.include_opd && "OPD", r.include_dental && "Den", r.include_maternity && "Mat"].filter(Boolean).join(", ") || "—";
                return (
                  <tr key={i} style={{ borderBottom: "1px solid #f1f3f5" }}>
                    <td style={{ padding: "8px 6px", color: TXT2, fontSize: 11 }}>{r.created_at ? new Date(r.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                    <td style={{ padding: "8px 6px" }}>{r.age}</td>
                    <td style={{ padding: "8px 6px" }}>{r.gender}</td>
                    <td style={{ padding: "8px 6px", fontSize: 11 }}>{r.region}</td>
                    <td style={{ padding: "8px 6px" }}>
                      <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, background: r.smoking === "Current" ? "#fef2f2" : r.smoking === "Former" ? "#fffbeb" : "#e1f5ee", color: r.smoking === "Current" ? "#dc2626" : r.smoking === "Former" ? "#b07a0a" : "#059669" }}>{r.smoking}</span>
                    </td>
                    <td style={{ padding: "8px 6px", fontSize: 11 }}>{r.occupation}</td>
                    <td style={{ padding: "8px 6px" }}>{r.preexist_count || 0}</td>
                    <td style={{ padding: "8px 6px" }}>
                      <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, background: tierBg[r.ipd_tier] || LTGRAY, color: tierColors[r.ipd_tier] || TXT, fontWeight: 600 }}>{r.ipd_tier}</span>
                    </td>
                    <td style={{ padding: "8px 6px", fontSize: 11 }}>{riders}</td>
                    <td style={{ padding: "8px 6px" }}>{r.family_size}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}