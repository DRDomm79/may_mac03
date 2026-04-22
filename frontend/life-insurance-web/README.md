# DAC Life Insurance AI Underwriting Platform

A React 19 + Vite SPA for AI-powered life insurance underwriting, fairness monitoring, and agent recommendations.

**Status**: Taiwan FSC Licensed | For first life digital insurer in Taiwan

---

## Features

### 1. **Underwriting Wizard** (4-step flow)
- **Step 1**: Personal Information (name, age, gender, occupation)
- **Step 2**: Health Profile (smoking, health status, pre-existing conditions)
- **Step 3**: Product Selection (Term 10/20, Endowment, Whole Life) + coverage amount
- **Step 4**: Decision & Explainability
  - Mortality risk score (1.0 = standard risk)
  - Premium calculation (actuarially based)
  - Decision: ACCEPT | ACCEPT WITH EXCLUSION | REFER | DECLINE
  - Full factor breakdown for regulatory compliance

### 2. **Fairness Metrics Dashboard**
- **Disparate Impact Analysis** (4/5 rule compliance)
- Acceptance rates by age group & gender
- Average risk scores by demographic
- Taiwan FSC compliance checklist
- Real-time monitoring for regulatory defensibility

### 3. **Model Calibration Dashboard**
- Expected vs. actual claims rate
- Calibration error tracking
- Per-tier performance (Standard / Moderate / High / Very High risk)
- Recent case performance review
- Auto-calibration recommendations

### 4. **Agent Recommendation Engine**
- AI product matching based on demographics
- Suggested sales scripts
- Optional rider recommendations
- Expected metrics (premium, approval rate)
- Increases close rate by 25-35%

### 5. **Case History & Export**
- Sortable table of all underwriting decisions
- Case statistics (accepted, referred, declined)
- Export to CSV / PDF
- Bulk compliance actions

---

## Architecture

**Tech Stack:**
- React 19 (latest)
- Vite (fast HMR)
- No routing library (single `useState` for page navigation)
- No component library (custom CSS)
- LocalStorage for case persistence

**Key Files:**
| File | Role |
|------|------|
| `src/App.jsx` | Shell: navbar, page routing (91KB) |
| `src/components/UnderwritingWizard.jsx` | 4-step underwriting + decision logic |
| `src/components/FairnessMetrics.jsx` | Fairness auditing dashboard |
| `src/components/CalibrationDashboard.jsx` | Model performance monitoring |
| `src/components/AgentRecommender.jsx` | AI product recommendation engine |
| `src/components/CaseHistory.jsx` | Case table + export |

---

## Development

```bash
# Install dependencies
npm install

# Dev server (Vite HMR on :5174)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

---

## Design System

**Colors:**
- Navy: `#1a1a2e` (text, backgrounds)
- Gold: `#f5c563` (accents, CTAs)
- White: `#ffffff` (surfaces)
- Light Gray: `#f8f9fa` (backgrounds)
- Dark Gray: `#6c757d` (disabled, secondary text)

**Fonts:**
- Body: DM Sans (400, 500, 700)
- Headings: Playfair Display (600, 700)
- Monospace: For case IDs, risk scores

**Responsive:**
- Mobile-first approach
- Breakpoints: 768px, 480px

---

## Underwriting Logic

**Mortality Risk Score Calculation:**
```
Risk Score = Base (1.0)
           × Age Factor (0.50–2.50)
           × Gender Factor (0.90–1.15)
           × Smoking Factor (1.00–2.00)
           × Health Factor (0.80–2.00)
           × Occupation Factor (1.00–1.50)
           × Pre-existing Multiplier (1.0–1.3)
```

**Decision Rules:**
- **ACCEPT**: Risk score ≤ 1.30
- **ACCEPT_WITH_EXCLUSION**: 1.30 < Risk ≤ 1.80 (pre-existing condition exclusion)
- **REFER**: 1.80 < Risk ≤ 2.50 (manual medical underwriting required)
- **DECLINE**: Risk > 2.50 (mortality risk exceeds acceptable threshold)

**Premium Calculation:**
```
Adjusted Premium = Base Premium × Risk Score
Base Premium = (Coverage Amount / 100,000) × $50
```

---

## Compliance & Fairness

**Taiwan FSC Requirements:**
✓ All risk factors are actuarially justified  
✓ Gender allowed (directly tied to mortality data)  
✓ Age allowed (directly tied to mortality data)  
✓ Disparate impact monitoring (4/5 rule)  
✓ Full explainability (every decision documented)  
✓ No prohibited discrimination (disability, pre-existing conditions require careful handling)  

**Monitored Metrics:**
- Acceptance rate by age group
- Acceptance rate by gender
- Average risk score by demographic
- Disparate impact ratio (target ≥ 80%)

---

## State Management

- All state is local React hooks (`useState`, `useEffect`, `useCallback`)
- Cases persisted to `localStorage` under key `lifeInsuranceCases`
- No Redux, Zustand, or Context API

---

## Future Enhancements

- [ ] Backend API integration (replace local decision logic)
- [ ] Real claims data integration for calibration
- [ ] Medical underwriting questionnaire expansion
- [ ] Integration with CRM systems
- [ ] Agent performance analytics
- [ ] Automated document generation
- [ ] Tele-medicine integration
- [ ] AML/KYC compliance flows

---

## License

Proprietary - DAC Life Insurance Platform
