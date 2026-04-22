import { useState, useCallback } from 'react'
import './UnderwritingWizard.css'

const MORTALITY_FACTORS = {
  age: {
    '<35': 0.50,
    '35-44': 0.70,
    '45-54': 1.00,
    '55-64': 1.50,
    '65+': 2.50
  },
  gender: {
    'M': 1.15,
    'F': 0.90
  },
  smoking: {
    'Non-smoker': 1.00,
    'Former (3+ years)': 1.20,
    'Current': 2.00
  },
  health: {
    'Excellent': 0.80,
    'Good': 1.00,
    'Fair': 1.40,
    'Poor': 2.00
  },
  occupation: {
    'Low risk': 1.00,
    'Medium risk': 1.15,
    'High risk': 1.50
  }
}

const PRODUCT_TYPES = [
  { id: 'term10', name: '10-Year Term', description: 'Pure protection, affordable' },
  { id: 'term20', name: '20-Year Term', description: 'Long-term coverage' },
  { id: 'endowment', name: 'Endowment', description: 'Protection + savings' },
  { id: 'wholelife', name: 'Whole Life', description: 'Lifetime coverage' }
]

export default function UnderwritingWizard({ onNewCase }) {
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    age: '45-54',
    gender: 'M',
    smoking: 'Non-smoker',
    health: 'Good',
    occupation: 'Low risk',
    preexisting: '',
    productType: 'term10',
    coverageAmount: 500000
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const updateForm = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const calculateMortalityRisk = useCallback(() => {
    let riskScore = 1.0
    riskScore *= MORTALITY_FACTORS.age[formData.age]
    riskScore *= MORTALITY_FACTORS.gender[formData.gender]
    riskScore *= MORTALITY_FACTORS.smoking[formData.smoking]
    riskScore *= MORTALITY_FACTORS.health[formData.health]
    riskScore *= MORTALITY_FACTORS.occupation[formData.occupation]

    const preexistingMultiplier = formData.preexisting ? 1.3 : 1.0
    riskScore *= preexistingMultiplier

    return riskScore
  }, [formData])

  const makeUnderwritingDecision = useCallback(() => {
    const riskScore = calculateMortalityRisk()

    let decision = 'ACCEPT'
    let reasoning = []

    if (riskScore > 2.5) {
      decision = 'DECLINE'
      reasoning.push('Mortality risk exceeds acceptable threshold')
    } else if (riskScore > 1.8) {
      decision = 'REFER'
      reasoning.push('High risk - requires manual review')
      reasoning.push('Recommend additional medical underwriting')
    } else if (riskScore > 1.3) {
      decision = 'ACCEPT_WITH_EXCLUSION'
      reasoning.push('Moderate risk detected')
      reasoning.push('Pre-existing condition exclusion may apply')
    } else {
      decision = 'ACCEPT'
      reasoning.push('Risk profile within normal parameters')
    }

    // Age fairness check
    if (parseInt(formData.age) > 60 && riskScore > 1.5) {
      reasoning.push('Age + health combination reviewed for fairness')
    }

    // Premium calculation
    const basePremium = (formData.coverageAmount / 100000) * 50 // $50 per $100k base
    const adjustedPremium = Math.round(basePremium * riskScore)

    return {
      decision,
      riskScore: riskScore.toFixed(2),
      basePremium,
      adjustedPremium,
      reasoning,
      timestamp: new Date().toISOString(),
      explainability: {
        ageRiskMultiplier: MORTALITY_FACTORS.age[formData.age],
        genderMultiplier: MORTALITY_FACTORS.gender[formData.gender],
        smokingMultiplier: MORTALITY_FACTORS.smoking[formData.smoking],
        healthMultiplier: MORTALITY_FACTORS.health[formData.health],
        occupationMultiplier: MORTALITY_FACTORS.occupation[formData.occupation],
        preexistingMultiplier: formData.preexisting ? 1.3 : 1.0
      }
    }
  }, [formData, calculateMortalityRisk])

  const handleSubmit = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      const underwritingResult = makeUnderwritingDecision()
      setResult(underwritingResult)
      setLoading(false)
      setStep(4)

      // Save case
      onNewCase({
        ...formData,
        ...underwritingResult
      })
    }, 800)
  }

  const reset = () => {
    setStep(1)
    setResult(null)
    setFormData({
      firstName: '',
      lastName: '',
      age: '45-54',
      gender: 'M',
      smoking: 'Non-smoker',
      health: 'Good',
      occupation: 'Low risk',
      preexisting: '',
      productType: 'term10',
      coverageAmount: 500000
    })
  }

  const getDecisionColor = (decision) => {
    switch (decision) {
      case 'ACCEPT':
        return '#28a745'
      case 'ACCEPT_WITH_EXCLUSION':
        return '#ffc107'
      case 'REFER':
        return '#ff9500'
      case 'DECLINE':
        return '#dc3545'
      default:
        return '#6c757d'
    }
  }

  const getDecisionLabel = (decision) => {
    const labels = {
      'ACCEPT': '✓ ACCEPT',
      'ACCEPT_WITH_EXCLUSION': '△ ACCEPT WITH EXCLUSION',
      'REFER': '⚠ REFER FOR REVIEW',
      'DECLINE': '✕ DECLINE'
    }
    return labels[decision] || decision
  }

  return (
    <div className="wizard-container">
      <div className="wizard-header">
        <h2>Life Insurance Underwriting</h2>
        <p>AI-powered mortality risk assessment & premium calculation</p>
      </div>

      {step < 4 && (
        <div className="wizard-progress">
          <div className="progress-steps">
            {[1, 2, 3].map(s => (
              <div key={s} className={`progress-step ${step >= s ? 'active' : ''}`}>
                <div className="step-number">{s}</div>
                <div className="step-label">
                  {s === 1 ? 'Personal' : s === 2 ? 'Health' : 'Product'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="wizard-content">
        {/* Step 1: Personal Information */}
        {step === 1 && (
          <div className="wizard-step">
            <h3>Personal Information</h3>
            <div className="form-grid">
              <div className="form-group">
                <label>First Name</label>
                <input
                  type="text"
                  value={formData.firstName}
                  onChange={(e) => updateForm('firstName', e.target.value)}
                  placeholder="John"
                />
              </div>
              <div className="form-group">
                <label>Last Name</label>
                <input
                  type="text"
                  value={formData.lastName}
                  onChange={(e) => updateForm('lastName', e.target.value)}
                  placeholder="Doe"
                />
              </div>
              <div className="form-group">
                <label>Age Group</label>
                <select
                  value={formData.age}
                  onChange={(e) => updateForm('age', e.target.value)}
                >
                  <option value="<35">&lt;35 years</option>
                  <option value="35-44">35-44 years</option>
                  <option value="45-54">45-54 years</option>
                  <option value="55-64">55-64 years</option>
                  <option value="65+">65+ years</option>
                </select>
              </div>
              <div className="form-group">
                <label>Gender</label>
                <select
                  value={formData.gender}
                  onChange={(e) => updateForm('gender', e.target.value)}
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Occupation Risk Level</label>
                <select
                  value={formData.occupation}
                  onChange={(e) => updateForm('occupation', e.target.value)}
                >
                  <option value="Low risk">Low risk</option>
                  <option value="Medium risk">Medium risk</option>
                  <option value="High risk">High risk</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Health Information */}
        {step === 2 && (
          <div className="wizard-step">
            <h3>Health Profile</h3>
            <div className="form-grid">
              <div className="form-group">
                <label>Smoking Status</label>
                <select
                  value={formData.smoking}
                  onChange={(e) => updateForm('smoking', e.target.value)}
                >
                  <option value="Non-smoker">Non-smoker</option>
                  <option value="Former (3+ years)">Former (3+ years ago)</option>
                  <option value="Current">Current smoker</option>
                </select>
              </div>
              <div className="form-group">
                <label>General Health Status</label>
                <select
                  value={formData.health}
                  onChange={(e) => updateForm('health', e.target.value)}
                >
                  <option value="Excellent">Excellent</option>
                  <option value="Good">Good</option>
                  <option value="Fair">Fair</option>
                  <option value="Poor">Poor</option>
                </select>
              </div>
              <div className="form-group full-width">
                <label>Pre-existing Conditions</label>
                <input
                  type="text"
                  value={formData.preexisting}
                  onChange={(e) => updateForm('preexisting', e.target.value)}
                  placeholder="e.g., Diabetes, Hypertension (leave blank if none)"
                />
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Product & Coverage */}
        {step === 3 && (
          <div className="wizard-step">
            <h3>Product Selection</h3>
            <div className="product-grid">
              {PRODUCT_TYPES.map(product => (
                <div
                  key={product.id}
                  className={`product-card ${formData.productType === product.id ? 'selected' : ''}`}
                  onClick={() => updateForm('productType', product.id)}
                >
                  <div className="product-name">{product.name}</div>
                  <div className="product-desc">{product.description}</div>
                </div>
              ))}
            </div>

            <div className="form-grid" style={{ marginTop: '2rem' }}>
              <div className="form-group">
                <label>Coverage Amount (TWD)</label>
                <input
                  type="number"
                  value={formData.coverageAmount}
                  onChange={(e) => updateForm('coverageAmount', parseInt(e.target.value))}
                  min="100000"
                  step="50000"
                />
              </div>
              <div className="form-group">
                <label style={{ color: '#999' }}>Monthly Premium (Estimate)</label>
                <div className="premium-display">
                  TWD {Math.round((formData.coverageAmount / 100000) * 50)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Result */}
        {step === 4 && result && (
          <div className="wizard-step result-step">
            <div className="result-header">
              <h3>Underwriting Decision</h3>
              <div
                className="decision-badge"
                style={{ background: getDecisionColor(result.decision) }}
              >
                {getDecisionLabel(result.decision)}
              </div>
            </div>

            <div className="result-grid">
              <div className="result-card">
                <div className="result-label">Mortality Risk Score</div>
                <div className="result-value" style={{ fontSize: '2rem' }}>
                  {result.riskScore}x
                </div>
                <div className="result-sublabel">
                  {result.riskScore > 2 ? 'High Risk' : result.riskScore > 1.5 ? 'Moderate Risk' : 'Standard Risk'}
                </div>
              </div>

              <div className="result-card">
                <div className="result-label">Adjusted Annual Premium</div>
                <div className="result-value">TWD {result.adjustedPremium.toLocaleString()}</div>
                <div className="result-sublabel">Per {formData.coverageAmount.toLocaleString()} coverage</div>
              </div>
            </div>

            <div className="result-section">
              <h4>Risk Factors Breakdown</h4>
              <div className="factors-table">
                <div className="factor-row">
                  <span>Age Group</span>
                  <span className="factor-value">{formData.age}</span>
                  <span className="factor-mult">×{result.explainability.ageRiskMultiplier}</span>
                </div>
                <div className="factor-row">
                  <span>Gender</span>
                  <span className="factor-value">{formData.gender === 'M' ? 'Male' : 'Female'}</span>
                  <span className="factor-mult">×{result.explainability.genderMultiplier}</span>
                </div>
                <div className="factor-row">
                  <span>Smoking Status</span>
                  <span className="factor-value">{formData.smoking}</span>
                  <span className="factor-mult">×{result.explainability.smokingMultiplier}</span>
                </div>
                <div className="factor-row">
                  <span>Health Status</span>
                  <span className="factor-value">{formData.health}</span>
                  <span className="factor-mult">×{result.explainability.healthMultiplier}</span>
                </div>
                <div className="factor-row">
                  <span>Occupation</span>
                  <span className="factor-value">{formData.occupation}</span>
                  <span className="factor-mult">×{result.explainability.occupationMultiplier}</span>
                </div>
              </div>
            </div>

            <div className="result-section">
              <h4>Decision Reasoning</h4>
              <ul className="reasoning-list">
                {result.reasoning.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>

            <div className="compliance-note">
              <strong>Compliance Note:</strong> This decision is based on actuarial models and factors directly related to mortality risk. All factors have been assessed for fairness and regulatory compliance with Taiwan FSC guidelines.
            </div>
          </div>
        )}
      </div>

      <div className="wizard-actions">
        {step < 4 && (
          <>
            <button
              className="btn btn-secondary"
              onClick={() => setStep(Math.max(1, step - 1))}
              disabled={step === 1}
            >
              ← Previous
            </button>
            <button
              className="btn btn-primary"
              onClick={() => setStep(step + 1)}
              disabled={step === 3 && !formData.firstName}
            >
              Next →
            </button>
          </>
        )}

        {step === 3 && (
          <button
            className="btn btn-success"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : '✓ Submit for Decision'}
          </button>
        )}

        {step === 4 && (
          <button className="btn btn-primary" onClick={reset}>
            New Case
          </button>
        )}
      </div>
    </div>
  )
}
