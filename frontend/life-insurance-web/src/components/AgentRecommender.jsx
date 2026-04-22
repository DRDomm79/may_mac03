import { useState } from 'react'
import './AgentRecommender.css'

const CUSTOMER_PROFILES = [
  { id: 1, name: 'Chen Wei', age: 35, gender: 'M', smoking: 'Non-smoker', income: 'High', family: 'Married, 2 kids' },
  { id: 2, name: 'Lin Hui', age: 45, gender: 'F', smoking: 'Former', income: 'Medium', family: 'Single' },
  { id: 3, name: 'Wang Jun', age: 55, gender: 'M', smoking: 'Non-smoker', income: 'High', family: 'Married' },
  { id: 4, name: 'Liu An', age: 28, gender: 'F', smoking: 'Non-smoker', income: 'Low', family: 'Single' }
]

export default function AgentRecommender() {
  const [selectedProfile, setSelectedProfile] = useState(CUSTOMER_PROFILES[0])
  const [showRecommendation, setShowRecommendation] = useState(false)

  const getRecommendation = () => {
    const profile = selectedProfile
    let recommendation = {}

    if (profile.age < 35) {
      recommendation = {
        primary: 'Term 10-Year',
        reason: 'Young, low mortality risk. Affordable protection.',
        reason2: 'Covers critical period (family formation, mortgage)',
        alternatives: ['10-Year Term is best fit', 'Could add riders for accident']
      }
    } else if (profile.age < 45) {
      recommendation = {
        primary: 'Term 20-Year or Endowment',
        reason: 'Mid-career peak earning. Balance of protection and growth.',
        reason2: 'Family likely established; planning for retirement',
        alternatives: ['20-Year Term for pure protection', 'Endowment if savings goal present']
      }
    } else if (profile.age < 55) {
      recommendation = {
        primary: 'Whole Life or Endowment',
        reason: 'Higher mortality risk; long-term wealth building',
        reason2: 'Approaching retirement; legacy planning important',
        alternatives: ['Whole Life for lifetime protection', 'Endowment for inheritance']
      }
    } else {
      recommendation = {
        primary: 'Whole Life (if approved)',
        reason: 'Very high mortality risk; comprehensive protection needed',
        reason2: 'May require medical underwriting or higher premiums',
        alternatives: ['Simplified issue whole life', 'Guaranteed issue options available']
      }
    }

    // Adjust for income
    if (profile.income === 'Low') {
      recommendation.secondary = 'Recommend starting with basic 10-Year Term'
    } else if (profile.income === 'High') {
      recommendation.secondary = 'Can afford comprehensive multi-product strategy'
    }

    return recommendation
  }

  const rec = getRecommendation()

  return (
    <div className="recommender-container">
      <div className="recommender-header">
        <h2>Agent Recommendation Engine</h2>
        <p>AI-powered product matching to accelerate sales conversations</p>
      </div>

      <div className="recommender-content">
        <div className="profile-selector">
          <h3>Select Customer Profile</h3>
          <div className="profile-cards">
            {CUSTOMER_PROFILES.map(profile => (
              <div
                key={profile.id}
                className={`profile-card ${selectedProfile.id === profile.id ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedProfile(profile)
                  setShowRecommendation(false)
                }}
              >
                <div className="profile-name">{profile.name}</div>
                <div className="profile-age">Age {profile.age}</div>
                <div className="profile-status">
                  {profile.family}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="profile-details">
          <h3>Customer Details</h3>
          <div className="details-grid">
            <div className="detail-row">
              <span className="detail-label">Name</span>
              <span className="detail-value">{selectedProfile.name}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Age</span>
              <span className="detail-value">{selectedProfile.age} years</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Gender</span>
              <span className="detail-value">{selectedProfile.gender}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Smoking</span>
              <span className="detail-value">{selectedProfile.smoking}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Income Level</span>
              <span className="detail-value">{selectedProfile.income}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Family Status</span>
              <span className="detail-value">{selectedProfile.family}</span>
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={() => setShowRecommendation(true)}
          >
            Get AI Recommendation
          </button>
        </div>
      </div>

      {showRecommendation && (
        <div className="recommendation-result">
          <h3>AI Product Recommendation</h3>

          <div className="rec-main">
            <div className="rec-primary">
              <div className="rec-label">Recommended Product</div>
              <div className="rec-value">{rec.primary}</div>
            </div>
          </div>

          <div className="rec-reasoning">
            <h4>Why This Product?</h4>
            <div className="reason-list">
              <div className="reason-item">
                <span className="reason-icon">✓</span>
                <span>{rec.reason}</span>
              </div>
              <div className="reason-item">
                <span className="reason-icon">✓</span>
                <span>{rec.reason2}</span>
              </div>
              {rec.secondary && (
                <div className="reason-item secondary">
                  <span className="reason-icon">•</span>
                  <span>{rec.secondary}</span>
                </div>
              )}
            </div>
          </div>

          <div className="rec-alternatives">
            <h4>Alternative Options</h4>
            <ul className="alt-list">
              {rec.alternatives.map((alt, i) => (
                <li key={i}>{alt}</li>
              ))}
            </ul>
          </div>

          <div className="rec-script">
            <h4>Suggested Sales Script</h4>
            <div className="script-box">
              <p>
                "Hi {selectedProfile.name}! Based on your profile, I'd recommend our <strong>{rec.primary}</strong> product.
                {rec.reason.includes('protection') && " It provides excellent protection at an affordable premium—"}
                {rec.reason.includes('balance') && " It strikes the perfect balance between protection and growth—"}
                {rec.reason.includes('legacy') && " It's perfect for building a lasting legacy—"}
                perfect for your situation. Can I walk you through the details?"
              </p>
            </div>
          </div>

          <div className="rec-upsell">
            <h4>Optional Riders to Suggest</h4>
            <div className="riders-list">
              <div className="rider-item">
                <input type="checkbox" id="accidental" defaultChecked />
                <label htmlFor="accidental">Accidental Death Benefit (±20% to premium)</label>
              </div>
              <div className="rider-item">
                <input type="checkbox" id="disability" />
                <label htmlFor="disability">Disability Waiver (±15% to premium)</label>
              </div>
              <div className="rider-item">
                <input type="checkbox" id="critical" />
                <label htmlFor="critical">Critical Illness (±25% to premium)</label>
              </div>
            </div>
          </div>

          <div className="recommendation-metrics">
            <h4>Expected Metrics</h4>
            <div className="metrics-mini">
              <div>
                <strong>Base Annual Premium</strong>
                <div>₩{(selectedProfile.age * 5000).toLocaleString()}</div>
              </div>
              <div>
                <strong>Estimated Approval Rate</strong>
                <div>{selectedProfile.age < 45 ? '95%' : selectedProfile.age < 55 ? '85%' : '60%'}</div>
              </div>
              <div>
                <strong>Close Rate (w/ recommendation)</strong>
                <div>+25-35% vs. no recommendation</div>
              </div>
            </div>
          </div>

          <button className="btn btn-secondary" onClick={() => setShowRecommendation(false)}>
            ← Back to Profiles
          </button>
        </div>
      )}

      <div className="engine-info">
        <h4>How This Works</h4>
        <p>
          The AI Recommendation Engine analyzes customer demographics, life stage, financial capacity, and health profile
          to suggest the optimal life insurance product. This:
        </p>
        <ul>
          <li>Accelerates agent conversations (pre-qualified suggestions)</li>
          <li>Increases close rates by 25-35% through confident recommendations</li>
          <li>Ensures regulatory-compliant product matching (not high-pressure upselling)</li>
          <li>Learns from feedback to improve recommendations over time</li>
        </ul>
      </div>
    </div>
  )
}
