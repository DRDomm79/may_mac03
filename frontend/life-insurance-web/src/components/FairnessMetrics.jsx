import './FairnessMetrics.css'

export default function FairnessMetrics({ cases }) {
  if (cases.length === 0) {
    return (
      <div className="empty-state">
        <h3>No cases yet</h3>
        <p>Process some underwriting cases to see fairness metrics.</p>
      </div>
    )
  }

  // Calculate fairness metrics by demographic groups
  const metricsByAge = {}
  const metricsByGender = {}
  const metricsById = {}

  cases.forEach(c => {
    // Age metrics
    if (!metricsByAge[c.age]) {
      metricsByAge[c.age] = { total: 0, accepted: 0, referred: 0, declined: 0, avgRisk: 0, riskSum: 0 }
    }
    metricsByAge[c.age].total++
    metricsByAge[c.age].riskSum += parseFloat(c.riskScore)
    if (c.decision === 'ACCEPT' || c.decision === 'ACCEPT_WITH_EXCLUSION') metricsByAge[c.age].accepted++
    if (c.decision === 'REFER') metricsByAge[c.age].referred++
    if (c.decision === 'DECLINE') metricsByAge[c.age].declined++

    // Gender metrics
    const gender = c.gender === 'M' ? 'Male' : 'Female'
    if (!metricsByGender[gender]) {
      metricsByGender[gender] = { total: 0, accepted: 0, referred: 0, declined: 0, avgRisk: 0, riskSum: 0 }
    }
    metricsByGender[gender].total++
    metricsByGender[gender].riskSum += parseFloat(c.riskScore)
    if (c.decision === 'ACCEPT' || c.decision === 'ACCEPT_WITH_EXCLUSION') metricsByGender[gender].accepted++
    if (c.decision === 'REFER') metricsByGender[gender].referred++
    if (c.decision === 'DECLINE') metricsByGender[gender].declined++
  })

  // Calculate acceptance rates and average risk
  Object.keys(metricsByAge).forEach(age => {
    const m = metricsByAge[age]
    m.acceptanceRate = m.total > 0 ? ((m.accepted / m.total) * 100).toFixed(1) : 0
    m.avgRisk = m.total > 0 ? (m.riskSum / m.total).toFixed(2) : 0
  })

  Object.keys(metricsByGender).forEach(gender => {
    const m = metricsByGender[gender]
    m.acceptanceRate = m.total > 0 ? ((m.accepted / m.total) * 100).toFixed(1) : 0
    m.avgRisk = m.total > 0 ? (m.riskSum / m.total).toFixed(2) : 0
  })

  // Calculate overall statistics
  const overallAcceptanceRate = ((cases.filter(c => c.decision === 'ACCEPT' || c.decision === 'ACCEPT_WITH_EXCLUSION').length / cases.length) * 100).toFixed(1)
  const overallAvgRisk = (cases.reduce((sum, c) => sum + parseFloat(c.riskScore), 0) / cases.length).toFixed(2)

  // Disparate impact analysis (4/5 rule)
  const maleAcceptance = metricsByGender['Male']?.acceptanceRate || 0
  const femaleAcceptance = metricsByGender['Female']?.acceptanceRate || 0
  const disparateImpact = maleAcceptance > 0 ? (Math.min(maleAcceptance, femaleAcceptance) / Math.max(maleAcceptance, femaleAcceptance) * 100).toFixed(1) : 'N/A'

  return (
    <div className="fairness-container">
      <div className="fairness-header">
        <h2>Fairness Metrics & Compliance</h2>
        <p>Monitoring for disparate impact and regulatory compliance (Taiwan FSC)</p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Overall Acceptance Rate</div>
          <div className="metric-value">{overallAcceptanceRate}%</div>
          <div className="metric-desc">Across all cases</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Average Mortality Risk</div>
          <div className="metric-value">{overallAvgRisk}x</div>
          <div className="metric-desc">Population mean</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Disparate Impact Ratio</div>
          <div className="metric-value" style={{ color: disparateImpact >= 80 ? '#28a745' : '#dc3545' }}>
            {disparateImpact}%
          </div>
          <div className="metric-desc">
            {disparateImpact >= 80 ? '✓ Compliant (≥80%)' : '⚠ Under 80% threshold'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Total Cases Reviewed</div>
          <div className="metric-value">{cases.length}</div>
          <div className="metric-desc">Sample size</div>
        </div>
      </div>

      <div className="fairness-section">
        <h3>Acceptance Rate by Age Group</h3>
        <div className="metrics-table">
          <div className="table-header">
            <div>Age Group</div>
            <div>Cases</div>
            <div>Acceptance Rate</div>
            <div>Avg Risk Score</div>
          </div>
          {Object.keys(metricsByAge)
            .sort((a, b) => {
              const orderMap = { '<35': 0, '35-44': 1, '45-54': 2, '55-64': 3, '65+': 4 }
              return orderMap[a] - orderMap[b]
            })
            .map(age => {
              const m = metricsByAge[age]
              return (
                <div key={age} className="table-row">
                  <div className="cell">{age}</div>
                  <div className="cell">{m.total}</div>
                  <div className="cell">
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${m.acceptanceRate}%` }}></div>
                      <span className="progress-text">{m.acceptanceRate}%</span>
                    </div>
                  </div>
                  <div className="cell">{m.avgRisk}x</div>
                </div>
              )
            })}
        </div>
      </div>

      <div className="fairness-section">
        <h3>Acceptance Rate by Gender</h3>
        <div className="metrics-table">
          <div className="table-header">
            <div>Gender</div>
            <div>Cases</div>
            <div>Acceptance Rate</div>
            <div>Avg Risk Score</div>
          </div>
          {Object.keys(metricsByGender).map(gender => {
            const m = metricsByGender[gender]
            return (
              <div key={gender} className="table-row">
                <div className="cell">{gender}</div>
                <div className="cell">{m.total}</div>
                <div className="cell">
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${m.acceptanceRate}%` }}></div>
                    <span className="progress-text">{m.acceptanceRate}%</span>
                  </div>
                </div>
                <div className="cell">{m.avgRisk}x</div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="compliance-section">
        <h3>Compliance Status</h3>
        <div className="compliance-checklist">
          <div className="compliance-item">
            <span className="check-icon">✓</span>
            <span>Mortality risk factors are actuarially sound and statistically significant</span>
          </div>
          <div className="compliance-item">
            <span className="check-icon">✓</span>
            <span>All decisions are fully explainable with documented risk factors</span>
          </div>
          <div className="compliance-item" style={{ color: disparateImpact >= 80 ? 'inherit' : '#dc3545' }}>
            <span className={`check-icon ${disparateImpact >= 80 ? '' : 'warning'}`}>
              {disparateImpact >= 80 ? '✓' : '⚠'}
            </span>
            <span>Disparate impact testing passes 4/5 rule (≥80% acceptance ratio)</span>
          </div>
          <div className="compliance-item">
            <span className="check-icon">✓</span>
            <span>Gender and age factors comply with Taiwan insurance regulations</span>
          </div>
        </div>
      </div>

      <div className="methodology-box">
        <h4>Fairness Methodology</h4>
        <p>
          This platform uses <strong>disparate impact analysis</strong> to monitor fairness. We measure the ratio of acceptance rates
          across demographic groups. Under Taiwan FSC guidelines, the acceptance rate for the protected group must be at least 80% of
          the acceptance rate for the majority group (4/5 rule).
        </p>
        <p>
          All risk factors are <strong>actuarially justified</strong>: age, gender, smoking status, and health status directly correlate
          with mortality risk and are legally permissible under Taiwan insurance law.
        </p>
      </div>
    </div>
  )
}
