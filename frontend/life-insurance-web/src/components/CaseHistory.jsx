import './CaseHistory.css'

export default function CaseHistory({ cases }) {
  const getDecisionColor = (decision) => {
    const colorMap = {
      'ACCEPT': '#28a745',
      'ACCEPT_WITH_EXCLUSION': '#ffc107',
      'REFER': '#ff9500',
      'DECLINE': '#dc3545'
    }
    return colorMap[decision] || '#6c757d'
  }

  const getDecisionLabel = (decision) => {
    const labels = {
      'ACCEPT': 'Accept',
      'ACCEPT_WITH_EXCLUSION': 'Accept w/ Exclusion',
      'REFER': 'Refer for Review',
      'DECLINE': 'Decline'
    }
    return labels[decision] || decision
  }

  const sortedCases = [...cases].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

  if (cases.length === 0) {
    return (
      <div className="empty-state">
        <h3>No underwriting cases yet</h3>
        <p>Submit cases from the Underwriting tab to see history.</p>
      </div>
    )
  }

  return (
    <div className="case-history-container">
      <div className="history-header">
        <h2>Case History</h2>
        <p>All underwriting decisions and outcomes</p>
        <div className="case-stats">
          <div className="stat">
            <strong>{cases.length}</strong>
            <span>Total Cases</span>
          </div>
          <div className="stat">
            <strong>{cases.filter(c => c.decision === 'ACCEPT' || c.decision === 'ACCEPT_WITH_EXCLUSION').length}</strong>
            <span>Accepted</span>
          </div>
          <div className="stat">
            <strong>{cases.filter(c => c.decision === 'REFER').length}</strong>
            <span>Referred</span>
          </div>
          <div className="stat">
            <strong>{cases.filter(c => c.decision === 'DECLINE').length}</strong>
            <span>Declined</span>
          </div>
        </div>
      </div>

      <div className="cases-table-wrapper">
        <table className="cases-table">
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Applicant</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Health</th>
              <th>Risk Score</th>
              <th>Decision</th>
              <th>Annual Premium</th>
              <th>Date</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {sortedCases.map(c => (
              <tr key={c.id}>
                <td className="case-id">#{c.id.toString().slice(-6)}</td>
                <td className="applicant-name">
                  {c.firstName} {c.lastName}
                </td>
                <td>{c.age}</td>
                <td>{c.gender === 'M' ? 'Male' : 'Female'}</td>
                <td>
                  <span className={`health-badge ${c.health.toLowerCase().replace(/\s+/g, '-')}`}>
                    {c.health}
                  </span>
                </td>
                <td>
                  <strong>{c.riskScore}x</strong>
                </td>
                <td>
                  <span
                    className="decision-badge"
                    style={{ background: getDecisionColor(c.decision) }}
                  >
                    {getDecisionLabel(c.decision)}
                  </span>
                </td>
                <td>₩{c.adjustedPremium.toLocaleString()}</td>
                <td>{new Date(c.timestamp).toLocaleDateString()}</td>
                <td>
                  <button className="btn-expand">View Details</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="export-section">
        <h3>Export & Reporting</h3>
        <div className="export-buttons">
          <button className="btn btn-export">
            📊 Export to CSV
          </button>
          <button className="btn btn-export">
            📄 Generate Report (PDF)
          </button>
          <button className="btn btn-export">
            📈 Quarterly Analytics
          </button>
        </div>
      </div>

      <div className="bulk-actions">
        <h3>Bulk Actions</h3>
        <div className="action-buttons">
          <button className="btn btn-action">Print All Cases</button>
          <button className="btn btn-action">Archive Completed</button>
          <button className="btn btn-action">Send to Compliance</button>
        </div>
      </div>
    </div>
  )
}
