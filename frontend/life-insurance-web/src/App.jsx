import { useState, useEffect } from 'react'
import './App.css'
import UnderwritingWizard from './components/UnderwritingWizard'
import FairnessMetrics from './components/FairnessMetrics'
import AgentRecommender from './components/AgentRecommender'
import CaseHistory from './components/CaseHistory'

export default function App() {
  const [currentPage, setCurrentPage] = useState('underwriting')
  const [cases, setCases] = useState([])
  const [isNavOpen, setIsNavOpen] = useState(false)

  useEffect(() => {
    const savedCases = localStorage.getItem('lifeInsuranceCases')
    if (savedCases) {
      setCases(JSON.parse(savedCases))
    }
  }, [])

  const handleNewCase = (caseData) => {
    const newCase = {
      id: Date.now(),
      ...caseData,
      timestamp: new Date().toISOString(),
      status: caseData.decision
    }
    const updatedCases = [newCase, ...cases]
    setCases(updatedCases)
    localStorage.setItem('lifeInsuranceCases', JSON.stringify(updatedCases))
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'underwriting':
        return <UnderwritingWizard onNewCase={handleNewCase} />
      case 'fairness':
        return <FairnessMetrics cases={cases} />
      case 'recommendation':
        return <AgentRecommender />
      case 'history':
        return <CaseHistory cases={cases} />
      default:
        return <UnderwritingWizard onNewCase={handleNewCase} />
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <h1>DAC Life Insurance</h1>
            <p className="subtitle">AI-Powered Underwriting Platform</p>
          </div>
          <button
            className="nav-toggle"
            onClick={() => setIsNavOpen(!isNavOpen)}
            aria-label="Toggle navigation"
          >
            ☰
          </button>
        </div>

        <nav className={`app-nav ${isNavOpen ? 'open' : ''}`}>
          <button
            className={`nav-btn ${currentPage === 'underwriting' ? 'active' : ''}`}
            onClick={() => { setCurrentPage('underwriting'); setIsNavOpen(false) }}
          >
            Underwriting
          </button>
          <button
            className={`nav-btn ${currentPage === 'recommendation' ? 'active' : ''}`}
            onClick={() => { setCurrentPage('recommendation'); setIsNavOpen(false) }}
          >
            Agent Recommendations
          </button>
          <button
            className={`nav-btn ${currentPage === 'fairness' ? 'active' : ''}`}
            onClick={() => { setCurrentPage('fairness'); setIsNavOpen(false) }}
          >
            Fairness Metrics
          </button>
          <button
            className={`nav-btn ${currentPage === 'history' ? 'active' : ''}`}
            onClick={() => { setCurrentPage('history'); setIsNavOpen(false) }}
          >
            Case History ({cases.length})
          </button>
        </nav>
      </header>

      <main className="app-main">
        {renderPage()}
      </main>

      <footer className="app-footer">
        <p>&copy; 2026 DAC Life Insurance. Taiwan FSC Licensed AI Underwriting Platform.</p>
      </footer>
    </div>
  )
}
