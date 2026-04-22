import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { authFetch } from './auth';

const PSI_WARNING = 0.10;
const PSI_DRIFT   = 0.25;

export default function DriftMonitor({ backendUrl }) {
  const [data, setData]       = useState([]);
  const [current, setCurrent] = useState(null);
  const [status, setStatus]   = useState('stable');
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    authFetch(`${backendUrl}/dashboard/stats`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(json => {
        setCurrent(json.psi.current);
        setStatus(json.psi.status);
        setData(
          json.psi_time_series.map(d => ({
            date: d.date,
            psi:  parseFloat(d.psi.toFixed(4)),
            n:    d.n_cases,
          }))
        );
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [backendUrl]);

  const statusColor = { stable: '#22c55e', warning: '#f59e0b', drift: '#ef4444' }[status] ?? '#6b7280';

  if (loading) return <div style={S.card}><p style={S.muted}>Loading drift data…</p></div>;
  if (error)   return (
    <div style={S.card}>
      <p style={{ color: '#ef4444', fontSize: 13 }}>
        Backend unavailable ({error}). Deploy <code>api/main.py</code> to see live PSI.
      </p>
    </div>
  );

  return (
    <div style={S.card}>
      <h2 style={S.heading}>Model Drift Monitor (PSI)</h2>
      <p style={S.subtext}>
        Current PSI:{' '}
        <span style={{ color: statusColor, fontWeight: 700 }}>{current?.toFixed(4) ?? '—'}</span>
        {' '}—{' '}
        <span style={{ color: statusColor, textTransform: 'uppercase', fontWeight: 600 }}>{status}</span>
      </p>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
          <YAxis domain={[0, 0.35]} tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v, _n, p) => [v.toFixed(4), `PSI (n=${p.payload?.n ?? '?'})`]} />
          <ReferenceLine y={PSI_WARNING} stroke="#f59e0b" strokeDasharray="4 4"
            label={{ value: 'Warn 0.10', fill: '#f59e0b', fontSize: 10, position: 'insideTopLeft' }} />
          <ReferenceLine y={PSI_DRIFT}   stroke="#ef4444" strokeDasharray="4 4"
            label={{ value: 'Drift 0.25', fill: '#ef4444', fontSize: 10, position: 'insideTopLeft' }} />
          <Line type="monotone" dataKey="psi" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
        </LineChart>
      </ResponsiveContainer>

      <p style={{ ...S.muted, marginTop: 8 }}>
        PSI &lt; 0.10 = stable · 0.10–0.25 = monitor · ≥ 0.25 = retrain
      </p>
    </div>
  );
}

const S = {
  card:    { padding: 20, background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 20 },
  heading: { fontSize: 15, fontWeight: 700, marginBottom: 4, color: '#0d2b7a' },
  subtext: { fontSize: 13, color: '#374151', marginBottom: 12 },
  muted:   { fontSize: 11, color: '#9ca3af', margin: 0 },
};
