import { useState, useEffect } from 'react'
import { GestureCamera } from './components/GestureCamera'
import { fetchGestures, fetchBenchmarkResults, fetchHealth } from './api'
import type { GestureInfo, BenchmarkResultSummary, HealthResponse } from './types'

type Tab = 'demo' | 'gestures' | 'benchmark'

export default function App() {
  const [tab, setTab] = useState<Tab>('demo')
  const [gestures, setGestures] = useState<GestureInfo[]>([])
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResultSummary[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    fetchHealth().then(setHealth).catch(console.error)
    fetchGestures().then((r) => setGestures(r.gestures)).catch(console.error)
    fetchBenchmarkResults().then((r) => setBenchmarkResults(r.results)).catch(console.error)
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 900, margin: '0 auto', padding: 24 }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 28 }}>SenyaSalin</h1>
        <p style={{ margin: '4px 0 0', color: '#718096', fontSize: 14 }}>
          Open Multimodal Benchmark for Filipino Accessibility Communication
        </p>
        {health && (
          <span
            style={{
              display: 'inline-block',
              marginTop: 8,
              padding: '2px 10px',
              borderRadius: 99,
              fontSize: 12,
              background: health.model_loaded ? '#c6f6d5' : '#fed7d7',
              color: health.model_loaded ? '#276749' : '#9b2c2c',
            }}
          >
            {health.model_loaded
              ? `Model: ${health.model_name} — ${health.num_classes} classes`
              : 'No model loaded — train first'}
          </span>
        )}
      </header>

      <nav style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {(['demo', 'gestures', 'benchmark'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 20px',
              borderRadius: 6,
              border: 'none',
              cursor: 'pointer',
              background: tab === t ? '#4299e1' : '#edf2f7',
              color: tab === t ? '#fff' : '#2d3748',
              fontWeight: tab === t ? 600 : 400,
            }}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      {tab === 'demo' && <GestureCamera />}

      {tab === 'gestures' && (
        <div>
          <h2 style={{ marginTop: 0 }}>Supported Gestures ({gestures.length})</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <thead>
              <tr style={{ background: '#edf2f7' }}>
                <th style={th}>Label</th>
                <th style={th}>Intent</th>
                <th style={th}>Description</th>
                <th style={th}>Scenarios</th>
              </tr>
            </thead>
            <tbody>
              {gestures.map((g) => (
                <tr key={g.label} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={td}><strong>{g.label}</strong></td>
                  <td style={td}>{g.intent}</td>
                  <td style={td}>{g.description}</td>
                  <td style={td}>{g.scenario_tags.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'benchmark' && (
        <div>
          <h2 style={{ marginTop: 0 }}>Benchmark Results</h2>
          {benchmarkResults.length === 0 ? (
            <p style={{ color: '#718096' }}>
              No benchmark results yet. Run:{' '}
              <code>python -m benchmark.run_benchmark</code>
            </p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ background: '#edf2f7' }}>
                  <th style={th}>Model</th>
                  <th style={th}>Accuracy</th>
                  <th style={th}>F1 (weighted)</th>
                  <th style={th}>Inference (ms)</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkResults
                  .sort((a, b) => b.f1_weighted - a.f1_weighted)
                  .map((r) => (
                    <tr key={r.model_name} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={td}><strong>{r.model_name}</strong></td>
                      <td style={td}>{(r.accuracy * 100).toFixed(1)}%</td>
                      <td style={td}>{(r.f1_weighted * 100).toFixed(1)}%</td>
                      <td style={td}>{r.inference_time_ms_mean.toFixed(2)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}

const th: React.CSSProperties = {
  textAlign: 'left',
  padding: '10px 12px',
  fontWeight: 600,
  color: '#4a5568',
}

const td: React.CSSProperties = {
  padding: '10px 12px',
  verticalAlign: 'top',
}
