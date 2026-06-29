import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

const HORIZONS = [1, 5, 10, 20, 30, 60, 90, 120, 150, 180, 252]

type Opportunity = {
  prediction_rank: number
  symbol: string
  name: string
  sector: string | null
  industry: string | null
  prediction_date: string
  horizon_days: number
  model_name: string
  predicted_return: number | null
  prediction_score: number | null
  risk_score: number | null
  final_score: number | null
}

type Prediction = {
  symbol: string
  horizon_days: number
  horizon_label: string
  prediction_date: string
  model_name: string
  model_display_name: string
  predicted_return: number | null
  predicted_return_percent: number | null
  prediction_score: number | null
  risk_score: number | null
  final_score: number | null
  prediction_rank: number | null
}

type JournalEntry = {
  id: number
  symbol: string
  horizon_days: number | null
  decision: string
  status: string
  title: string
  thesis: string | null
  plan: string | null
  notes: string | null
  emotion: string | null
  confidence: number | null
  created_at: string
}

type CompanyAnalysis = {
  overview: {
    id: number
    symbol: string
    name: string
    exchange: string | null
    currency: string | null
    sector: string | null
    industry: string | null
    country: string | null
    market_cap: number | null
    universe_name: string | null
    universe_rank: number | null
    latest_price: {
      date: string
      close: string | number | null
      adjusted_close: string | number | null
      volume: number | null
    } | null
  }
  predictions: Prediction[]
  journal: {
    entries: JournalEntry[]
    count: number
  }
}

type PricePoint = {
  date: string
  close?: string | number | null
  adjusted_close?: string | number | null
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Unknown error'
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with status ${response.status}`)
  }

  return response.json()
}

function formatRawReturn(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(1)
}

function modelName(modelName: string) {
  if (modelName.includes('xgboost')) return 'XGBoost'
  if (modelName.includes('lightgbm')) return 'LightGBM'
  if (modelName.includes('catboost')) return 'CatBoost'
  if (modelName.includes('extra_trees')) return 'Extra Trees'
  return modelName
}

function riskLabel(score: number | null | undefined) {
  if (score === null || score === undefined) return '—'
  if (score >= 75) return 'High'
  if (score >= 45) return 'Medium'
  return 'Low'
}

function toNumber(value: string | number | null | undefined) {
  if (value === null || value === undefined) return Number.NaN
  return Number(value)
}

function PriceChart({ prices }: { prices: PricePoint[] }) {
  const points = useMemo(() => {
    return [...prices]
      .map((price) => ({
        date: price.date,
        value: toNumber(price.adjusted_close ?? price.close),
      }))
      .filter((price) => Number.isFinite(price.value))
      .sort((a, b) => a.date.localeCompare(b.date))
  }, [prices])

  if (points.length < 2) {
    return <div className="empty">Brak danych do wykresu.</div>
  }

  const width = 900
  const height = 260
  const padding = 28

  const values = points.map((point) => point.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const path = points
    .map((point, index) => {
      const x = padding + (index / (points.length - 1)) * (width - padding * 2)
      const y = height - padding - ((point.value - min) / range) * (height - padding * 2)
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')

  const first = points[0]
  const last = points[points.length - 1]
  const change = (last.value / first.value - 1) * 100

  return (
    <div className="chart-box">
      <div className="chart-meta">
        <span>{first.date}</span>
        <strong className={change >= 0 ? 'green' : 'red'}>
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}%
        </strong>
        <span>{last.date}</span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        <path d={path} />
      </svg>
    </div>
  )
}

function App() {
  const [horizon, setHorizon] = useState(30)
  const [opportunities, setOpportunities] = useState<Opportunity[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState('AMD')
  const [symbolInput, setSymbolInput] = useState('AMD')
  const [analysis, setAnalysis] = useState<CompanyAnalysis | null>(null)
  const [prices, setPrices] = useState<PricePoint[]>([])
  const [error, setError] = useState<string | null>(null)

  const [journalDecision, setJournalDecision] = useState('watch')
  const [journalTitle, setJournalTitle] = useState('')
  const [journalThesis, setJournalThesis] = useState('')
  const [journalPlan, setJournalPlan] = useState('')

  useEffect(() => {
    setError(null)

    fetchJson<Opportunity[]>(`/api/dashboard/top-opportunities?horizon_days=${horizon}&limit=10`)
      .then(setOpportunities)
      .catch((err) => setError(getErrorMessage(err)))
  }, [horizon])

  useEffect(() => {
    setError(null)

    fetchJson<CompanyAnalysis>(`/api/companies/${selectedSymbol}/analysis`)
      .then(setAnalysis)
      .catch((err) => setError(getErrorMessage(err)))

    fetchJson<PricePoint[]>(`/api/assets/${selectedSymbol}/prices?limit=180`)
      .then(setPrices)
      .catch(() => setPrices([]))
  }, [selectedSymbol])

  function analyzeSymbol(event: FormEvent) {
    event.preventDefault()

    const cleanSymbol = symbolInput.trim().toUpperCase()
    if (!cleanSymbol) return

    setSelectedSymbol(cleanSymbol)
  }

  async function createJournalEntry(event: FormEvent) {
    event.preventDefault()

    if (!analysis || !journalTitle.trim()) return

    await fetchJson('/api/journal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol: analysis.overview.symbol,
        horizon_days: horizon,
        decision: journalDecision,
        status: 'open',
        title: journalTitle,
        thesis: journalThesis || null,
        plan: journalPlan || null,
        emotion: 'neutral',
        confidence: 7,
        tags: ['frontend-note'],
      }),
    })

    setJournalTitle('')
    setJournalThesis('')
    setJournalPlan('')

    const refreshed = await fetchJson<CompanyAnalysis>(
      `/api/companies/${analysis.overview.symbol}/analysis`,
    )

    setAnalysis(refreshed)
  }

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>MarketCopilotex</h1>
          <p>Prosty panel do analizy spółek, predykcji modeli i własnego journalu.</p>
        </div>

        <div className="top-status">Backend online</div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="card">
        <div className="section-head">
          <div>
            <h2>Ranking okazji</h2>
            <p>Wybierz horyzont. Ranking używa najlepszego modelu dla danego timeline’u.</p>
          </div>

          <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
            {HORIZONS.map((item) => (
              <option key={item} value={item}>
                {item}D
              </option>
            ))}
          </select>
        </div>

        <div className="tabs">
          {HORIZONS.map((item) => (
            <button
              key={item}
              className={item === horizon ? 'active' : ''}
              onClick={() => setHorizon(item)}
            >
              {item}D
            </button>
          ))}
        </div>

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Symbol</th>
                <th>Nazwa</th>
                <th>Forecast</th>
                <th>Score</th>
                <th>Risk</th>
                <th>Model</th>
                <th>Sektor</th>
              </tr>
            </thead>

            <tbody>
              {opportunities.map((row) => (
                <tr
                  key={`${row.symbol}-${row.horizon_days}`}
                  onClick={() => {
                    setSelectedSymbol(row.symbol)
                    setSymbolInput(row.symbol)
                  }}
                >
                  <td>#{row.prediction_rank}</td>
                  <td>
                    <strong>{row.symbol}</strong>
                  </td>
                  <td>{row.name}</td>
                  <td className={row.predicted_return && row.predicted_return >= 0 ? 'green' : 'red'}>
                    {formatRawReturn(row.predicted_return)}
                  </td>
                  <td>{formatScore(row.final_score)}</td>
                  <td>{riskLabel(row.risk_score)}</td>
                  <td>{modelName(row.model_name)}</td>
                  <td>{row.sector ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="layout">
        <main className="card">
          <div className="section-head">
            <div>
              <h2>Company Analyzer</h2>
              <p>Wpisz ticker albo kliknij spółkę z rankingu.</p>
            </div>

            <form className="search" onSubmit={analyzeSymbol}>
              <input
                value={symbolInput}
                onChange={(event) => setSymbolInput(event.target.value)}
                placeholder="AMD"
              />
              <button type="submit">Analyze</button>
            </form>
          </div>

          {analysis && (
            <>
              <div className="company-title">
                <div>
                  <h3>
                    {analysis.overview.symbol} — {analysis.overview.name}
                  </h3>
                  <p>
                    {analysis.overview.sector ?? '—'} / {analysis.overview.industry ?? '—'}
                  </p>
                </div>

                <div className="price-box">
                  <span>Latest close</span>
                  <strong>
                    {analysis.overview.latest_price?.close ??
                      analysis.overview.latest_price?.adjusted_close ??
                      '—'}
                  </strong>
                  <small>{analysis.overview.latest_price?.date ?? '—'}</small>
                </div>
              </div>

              <PriceChart prices={prices} />

              <h3 className="subheading">Predykcje dla wszystkich horyzontów</h3>

              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Horyzont</th>
                      <th>Forecast</th>
                      <th>Rank</th>
                      <th>Score</th>
                      <th>Risk</th>
                      <th>Model</th>
                    </tr>
                  </thead>

                  <tbody>
                    {analysis.predictions.map((prediction) => (
                      <tr key={prediction.horizon_days}>
                        <td>
                          <strong>{prediction.horizon_label}</strong>
                        </td>
                        <td
                          className={
                            prediction.predicted_return_percent &&
                            prediction.predicted_return_percent >= 0
                              ? 'green'
                              : 'red'
                          }
                        >
                          {formatPercent(prediction.predicted_return_percent)}
                        </td>
                        <td>#{prediction.prediction_rank ?? '—'}</td>
                        <td>{formatScore(prediction.final_score)}</td>
                        <td>{riskLabel(prediction.risk_score)}</td>
                        <td>{prediction.model_display_name}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </main>

        <aside className="card">
          <h2>Journal</h2>
          <p className="small">Twoje notatki do aktualnie wybranej spółki.</p>

          <form className="journal-form" onSubmit={createJournalEntry}>
            <label>
              Decyzja
              <select
                value={journalDecision}
                onChange={(event) => setJournalDecision(event.target.value)}
              >
                <option value="watch">Watch</option>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
                <option value="avoid">Avoid</option>
              </select>
            </label>

            <label>
              Tytuł
              <input
                value={journalTitle}
                onChange={(event) => setJournalTitle(event.target.value)}
                placeholder="Np. Mocny setup na 30D"
              />
            </label>

            <label>
              Teza
              <textarea
                value={journalThesis}
                onChange={(event) => setJournalThesis(event.target.value)}
                placeholder="Dlaczego warto obserwować?"
              />
            </label>

            <label>
              Plan
              <textarea
                value={journalPlan}
                onChange={(event) => setJournalPlan(event.target.value)}
                placeholder="Co musiałoby się stać?"
              />
            </label>

            <button type="submit">Dodaj notatkę</button>
          </form>

          <div className="journal-list">
            {analysis?.journal.entries.length ? (
              analysis.journal.entries.map((entry) => (
                <article key={entry.id} className="journal-entry">
                  <div className="journal-entry-head">
                    <strong>{entry.title}</strong>
                    <span>{entry.decision.toUpperCase()}</span>
                  </div>

                  <p>{entry.thesis ?? 'Brak tezy.'}</p>

                  {entry.plan && <small>{entry.plan}</small>}
                </article>
              ))
            ) : (
              <div className="empty">Brak notatek.</div>
            )}
          </div>
        </aside>
      </section>

      <section className="card">
        <h2>News AI</h2>
        <p className="small">
          Ten moduł dodamy później: newsy, sentyment, streszczenia AI i wpływ wydarzeń na spółkę.
        </p>
      </section>
    </div>
  )
}

export default App
