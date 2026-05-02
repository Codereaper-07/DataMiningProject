import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function App() {
  const API_BASE = 'http://127.0.0.1:5001'

  const CITY_OPTIONS = useMemo(
    () => [
      'Delhi',
      'Mumbai',
      'Chennai',
      'Bengaluru',
      'Kolkata',
      'Hyderabad',
      'Pune',
      'Ahmedabad',
      'Jaipur',
      'Lucknow',
    ],
    [],
  )

  const [selectedCity, setSelectedCity] = useState('Delhi')

  const [aqiData, setAqiData] = useState(null)
  const [history, setHistory] = useState([])
  const [topCities, setTopCities] = useState([])
  const [modelMetrics, setModelMetrics] = useState([])

  const [loading, setLoading] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [loadingTop, setLoadingTop] = useState(false)
  const [loadingModels, setLoadingModels] = useState(false)
  const [error, setError] = useState('')

  const aqiColor = useMemo(() => {
    const aqi = Number(aqiData?.AQI)
    if (!Number.isFinite(aqi)) return '#94a3b8'
    if (aqi < 50) return '#22c55e' // green
    if (aqi < 100) return '#eab308' // yellow
    if (aqi <= 150) return '#f97316' // orange
    return '#ef4444' // red
  }, [aqiData])

  const aqiLabel = useMemo(() => {
    const aqi = Number(aqiData?.AQI)
    if (!Number.isFinite(aqi)) return '—'
    if (aqi < 50) return 'Good'
    if (aqi < 100) return 'Moderate'
    if (aqi <= 150) return 'Unhealthy for Sensitive Groups'
    return 'Unhealthy'
  }, [aqiData])

  const chartData = useMemo(() => {
    // history records are expected to include datetime and PM2.5
    const rows = Array.isArray(history) ? history : []
    const parsed = rows
      .map((r) => {
        const t = r?.datetime ? new Date(r.datetime) : null
        const pm25 = Number(r?.['PM2.5'])
        return {
          time: t && !Number.isNaN(t.getTime()) ? t : null,
          timeLabel:
            t && !Number.isNaN(t.getTime())
              ? t.toLocaleString([], {
                  month: 'short',
                  day: '2-digit',
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : '',
          pm25: Number.isFinite(pm25) ? pm25 : null,
        }
      })
      .filter((x) => x.time && x.pm25 !== null)
      .sort((a, b) => a.time.getTime() - b.time.getTime())

    return parsed.map((x) => ({ timeLabel: x.timeLabel, pm25: x.pm25 }))
  }, [history])

  async function fetchHistory(city) {
    setLoadingHistory(true)
    setError('')
    try {
      const res = await axios.get(`${API_BASE}/history`, { params: { city } })
      const records = res?.data?.records || []
      setHistory(records)
    } catch (e) {
      setHistory([])
      setError(e?.response?.data?.error || e?.message || 'Failed to fetch history.')
    } finally {
      setLoadingHistory(false)
    }
  }

  async function fetchTopPolluted() {
    setLoadingTop(true)
    setError('')
    try {
      const res = await axios.get(`${API_BASE}/top-polluted`)
      setTopCities(res?.data?.top_5 || [])
    } catch (e) {
      setTopCities([])
      setError(
        e?.response?.data?.error ||
          e?.message ||
          'Failed to fetch top polluted cities.',
      )
    } finally {
      setLoadingTop(false)
    }
  }

  async function fetchModelComparison() {
    setLoadingModels(true)
    setError('')
    try {
      const res = await axios.get(`${API_BASE}/model-comparison`)
      setModelMetrics(res?.data?.metrics || [])
    } catch (e) {
      setModelMetrics([])
      setError(
        e?.response?.data?.error ||
          e?.message ||
          'Failed to fetch model comparison.',
      )
    } finally {
      setLoadingModels(false)
    }
  }

  async function getAQI() {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/predict`, { city: selectedCity })
      setAqiData(res.data)
      // refresh chart data after a prediction (same selected city)
      await fetchHistory(selectedCity)
    } catch (e) {
      setAqiData(null)
      setError(
        e?.response?.data?.error ||
          e?.response?.data?.details ||
          e?.message ||
          'Prediction failed.',
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTopPolluted()
    fetchModelComparison()
  }, [])

  useEffect(() => {
    fetchHistory(selectedCity)
  }, [selectedCity])

  const styles = useMemo(
    () => ({
      page: {
        minHeight: '100vh',
        background:
          'radial-gradient(1200px 600px at 10% 10%, rgba(59,130,246,0.35), transparent 60%), radial-gradient(1000px 600px at 90% 20%, rgba(168,85,247,0.35), transparent 55%), linear-gradient(135deg, #0b1020 0%, #22113a 55%, #0b1020 100%)',
        padding: '32px 16px',
        color: '#e5e7eb',
        fontFamily:
          'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"',
      },
      container: {
        maxWidth: 1200,
        margin: '0 auto',
      },
      headerCard: {
        background: 'rgba(255,255,255,0.06)',
        border: '1px solid rgba(255,255,255,0.10)',
        boxShadow: '0 20px 60px rgba(0,0,0,0.35)',
        borderRadius: 20,
        padding: 22,
        backdropFilter: 'blur(10px)',
        textAlign: 'center',
      },
      title: {
        fontSize: 28,
        fontWeight: 800,
        letterSpacing: '-0.02em',
        margin: 0,
      },
      subtitle: {
        marginTop: 8,
        marginBottom: 0,
        color: 'rgba(229,231,235,0.75)',
        fontSize: 14,
      },
      grid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(12, minmax(0, 1fr))',
        gap: 16,
        marginTop: 16,
      },
      card: {
        background: 'rgba(255,255,255,0.06)',
        border: '1px solid rgba(255,255,255,0.10)',
        boxShadow: '0 18px 50px rgba(0,0,0,0.28)',
        borderRadius: 20,
        padding: 18,
        backdropFilter: 'blur(10px)',
      },
      label: { fontSize: 12, color: 'rgba(229,231,235,0.7)' },
      controlRow: {
        display: 'flex',
        gap: 12,
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: 16,
      },
      select: {
        background: 'rgba(15,23,42,0.65)',
        color: '#e5e7eb',
        border: '1px solid rgba(255,255,255,0.14)',
        borderRadius: 12,
        padding: '10px 12px',
        minWidth: 240,
        outline: 'none',
      },
      button: {
        border: 'none',
        borderRadius: 12,
        padding: '10px 14px',
        color: '#071018',
        fontWeight: 800,
        cursor: 'pointer',
        background:
          'linear-gradient(135deg, #22c55e 0%, #38bdf8 45%, #a855f7 100%)',
        boxShadow: '0 14px 30px rgba(0,0,0,0.35)',
        transition: 'transform 180ms ease, filter 180ms ease',
      },
      buttonDisabled: {
        opacity: 0.55,
        cursor: 'not-allowed',
        filter: 'grayscale(20%)',
      },
      row: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
      },
      badge: (bg) => ({
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 10px',
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 700,
        background: bg,
        border: '1px solid rgba(255,255,255,0.14)',
        color: '#e5e7eb',
      }),
      table: {
        width: '100%',
        borderCollapse: 'separate',
        borderSpacing: 0,
        overflow: 'hidden',
        borderRadius: 14,
        border: '1px solid rgba(255,255,255,0.10)',
      },
      th: {
        textAlign: 'left',
        fontSize: 12,
        color: 'rgba(229,231,235,0.75)',
        padding: '12px 12px',
        background: 'rgba(15,23,42,0.55)',
      },
      td: {
        padding: '12px 12px',
        borderTop: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(255,255,255,0.04)',
        fontSize: 13,
      },
      error: {
        marginTop: 14,
        padding: '10px 12px',
        borderRadius: 14,
        border: '1px solid rgba(239,68,68,0.35)',
        background: 'rgba(239,68,68,0.10)',
        color: '#fecaca',
      },
    }),
    [],
  )

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.headerCard}>
          <h1 style={styles.title}>Air Quality &amp; Climate Dashboard</h1>
          <p style={styles.subtitle}>
            Real-time AQI prediction, health advisory, climate impact, and PM2.5
            trends
          </p>

          <div style={styles.controlRow}>
            <select
              value={selectedCity}
              onChange={(e) => setSelectedCity(e.target.value)}
              style={styles.select}
            >
              {CITY_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={getAQI}
              disabled={loading}
              style={{
                ...styles.button,
                ...(loading ? styles.buttonDisabled : null),
              }}
              onMouseEnter={(e) => {
                if (loading) return
                e.currentTarget.style.transform = 'translateY(-1px)'
                e.currentTarget.style.filter = 'brightness(1.05)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0px)'
                e.currentTarget.style.filter = 'brightness(1)'
              }}
            >
              {loading ? 'Loading...' : 'Get AQI'}
            </button>
          </div>

          {error ? <div style={styles.error}>{error}</div> : null}
        </div>

        <div style={styles.grid}>
          {/* AQI Card */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div style={styles.row}>
              <div>
                <div style={styles.label}>Predicted AQI</div>
                <div
                  style={{
                    fontSize: 54,
                    fontWeight: 900,
                    lineHeight: 1,
                    marginTop: 8,
                    color: aqiColor,
                    textShadow: '0 12px 40px rgba(0,0,0,0.35)',
                  }}
                >
                  {Number.isFinite(Number(aqiData?.AQI))
                    ? Math.round(aqiData.AQI)
                    : '—'}
                </div>
                <div style={{ marginTop: 8, color: 'rgba(229,231,235,0.85)' }}>
                  <span style={{ fontWeight: 800 }}>
                    {aqiData?.city || selectedCity}
                  </span>
                  <span style={{ opacity: 0.7 }}> • </span>
                  <span style={{ opacity: 0.9 }}>{aqiData?.region || '—'}</span>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={styles.label}>AQI Category</div>
                <div style={{ marginTop: 10 }}>
                  <span style={styles.badge('rgba(2,6,23,0.45)')}>
                    {aqiLabel}
                  </span>
                </div>
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 12,
                    color: 'rgba(229,231,235,0.70)',
                  }}
                >
                  Color coding: Green &lt;50, Yellow 50–100, Orange 100–150, Red
                  &gt;150
                </div>
              </div>
            </div>
          </div>

          {/* Health Advisory */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div>
                <div style={styles.label}>Health Advisory</div>
                <div style={{ fontSize: 18, fontWeight: 900, marginTop: 6 }}>
                  Recommended Actions
                </div>
              </div>
              <span style={styles.badge('rgba(34,197,94,0.15)')}>Be safe</span>
            </div>

            <div style={{ marginTop: 12 }}>
              {loading ? (
                <div style={{ color: 'rgba(229,231,235,0.75)' }}>Loading...</div>
              ) : (
                <ul
                  style={{
                    margin: 0,
                    paddingLeft: 18,
                    color: 'rgba(229,231,235,0.88)',
                  }}
                >
                  {(aqiData?.health_advice?.actions || []).length ? (
                    aqiData.health_advice.actions.map((a, idx) => (
                      <li
                        key={`${a}-${idx}`}
                        style={{ marginBottom: 8, lineHeight: 1.4 }}
                      >
                        {a}
                      </li>
                    ))
                  ) : (
                    <li style={{ opacity: 0.75 }}>
                      Click “Get AQI” to fetch real-time advisory.
                    </li>
                  )}
                </ul>
              )}
            </div>
          </div>

          {/* Climate Impact */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div>
                <div style={styles.label}>Climate Section</div>
                <div style={{ fontSize: 18, fontWeight: 900, marginTop: 6 }}>
                  Climate Impact & Emissions
                </div>
              </div>
              <span
                style={styles.badge(
                  aqiData?.climate_impact === 'High'
                    ? 'rgba(239,68,68,0.16)'
                    : aqiData?.climate_impact === 'Moderate'
                      ? 'rgba(234,179,8,0.16)'
                      : 'rgba(34,197,94,0.16)',
                )}
              >
                {aqiData?.climate_impact || '—'}
              </span>
            </div>

            <div
              style={{ marginTop: 12, display: 'flex', gap: 14, flexWrap: 'wrap' }}
            >
              <div
                style={{
                  flex: '1 1 220px',
                  padding: 12,
                  borderRadius: 16,
                  background: 'rgba(15,23,42,0.45)',
                  border: '1px solid rgba(255,255,255,0.10)',
                }}
              >
                <div style={styles.label}>Emission Estimate</div>
                <div style={{ fontSize: 22, fontWeight: 900, marginTop: 6 }}>
                  {Number.isFinite(Number(aqiData?.emission_estimate))
                    ? Number(aqiData.emission_estimate).toFixed(2)
                    : '—'}
                </div>
                <div
                  style={{
                    marginTop: 6,
                    fontSize: 12,
                    color: 'rgba(229,231,235,0.70)',
                  }}
                >
                  Proxy based on CO &amp; NO2 (for awareness/demo).
                </div>
              </div>

              <div
                style={{
                  flex: '2 1 320px',
                  padding: 12,
                  borderRadius: 16,
                  background: 'rgba(15,23,42,0.45)',
                  border: '1px solid rgba(255,255,255,0.10)',
                }}
              >
                <div style={styles.label}>Climate Awareness Tip</div>
                <div style={{ marginTop: 8, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <span style={styles.badge('rgba(56,189,248,0.14)')}>
                    Reduce vehicle usage
                  </span>
                  <span style={styles.badge('rgba(168,85,247,0.14)')}>
                    Use public transport
                  </span>
                  <span style={styles.badge('rgba(34,197,94,0.14)')}>Plant trees</span>
                </div>
              </div>
            </div>
          </div>

          {/* Top Polluted */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div style={styles.row}>
              <div>
                <div style={styles.label}>Top Polluted Cities</div>
                <div style={{ fontSize: 18, fontWeight: 900, marginTop: 6 }}>
                  Top 5 (Predicted AQI)
                </div>
              </div>
              <button
                type="button"
                onClick={fetchTopPolluted}
                disabled={loadingTop}
                style={{
                  ...styles.button,
                  padding: '8px 12px',
                  ...(loadingTop ? styles.buttonDisabled : null),
                }}
              >
                {loadingTop ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            <div
              style={{
                marginTop: 12,
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: 12,
              }}
            >
              {(topCities || []).length ? (
                topCities.map((c) => (
                  <div
                    key={`${c.city}-${c.region}`}
                    style={{
                      padding: 12,
                      borderRadius: 16,
                      background: 'rgba(15,23,42,0.45)',
                      border: '1px solid rgba(255,255,255,0.10)',
                      transition: 'transform 180ms ease',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'baseline',
                        justifyContent: 'space-between',
                        gap: 10,
                      }}
                    >
                      <div style={{ fontWeight: 900, fontSize: 16 }}>{c.city}</div>
                      <div style={{ fontWeight: 900, color: 'rgba(229,231,235,0.85)' }}>
                        {Number.isFinite(Number(c.AQI)) ? Math.round(c.AQI) : '—'}
                      </div>
                    </div>
                    <div
                      style={{
                        marginTop: 6,
                        color: 'rgba(229,231,235,0.70)',
                        fontSize: 12,
                      }}
                    >
                      {c.region}
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ color: 'rgba(229,231,235,0.75)' }}>
                  {loadingTop
                    ? 'Loading...'
                    : 'No data yet. Start the backend and refresh.'}
                </div>
              )}
            </div>
          </div>

          {/* History Chart */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div style={styles.row}>
              <div>
                <div style={styles.label}>Historical Data Graph</div>
                <div style={{ fontSize: 18, fontWeight: 900, marginTop: 6 }}>
                  PM2.5 vs Time ({selectedCity})
                </div>
              </div>
              <div style={{ color: 'rgba(229,231,235,0.70)', fontSize: 12 }}>
                {loadingHistory ? 'Loading...' : `${chartData.length} points`}
              </div>
            </div>

            <div style={{ marginTop: 12, height: 320, width: '100%' }}>
              {loadingHistory ? (
                <div style={{ color: 'rgba(229,231,235,0.75)' }}>Loading...</div>
              ) : chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
                  >
                    <CartesianGrid
                      stroke="rgba(255,255,255,0.10)"
                      strokeDasharray="4 4"
                    />
                    <XAxis
                      dataKey="timeLabel"
                      tick={{ fill: 'rgba(229,231,235,0.70)', fontSize: 11 }}
                      interval="preserveStartEnd"
                      minTickGap={22}
                      axisLine={{ stroke: 'rgba(255,255,255,0.18)' }}
                      tickLine={{ stroke: 'rgba(255,255,255,0.18)' }}
                    />
                    <YAxis
                      tick={{ fill: 'rgba(229,231,235,0.70)', fontSize: 11 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.18)' }}
                      tickLine={{ stroke: 'rgba(255,255,255,0.18)' }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(2,6,23,0.92)',
                        border: '1px solid rgba(255,255,255,0.14)',
                        borderRadius: 12,
                        color: '#e5e7eb',
                      }}
                      labelStyle={{ color: 'rgba(229,231,235,0.8)' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="pm25"
                      stroke="#38bdf8"
                      strokeWidth={3}
                      dot={false}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: 'rgba(229,231,235,0.75)' }}>
                  No history data for this city yet. Run your fetch script a few
                  times to populate the CSV.
                </div>
              )}
            </div>
          </div>

          {/* Model Comparison */}
          <div style={{ ...styles.card, gridColumn: 'span 12' }}>
            <div style={styles.row}>
              <div>
                <div style={styles.label}>Model Comparison</div>
                <div style={{ fontSize: 18, fontWeight: 900, marginTop: 6 }}>
                  Performance Metrics
                </div>
              </div>
              <button
                type="button"
                onClick={fetchModelComparison}
                disabled={loadingModels}
                style={{
                  ...styles.button,
                  padding: '8px 12px',
                  ...(loadingModels ? styles.buttonDisabled : null),
                }}
              >
                {loadingModels ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            <div style={{ marginTop: 12 }}>
              {loadingModels ? (
                <div style={{ color: 'rgba(229,231,235,0.75)' }}>Loading...</div>
              ) : (
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Model</th>
                      <th style={styles.th}>R2</th>
                      <th style={styles.th}>RMSE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(modelMetrics || []).length ? (
                      modelMetrics.map((m, idx) => (
                        <tr key={`${m.model}-${idx}`}>
                          <td style={styles.td}>{m.model}</td>
                          <td style={styles.td}>
                            {m.r2 === null ||
                            m.r2 === undefined ||
                            Number.isNaN(Number(m.r2))
                              ? 'NA'
                              : Number(m.r2).toFixed(4)}
                          </td>
                          <td style={styles.td}>
                            {m.rmse === null ||
                            m.rmse === undefined ||
                            Number.isNaN(Number(m.rmse))
                              ? 'NA'
                              : Number(m.rmse).toFixed(4)}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td style={styles.td} colSpan={3}>
                          No metrics found. Ensure `model_comparison.csv` exists on
                          the backend.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
