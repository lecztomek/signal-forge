import { useEffect, useState } from 'react'

const DEBUG_API_URL = import.meta.env.VITE_DEBUG_API_URL as string | undefined

type DailyInfo = {
  bias?: string
  close?: number
  ema20?: number
  ema50?: number
  rsi14?: number
}

type H1Info = {
  setup_ok?: boolean
  close?: number
  ema?: number
  rsi14?: number
  last_high?: number
  prev_high?: number
  last_low?: number
  prev_low?: number
} | null

type SrInfo = {
  support: number[]
  resistance: number[]
}

type M5Info = {
  last_close?: number
  last_rsi?: number
  prev_rsi?: number
  atr?: number
  recent_high?: number
  recent_low?: number
} | null

export type DebugSnapshot = {
  ts: string
  instrument: string
  has_signal: boolean
  reason: string
  daily: DailyInfo
  h1: H1Info
  sr: SrInfo
  m5: M5Info
}

interface Props {
  instrument?: string
  limit?: number
}

export function DebugTable({ instrument, limit = 30 }: Props) {
  const [rows, setRows] = useState<DebugSnapshot[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    if (!DEBUG_API_URL) {
      setError('Brak VITE_DEBUG_API_URL – ustaw w .env')
      return
    }

    const abort = new AbortController()

    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const url = new URL(DEBUG_API_URL)
        if (instrument) url.searchParams.set('instrument', instrument)
        if (limit) url.searchParams.set('limit', String(limit))

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items: DebugSnapshot[] = json.items ?? []

        // SORTOWANIE MALEJĄCO po ts
        const sorted = items.slice().sort((a, b) => {
          const ta = new Date(a.ts).getTime()
          const tb = new Date(b.ts).getTime()
          return tb - ta // malejąco
        })

        setRows(sorted)
      } catch (e: any) {
        if (e?.name !== 'AbortError') {
          setError(String(e?.message || e))
        }
      } finally {
        setLoading(false)
      }
    })()

    return () => abort.abort()
  }, [instrument, limit])

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: 8,
          alignItems: 'center',
        }}
      >
        <strong>Ostatnie wywołania strategii</strong>
        {loading && <small>Ładowanie…</small>}
      </div>

      {error && (
        <div
          style={{
            marginBottom: 8,
            padding: 8,
            borderRadius: 4,
            border: '1px solid #fecaca',
            background: '#fef2f2',
            color: '#991b1b',
          }}
        >
          <small>{error}</small>
        </div>
      )}

      {!rows.length && !loading && !error && (
        <small className="muted">
          Brak danych debug (jeszcze nie było wywołań Lambdy?)
        </small>
      )}

      {rows.length > 0 && (
        <div
          style={{
            overflowX: 'auto',
            maxHeight: 320, // ZMNIEJSZONA WYSOKOŚĆ
            overflowY: 'auto',
            border: '1px solid #e5e7eb',
            borderRadius: 4,
          }}
        >
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: 12,
            }}
          >
            <thead>
              <tr>
                <th style={thStyle}>Czas (run)</th>
                <th style={thStyle}>Bias 1D</th>
                <th style={thStyle}>RSI 1D</th>
                <th style={thStyle}>EMA20 / EMA50</th>
                <th style={thStyle}>Setup 1H</th>
                <th style={thStyle}>RSI 1H</th>
                <th style={thStyle}>Cena 1H</th>

                {/* NOWE KOLUMNY 5M */}
                <th style={thStyle}>RSI 5M (prev → last)</th>
                <th style={thStyle}>Cena 5M</th>
                <th style={thStyle}>ATR 5M</th>
                <th style={thStyle}>5M High / Low</th>

                <th style={thStyle}>Sygnał</th>
                <th style={thStyle}>Powód (why)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const d = r.daily || {}
                const h = r.h1 || {}
                const m5 = r.m5 || {}
                const runDate = new Date(r.ts)

                const signalLabel = r.has_signal ? 'TAK' : 'NIE'
                const signalColor = r.has_signal ? '#166534' : '#991b1b'

                return (
                  <tr
                    key={`${r.ts}-${idx}`}
                    style={idx % 2 ? rowAltStyle : undefined}
                  >
                    <td style={tdStyle}>
                      {runDate.toLocaleString('pl-PL', {
                        timeZone: 'Europe/Warsaw',
                        hour12: false,
                      })}
                    </td>
                    <td style={tdStyle}>{d.bias || '-'}</td>
                    <td style={tdStyle}>
                      {d.rsi14 != null ? d.rsi14.toFixed(1) : '-'}
                    </td>
                    <td style={tdStyle}>
                      {d.ema20 != null && d.ema50 != null
                        ? `${d.ema20.toFixed(2)} / ${d.ema50.toFixed(2)}`
                        : '-'}
                    </td>
                    <td style={tdStyle}>
                      {h && typeof h.setup_ok === 'boolean'
                        ? h.setup_ok
                          ? 'OK'
                          : 'NIE'
                        : '-'}
                    </td>
                    <td style={tdStyle}>
                      {h && h.rsi14 != null ? h.rsi14.toFixed(1) : '-'}
                    </td>
                    <td style={tdStyle}>
                      {h && h.close != null ? h.close.toFixed(2) : '-'}
                    </td>

                    {/* NOWE KOMÓRKI 5M */}
                    <td style={tdStyle}>
                      {m5 && (m5.prev_rsi != null || m5.last_rsi != null)
                        ? `${m5.prev_rsi?.toFixed(1) ?? '-'} → ${
                            m5.last_rsi?.toFixed(1) ?? '-'
                          }`
                        : '-'}
                    </td>
                    <td style={tdStyle}>
                      {m5 && m5.last_close != null
                        ? m5.last_close.toFixed(2)
                        : '-'}
                    </td>
                    <td style={tdStyle}>
                      {m5 && m5.atr != null ? m5.atr.toFixed(3) : '-'}
                    </td>
                    <td style={tdStyle}>
                      {m5 &&
                      m5.recent_high != null &&
                      m5.recent_low != null
                        ? `${m5.recent_high.toFixed(2)} / ${m5.recent_low.toFixed(
                            2,
                          )}`
                        : '-'}
                    </td>

                    <td
                      style={{
                        ...tdStyle,
                        color: signalColor,
                        fontWeight: 600,
                      }}
                    >
                      {signalLabel}
                    </td>
                    <td style={{ ...tdStyle, maxWidth: 260 }}>
                      <span title={r.reason}>
                        {truncate(r.reason, 80)}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  borderBottom: '1px solid #e5e7eb',
  padding: '4px 6px',
  fontWeight: 600,
  whiteSpace: 'nowrap',
}

const tdStyle: React.CSSProperties = {
  padding: '3px 6px',
  borderBottom: '1px solid #f3f4f6',
  whiteSpace: 'nowrap',
}

const rowAltStyle: React.CSSProperties = {
  background: '#f9fafb',
}

function truncate(str: string, max: number): string {
  if (!str) return ''
  return str.length > max ? str.slice(0, max - 1) + '…' : str
}
