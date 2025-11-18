// src/SignalsTable.tsx
import { useEffect, useState } from 'react'

const SIGNALS_API_URL = import.meta.env.VITE_SIGNALS_API_URL as string | undefined

export type SignalRow = {
  ts: string
  instrument: string
  side: 'BUY' | 'SELL'
  bias: string
  timeframe_entry: string
  entry_price: number
  sl: number
  tp: number
  rr: number
  score: number
  sr_level?: number
  sr_strength?: number
  sr_distance_atr?: number
}

interface Props {
  instrument?: string
  limit?: number
}

export function SignalsTable({ instrument, limit = 30 }: Props) {
  const [rows, setRows] = useState<SignalRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    if (!SIGNALS_API_URL) {
      setError('Brak VITE_SIGNALS_API_URL – ustaw w .env')
      return
    }

    const abort = new AbortController()

    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const url = new URL(SIGNALS_API_URL)
        if (instrument) url.searchParams.set('instrument', instrument)
        if (limit) url.searchParams.set('limit', String(limit))

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items: SignalRow[] = json.items ?? []

        // sort malejąco po ts
        const sorted = items.slice().sort((a, b) => {
          const ta = new Date(a.ts).getTime()
          const tb = new Date(b.ts).getTime()
          return tb - ta
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
        <strong>Ostatnie sygnały</strong>
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
        <small className="muted">Brak zapisanych sygnałów</small>
      )}

      {rows.length > 0 && (
        <div
          style={{
            overflowX: 'auto',
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
                <th style={thStyle}>Czas</th>
                <th style={thStyle}>Strona</th>
                <th style={thStyle}>Bias</th>
                <th style={thStyle}>TF</th>
                <th style={thStyle}>Wejście</th>
                <th style={thStyle}>SL</th>
                <th style={thStyle}>TP</th>
                <th style={thStyle}>RR</th>
                <th style={thStyle}>Score</th>
                <th style={thStyle}>SR poziom</th>
                <th style={thStyle}>SR siła</th>
                <th style={thStyle}>Dist ATR</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const tsDate = new Date(r.ts)
                const isLong = r.side === 'BUY'
                const sideColor = isLong ? '#166534' : '#b91c1c'

                return (
                  <tr
                    key={`${r.ts}-${idx}`}
                    style={idx % 2 ? rowAltStyle : undefined}
                  >
                    <td style={tdStyle}>
                      {tsDate.toLocaleString('pl-PL', {
                        timeZone: 'Europe/Warsaw',
                        hour12: false,
                      })}
                    </td>
                    <td
                      style={{
                        ...tdStyle,
                        color: sideColor,
                        fontWeight: 600,
                      }}
                    >
                      {r.side}
                    </td>
                    <td style={tdStyle}>{r.bias}</td>
                    <td style={tdStyle}>{r.timeframe_entry}</td>
                    <td style={tdStyle}>{r.entry_price.toFixed(2)}</td>
                    <td style={tdStyle}>{r.sl.toFixed(2)}</td>
                    <td style={tdStyle}>{r.tp.toFixed(2)}</td>
                    <td style={tdStyle}>{r.rr.toFixed(2)}</td>
                    <td style={tdStyle}>{r.score.toFixed(0)}</td>
                    <td style={tdStyle}>
                      {r.sr_level != null ? r.sr_level.toFixed(2) : '-'}
                    </td>
                    <td style={tdStyle}>
                      {r.sr_strength != null ? r.sr_strength : '-'}
                    </td>
                    <td style={tdStyle}>
                      {r.sr_distance_atr != null
                        ? r.sr_distance_atr.toFixed(2)
                        : '-'}
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
