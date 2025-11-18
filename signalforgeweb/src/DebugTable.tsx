import { useEffect, useState } from 'react'
import type React from 'react'

const DEBUG_API_URL = import.meta.env.VITE_DEBUG_API_URL as string | undefined

export type DebugRow = {
  ts: string
  strategy_name?: string
  reason?: string
  stage?: string
}

interface Props {
  instrument?: string
  limit?: number
  // jeśli kiedyś chcesz filtrować po strategii:
  // strategyName?: string
}

export function DebugTable({ instrument, limit = 30 }: Props) {
  const [rows, setRows] = useState<DebugRow[]>([])
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
        // jeśli dodasz filtrowanie po strategii:
        // if (strategyName) url.searchParams.set('strategy_name', strategyName)

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items: DebugRow[] = json.items ?? []

        // sort malejąco po ts (na wszelki wypadek, nawet jeśli backend już sortuje)
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
                <th style={thStyle}>Strategia</th>
                <th style={thStyle}>Stage</th>
                <th style={thStyle}>Powód (reason)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const runDate = new Date(r.ts)

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
                    <td style={tdStyle}>{r.strategy_name || '-'}</td>
                    <td style={tdStyle}>{r.stage || '-'}</td>
                    <td style={{ ...tdStyle, maxWidth: 260 }}>
                      <span title={r.reason || ''}>
                        {truncate(r.reason || '', 80)}
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
