import  { useEffect, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  ColorType,
  type ISeriesApi,
  type Time
} from 'lightweight-charts'

import { transformApiData } from './transform'
import type { Candle } from './transform'

const API_URL = import.meta.env.VITE_API_URL as string
const DEFAULT_TF = (import.meta.env.VITE_DEFAULT_TF as string) || '5m'
const DEFAULT_LIMIT = Number(import.meta.env.VITE_DEFAULT_LIMIT || 300)

export default function App() {
  const [tf, setTf] = useState(DEFAULT_TF)
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [rows, setRows] = useState<Candle[]>([])

  // Fetch
  useEffect(() => {
    if (!API_URL) {
      setError('Brak VITE_API_URL – ustaw w .env')
      return
    }
    const abort = new AbortController()
    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const url = new URL(API_URL)
        url.searchParams.set('tf', tf)
        url.searchParams.set('limit', String(limit))
        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setRows(transformApiData(data))
      } catch (e: any) {
        if (e?.name !== 'AbortError') setError(String(e?.message || e))
      } finally {
        setLoading(false)
      }
    })()
    return () => abort.abort()
  }, [tf, limit])

  // Chart
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null)

  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const lineSeriesRef   = useRef<ISeriesApi<'Line'> | null>(null)

  useEffect(() => {
    if (!wrapRef.current) return

    const chart = createChart(wrapRef.current, {
      width: wrapRef.current.clientWidth,
      height: 440,
      crosshair: { mode: CrosshairMode.Normal },
      layout: { background: { type: ColorType.Solid, color: '#fff' }, textColor: '#0f172a' },
      grid: { vertLines: { color: '#e2e8f0' }, horzLines: { color: '#e2e8f0' } },
      rightPriceScale: { borderVisible: false },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: true,
      },
    })

    chartRef.current = chart

    // Najpierw linia (będzie pod świeczkami)
    const lineSeries = chart.addSeries(LineSeries, {
      lineWidth: 1,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })
    lineSeriesRef.current = lineSeries

    // Potem świeczki (będą na wierzchu)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#16a34a',
      downColor: '#dc2626',
      borderVisible: true,
      wickUpColor: '#090a0aff',
      wickDownColor: '#dc2626',
    })
    candleSeriesRef.current = candleSeries

    const onResize = () => {
      if (!wrapRef.current) return
      chart.applyOptions({ width: wrapRef.current.clientWidth })
    }

    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      lineSeriesRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!rows.length || !candleSeriesRef.current || !lineSeriesRef.current) return

    const candleData = rows.map(r => ({
      time: Math.floor(r.t.getTime() / 1000) as Time,
      open: r.o,
      high: r.h,
      low: r.l,
      close: r.c,
    }))

    const lineData = rows.map(r => ({
      time: Math.floor(r.t.getTime() / 1000) as Time,
      value: r.c, // linia po cenie zamknięcia
    }))

  candleSeriesRef.current?.setData(candleData)
  lineSeriesRef.current?.setData(lineData)
  }, [rows, tf])

  return (
    <div className="container">
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
        }}
      >
        <h1 style={{ fontSize: 24, fontWeight: 700 }}>SignalForgeWeb</h1>
        <small className="muted">React + Vite · Candlesticks + Line</small>
      </header>

      <div className="row" style={{ marginBottom: 12 }}>
        <div>
          <label>Interwał</label>
          <select value={tf} onChange={e => setTf(e.target.value)}>
            <option value="5m">5m</option>
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
          </select>
        </div>
        <div>
          <label>Ilość świec</label>
          <input
            type="number"
            min={50}
            max={2000}
            step={50}
            value={limit}
            onChange={e => setLimit(Number(e.target.value))}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'end' }}>
          {loading ? (
            <small>Ładowanie…</small>
          ) : (
            <small className="muted">{rows.length} świec</small>
          )}
        </div>
      </div>

      {error && (
        <div
          className="card"
          style={{
            padding: 12,
            borderColor: '#fecaca',
            background: '#fef2f2',
            color: '#991b1b',
            marginBottom: 12,
          }}
        >
          <small>{error}</small>
        </div>
      )}

      <div className="card" style={{ padding: 12 }}>
        <div ref={wrapRef} style={{ width: '100%', height: 440 }} />
      </div>

      <p>
        <small className="muted">
          Dane z Twojego API (Lambda → DynamoDB). Świece OHLC + linia (Close) dla
          wybranego interwału.
        </small>
      </p>
    </div>
  )
}
