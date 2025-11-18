import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  ColorType,
  type ISeriesApi,
  type Time,
  createSeriesMarkers,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
} from 'lightweight-charts'
import { DebugTable } from './DebugTable'
import { SignalsTable, type SignalRow } from './SignalsTable'

import { transformApiData } from './transform'
import type { Candle } from './transform'

const API_URL = import.meta.env.VITE_API_URL as string
const DEFAULT_TF = (import.meta.env.VITE_DEFAULT_TF as string) || '5m'
const DEFAULT_LIMIT = Number(import.meta.env.VITE_DEFAULT_LIMIT || 300)

// nowe: dostępne instrumenty
const INSTRUMENTS = ['BZ=F', 'GC=F'] as const
type Instrument = (typeof INSTRUMENTS)[number]
const DEFAULT_INSTRUMENT: Instrument = 'BZ=F'

const SIGNALS_API_URL = import.meta.env.VITE_SIGNALS_API_URL as string | undefined
const DEBUG_API_URL = import.meta.env.VITE_DEBUG_API_URL as string | undefined

type SrLevels = {
  support: number[]
  resistance: number[]
}

export default function App() {
  const [instrument, setInstrument] = useState<Instrument>(DEFAULT_INSTRUMENT)
  const [tf, setTf] = useState(DEFAULT_TF)
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [rows, setRows] = useState<Candle[]>([])
  const [signals, setSignals] = useState<SignalRow[]>([])
  const [srLevels, setSrLevels] = useState<SrLevels | null>(null)

  // --- sygnały do markerów ---
  useEffect(() => {
    if (!SIGNALS_API_URL) {
      console.warn('Brak VITE_SIGNALS_API_URL – sygnały na wykresie nie będą widoczne')
      return
    }

    const abort = new AbortController()
    ;(async () => {
      try {
        const url = new URL(SIGNALS_API_URL)
        url.searchParams.set('instrument', instrument) // tu było BZ=F
        url.searchParams.set('limit', '200') // np. 200 ostatnich sygnałów

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items: SignalRow[] = json.items ?? []
        setSignals(items)
      } catch (e: any) {
        if (e?.name !== 'AbortError') {
          console.error('Błąd pobierania sygnałów:', e)
        }
      }
    })()

    return () => abort.abort()
  }, [instrument])

  // --- poziomy S/R z debug-API (ostatni snapshot) ---
  useEffect(() => {
    if (!DEBUG_API_URL) {
      console.warn('Brak VITE_DEBUG_API_URL – poziomy S/R na wykresie nie będą widoczne')
      return
    }

    const abort = new AbortController()
    ;(async () => {
      try {
        const url = new URL(DEBUG_API_URL)
        url.searchParams.set('instrument', instrument) // tu było BZ=F
        url.searchParams.set('limit', '1') // wystarczy ostatni snapshot

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items = json.items ?? []
        if (!items.length) {
          setSrLevels(null)
          return
        }

        const last = items[items.length - 1] // przy limit=1 i tak jest jeden
        const sr = last.sr || {}

        setSrLevels({
          support: Array.isArray(sr.support) ? sr.support.filter((x: any) => x != null) : [],
          resistance: Array.isArray(sr.resistance) ? sr.resistance.filter((x: any) => x != null) : [],
        })
      } catch (e: any) {
        if (e?.name !== 'AbortError') {
          console.error('Błąd pobierania SR z debug-API:', e)
        }
      }
    })()

    return () => abort.abort()
  }, [instrument])

  // --- świece / dane główne ---
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
        url.searchParams.set('instrument', instrument) // nowy parametr
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
  }, [tf, limit, instrument])

  // Chart refs
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null)

  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const lineSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const markersPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)

  // price lines S/R
  const supportLinesRef = useRef<any[]>([])
  const resistanceLinesRef = useRef<any[]>([])

  // Init chart
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

    // Plugin markerów powiązany z serią świec
    markersPluginRef.current = createSeriesMarkers<Time>(candleSeries, [])

    const onResize = () => {
      if (!wrapRef.current) return
      chart.applyOptions({ width: wrapRef.current.clientWidth })
    }

    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)

      // wyczyść S/R price lines
      if (candleSeriesRef.current) {
        const seriesAny = candleSeriesRef.current as any
        supportLinesRef.current.forEach(line => seriesAny.removePriceLine(line))
        resistanceLinesRef.current.forEach(line => seriesAny.removePriceLine(line))
      }
      supportLinesRef.current = []
      resistanceLinesRef.current = []

      // wyczyść markery
      if (markersPluginRef.current) {
        markersPluginRef.current.setMarkers([])
      }
      markersPluginRef.current = null

      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      lineSeriesRef.current = null
    }
  }, [])

  // Update chart data + markery sygnałów
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
      value: r.c,
    }))

    candleSeriesRef.current.setData(candleData)
    lineSeriesRef.current.setData(lineData)

    if (!markersPluginRef.current) return

    // MARKERY Z SYGNAŁÓW
    if (signals.length) {
      const candleTimes = new Set<Time>(candleData.map(c => c.time))

      const markers: SeriesMarker<Time>[] = signals
        .map(sig => {
          if (!sig.ts) return null
          const tsSec = Math.floor(new Date(sig.ts).getTime() / 1000) as Time
          if (!candleTimes.has(tsSec)) {
            // sygnał poza zakresem załadowanych świec
            return null
          }

          const isBuy = sig.side === 'BUY'

          return {
            time: tsSec,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: isBuy ? '#16a34a' : '#dc2626',
            shape: isBuy ? 'arrowUp' : 'arrowDown',
            text: isBuy ? 'B' : 'S',
          } as SeriesMarker<Time>
        })
        .filter(Boolean) as SeriesMarker<Time>[]

      markersPluginRef.current.setMarkers(markers)
    } else {
      // brak sygnałów → czyścimy markery
      markersPluginRef.current.setMarkers([])
    }
  }, [rows, tf, signals])

  // Rysowanie poziomów S/R jako priceLines
  useEffect(() => {
    if (!candleSeriesRef.current) return

    const seriesAny = candleSeriesRef.current as any

    // najpierw usuń stare linie
    supportLinesRef.current.forEach(line => seriesAny.removePriceLine(line))
    resistanceLinesRef.current.forEach(line => seriesAny.removePriceLine(line))
    supportLinesRef.current = []
    resistanceLinesRef.current = []

    if (!srLevels) return

    // wsparcia – zielone linie
    srLevels.support.forEach(level => {
      if (level == null) return
      const line = seriesAny.createPriceLine({
        price: level,
        color: '#16a34a',
        lineWidth: 1,
        lineStyle: 2, // dashed
        axisLabelVisible: true,
        title: 'S',
      })
      supportLinesRef.current.push(line)
    })

    // opory – czerwone linie
    srLevels.resistance.forEach(level => {
      if (level == null) return
      const line = seriesAny.createPriceLine({
        price: level,
        color: '#dc2626',
        lineWidth: 1,
        lineStyle: 2, // dashed
        axisLabelVisible: true,
        title: 'R',
      })
      resistanceLinesRef.current.push(line)
    })
  }, [srLevels])

  return (
    <div
      className="container"
      style={{
        maxWidth: 1800,
        margin: '0 auto',
        padding: 16,
      }}
    >
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

      <div className="row" style={{ marginBottom: 12, display: 'flex', gap: 12 }}>
        <div>
          <label>Instrument</label>
          <select
            value={instrument}
            onChange={e => setInstrument(e.target.value as Instrument)}
          >
            <option value="BZ=F">BZ=F (Brent)</option>
            <option value="GC=F">GC=F (Gold)</option>
          </select>
        </div>

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
          {loading ? <small>Ładowanie…</small> : <small className="muted">{rows.length} świec</small>}
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

      <div
        className="layout-main"
        style={{
          display: 'grid',
          gap: 16,
          alignItems: 'flex-start',
          marginTop: 12,
          gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)',
        }}
>
        {/* LEWA KOLUMNA – WYKRES + SYGNAŁY */}
        <div style={{ minWidth: 0 }}>
          <div className="card" style={{ padding: 12, marginBottom: 12 }}>
            <div ref={wrapRef} style={{ width: '100%', height: 440 }} />
          </div>

          <div className="card" style={{ padding: 12 }}>
            <SignalsTable instrument={instrument} limit={30} />
          </div>
        </div>

        {/* PRAWA KOLUMNA – DEBUG */}
        <div
          className="card"
          style={{
            padding: 12,
          }}
        >
          <DebugTable instrument={instrument} limit={30} />
        </div>
      </div>


      <p style={{ marginTop: 12 }}>
        <small className="muted">
          Dane z Twojego API (Lambda → DynamoDB). Świece OHLC + linia (Close) dla
          wybranego interwału, strefy S/R z H1 i sygnały 5M naniesione na wykres dla
          instrumentu {instrument}.
        </small>
      </p>
    </div>
  )
}
