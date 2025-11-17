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

const SIGNALS_API_URL = import.meta.env.VITE_SIGNALS_API_URL as string | undefined
const DEBUG_API_URL = import.meta.env.VITE_DEBUG_API_URL as string | undefined

type SrLevels = {
  support: number[]
  resistance: number[]
}

export default function App() {
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
        url.searchParams.set('instrument', 'BZ=F')
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
  }, [])

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
        url.searchParams.set('instrument', 'BZ=F')
        url.searchParams.set('limit', '1') // wystarczy ostatni snapshot

        const res = await fetch(url.toString(), { signal: abort.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const json = await res.json()
        const items = json.items ?? []
        if (!items.length) return

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
  }, [])

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

    // jeśli plugin markerów nie jest gotowy – nic nie robimy
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

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '320px minmax(0, 1fr) 360px',
          gap: 16,
          alignItems: 'flex-start',
          marginTop: 12,
        }}
      >
        {/* LEWA KOLUMNA – OPIS STRATEGII */}
        <aside
          style={{
            fontSize: 12,
            lineHeight: 1.5,
          }}
        >
          <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>
            Jak generujemy sygnał?
          </h2>
          <ol style={{ paddingLeft: 18, margin: 0 }}>
            <li style={{ marginBottom: 6 }}>
              <strong>Bias dzienny (1D)</strong> – liczymy EMA20, EMA50 i RSI14 na{' '}
              przedostatniej świecy dziennej (ostatnia może być w trakcie).
              <ul style={{ paddingLeft: 16, marginTop: 4 }}>
                <li>LONG: EMA20 &gt; EMA50 i RSI &gt; 50</li>
                <li>SHORT: EMA20 &lt; EMA50 i RSI &lt; 50</li>
                <li>inaczej: BIAS = NONE → brak dalszych kroków</li>
              </ul>
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Strefy S/R z H1</strong> – na 1H szukamy swing high/low w
              ostatnich świecach, klastrujemy poziomy i tworzymy strefy wsparcia i
              oporu.
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Filtr na 1H (setup)</strong> – na przedostatniej świecy 1H:
              <ul style={{ paddingLeft: 16, marginTop: 4 }}>
                <li>sprawdzamy EMA (dynamiczna długość), RSI14 oraz strukturę HL/LH</li>
                <li>
                  dla LONG: wystarczy, że spełniony jest którykolwiek z warunków
                  (cena &gt; EMA lub RSI &gt; 50 lub higher low)
                </li>
                <li>
                  dla SHORT: analogicznie (cena &lt; EMA lub RSI &lt; 50 lub lower
                  high)
                </li>
                <li>jeśli warunki nie są spełnione → „1H conditions not met”</li>
              </ul>
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Trigger na 5M</strong> – patrzymy na ostatnią świecę 5m:
              <ul style={{ paddingLeft: 16, marginTop: 4 }}>
                <li>liczymy RSI14, ATR14, EMA20 i EMA50 na 5m</li>
                <li>
                  LONG:
                  <ul style={{ paddingLeft: 16, marginTop: 2 }}>
                    <li>RSI przechodzi z &lt; 40 do &gt; 40</li>
                    <li>wybicie powyżej local high z ostatnich świec</li>
                    <li>cena powyżej EMA20 i EMA50</li>
                    <li>
                      cena w zasięgu ≤ 1× ATR od najbliższej strefy wsparcia z H1
                    </li>
                  </ul>
                </li>
                <li>
                  SHORT:
                  <ul style={{ paddingLeft: 16, marginTop: 2 }}>
                    <li>RSI przechodzi z &gt; 60 do &lt; 60</li>
                    <li>wybicie poniżej local low z ostatnich świec</li>
                    <li>cena poniżej EMA20 i EMA50</li>
                    <li>
                      cena w zasięgu ≤ 1× ATR od najbliższej strefy oporu z H1
                    </li>
                  </ul>
                </li>
                <li>
                  jeśli któryś warunek nie jest spełniony → brak sygnału i w tabeli
                  zobaczysz np.:
                  <br />
                  <code>No 5M trigger: short_rsi_pattern_not_met</code>
                </li>
              </ul>
            </li>
            <li style={{ marginBottom: 6 }}>
              <strong>Risk / RR / score</strong> – jeśli warunki na 5M są spełnione:
              <ul style={{ paddingLeft: 16, marginTop: 4 }}>
                <li>SL ustawiany w odległości 1× ATR</li>
                <li>TP w odległości 2× ATR (RR 1:2)</li>
                <li>liczony jest score (bazowo ~60 + bonus za S/R i RR)</li>
                <li>
                  sygnał trafia do tabeli sygnałów tylko jeśli score ≥ 70, inaczej
                  „Signal score too low.”
                </li>
              </ul>
            </li>
          </ol>

          <p style={{ marginTop: 8, fontSize: 11, color: '#6b7280' }}>
            Środkowa część pokazuje wykres i debug, prawa kolumna – wygenerowane
            sygnały. Na wykresie:
            <br />
            • zielone poziome linie – wsparcia z H1,
            <br />
            • czerwone poziome linie – opory z H1,
            <br />
            • strzałki B/S – wejścia z 5M.
          </p>
        </aside>

        {/* ŚRODKOWA KOLUMNA – WYKRES + TABELA DEBUG */}
        <div style={{ minWidth: 0 }}>
          <div className="card" style={{ padding: 12, marginBottom: 12 }}>
            <div ref={wrapRef} style={{ width: '100%', height: 440 }} />
          </div>

          <div className="card" style={{ padding: 12 }}>
            <DebugTable instrument="BZ=F" limit={30} />
          </div>
        </div>

        {/* PRAWA KOLUMNA – SYGNAŁY */}
        <div
          className="card"
          style={{
            padding: 12,
          }}
        >
          <SignalsTable instrument="BZ=F" limit={30} />
        </div>
      </div>

      <p style={{ marginTop: 12 }}>
        <small className="muted">
          Dane z Twojego API (Lambda → DynamoDB). Świece OHLC + linia (Close) dla
          wybranego interwału, strefy S/R z H1 i sygnały 5M naniesione na wykres.
        </small>
      </p>
    </div>
  )
}
