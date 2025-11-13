export type Candle = {
  t: Date
  o: number
  h: number
  l: number
  c: number
  v: number
}

export function transformApiData(input: any): Candle[] {
  const arr = Array.isArray(input) ? input : input?.items || []

  const candles: Candle[] = []

  for (const x of arr) {
    // wspieramy zarówno `timestamp`, jak i `date`
    const ts = x.timestamp ?? x.date
    const t = new Date(ts)

    const o = Number(x.open)
    const h = Number(x.high)
    const l = Number(x.low)
    const c = Number(x.close)
    const v = Number(x.volume ?? 0)

    // odrzucamy śmieci
    if (
      ![o, h, l, c].every(Number.isFinite) ||
      !(t instanceof Date) ||
      isNaN(t.getTime())
    ) {
      continue
    }

    candles.push({ t, o, h, l, c, v })
  }

  // 1) sort rosnąco po czasie
  candles.sort((a, b) => a.t.getTime() - b.t.getTime())

  // 2) deduplikacja po czasie (ms) – lightweight-charts wymaga strictly increasing
  const dedup: Candle[] = []
  let lastTs: number | null = null

  for (const c of candles) {
    const ts = c.t.getTime()
    if (lastTs === ts) {
      // jeśli chcesz, żeby "nowszy" rekord wygrywał – nadpisujemy poprzedni
      dedup[dedup.length - 1] = c
    } else {
      dedup.push(c)
      lastTs = ts
    }
  }

  return dedup
}

/** Minimal smoke tests – uruchomią się raz w przeglądarce (możesz wyłączyć window.__NO_TESTS__=true) */
function runTests() {
  // T1: parsuje pojedynczy wiersz
  const one = transformApiData([
    { timestamp: '2025-01-01T10:00:00Z', open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 }
  ])
  console.assert(one.length === 1 && one[0].c === 1.5 && one[0].h === 2 && one[0].l === 0.5, 'T1 failed')

  // T2: filtruje zły timestamp
  const empty = transformApiData([{ timestamp: 'bad', close: '1.2', open: 1, high: 2, low: 1 }])
  console.assert(empty.length === 0, 'T2 failed')

  // T3: sortuje rosnąco po czasie
  const sorted = transformApiData([
    { timestamp: '2025-01-01T10:05:00Z', open: 2, high: 3, low: 1.5, close: 2 },
    { timestamp: '2025-01-01T10:00:00Z', open: 1, high: 1.5, low: 0.9, close: 1 }
  ])
  console.assert(sorted.length === 2 && sorted[0].c === 1 && sorted[1].c === 2, 'T3 failed')

  // T4: wspiera payload { items: [...] }
  const wrapped = transformApiData({ items: [{ timestamp: '2025-01-01T10:00:00Z', open: 5, high: 6, low: 4.5, close: 5.5 }] })
  console.assert(wrapped.length === 1 && wrapped[0].c === 5.5, 'T4 failed')

  // T5: odrzuca braki OHLC
  const oh = transformApiData([{ timestamp: '2025-01-01T10:00:00Z', close: 7 }])
  console.assert(oh.length === 0, 'T5 failed')
}

declare global {
  interface Window { __NO_TESTS__?: boolean }
}
if (typeof window !== 'undefined' && !window.__NO_TESTS__) {
  try { runTests() } catch (e) { console.warn('Mini-tests failed:', e) }
}
