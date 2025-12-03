import csv
import math
import json
from datetime import datetime, timezone, timedelta
import argparse


# ============================================
# POMOCNICZE STATYSTYKI / NARZĘDZIA
# ============================================

def percentile_rank(value, sample):
    if value is None or not sample:
        return None
    n = len(sample)
    if n == 0:
        return None
    count_le = sum(1 for v in sample if v is not None and v <= value)
    return 100.0 * count_le / n


def percentile(sample, q):
    if not sample:
        return None
    if q <= 0:
        return min(sample)
    if q >= 100:
        return max(sample)

    s = sorted(sample)
    n = len(s)
    if n == 1:
        return s[0]

    pos = (n - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    if lower == upper:
        return s[lower]
    w = pos - lower
    return s[lower] * (1.0 - w) + s[upper] * w


def linear_regression_slope(y_values):
    if not y_values or len(y_values) < 2:
        return None
    n = len(y_values)
    sumx = (n - 1) * n / 2.0
    sumx2 = (n - 1) * n * (2 * n - 1) / 6.0
    sumy = sum(y_values)
    sumxy = sum(i * y_values[i] for i in range(n))

    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return None
    slope = (n * sumxy - sumx * sumy) / denom
    return slope


def linear_regression(y_values):
    """
    Prosta regresja liniowa: zwraca (slope, intercept).
    Używane do rysowania kanału/flag na M5 (tu już nie używamy).
    """
    if not y_values or len(y_values) < 2:
        return None, None
    n = len(y_values)
    sumx = (n - 1) * n / 2.0
    sumx2 = (n - 1) * n * (2 * n - 1) / 6.0
    sumy = sum(y_values)
    sumxy = sum(i * y_values[i] for i in range(n))

    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return None, None
    slope = (n * sumxy - sumx * sumy) / denom
    intercept = (sumy - slope * sumx) / n
    return slope, intercept


def parse_ts_utc(ts_str: str):
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


# ============================================
# WSKAŹNIKI KLASYCZNE (EMA, RSI, ATR)
# ============================================

def ema(values, period):
    if len(values) < period or period <= 0:
        return [None] * len(values)
    k = 2 / (period + 1)
    ema_vals = [None] * len(values)
    sma = sum(values[:period]) / period
    ema_vals[period - 1] = sma
    for i in range(period, len(values)):
        ema_vals[i] = values[i] * k + ema_vals[i - 1] * (1 - k)
    return ema_vals


def rsi(values, period=14):
    if len(values) < period + 1:
        return [None] * len(values)
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_vals = [None] * len(values)
    if avg_loss == 0:
        rsi_vals[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_vals[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi_vals[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i] = 100 - (100 / (1 + rs))
    return rsi_vals


def atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return [None] * len(highs)
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
    atr_vals = [None] * len(highs)
    first_atr = sum(trs[1:period + 1]) / period
    atr_vals[period] = first_atr
    for i in range(period + 1, len(trs)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + trs[i]) / period
    return atr_vals


# ============================================
# SWINGI + STARE S/R
# ============================================

def detect_swings(candles, lookback=80, swing_window=2):
    if len(candles) == 0:
        return [], []
    if len(candles) < lookback:
        recent = candles
    else:
        recent = candles[-lookback:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    swing_highs = []
    swing_lows = []
    for i in range(swing_window, len(recent) - swing_window):
        local_high = max(highs[i - swing_window:i + swing_window + 1])
        local_low = min(lows[i - swing_window:i + swing_window + 1])
        if highs[i] == local_high and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append(highs[i])
        if lows[i] == local_low and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def cluster_levels(levels, tolerance_pct=0.002):
    if not levels:
        return []
    levels_sorted = sorted(levels)
    zones = []
    current_cluster = [levels_sorted[0]]
    for p in levels_sorted[1:]:
        cluster_avg = sum(current_cluster) / len(current_cluster)
        if abs(p - cluster_avg) / cluster_avg <= tolerance_pct:
            current_cluster.append(p)
        else:
            level = sum(current_cluster) / len(current_cluster)
            strength = len(current_cluster)
            zones.append({"level": level, "strength": strength})
            current_cluster = [p]
    if current_cluster:
        level = sum(current_cluster) / len(current_cluster)
        strength = len(current_cluster)
        zones.append({"level": level, "strength": strength})
    zones.sort(key=lambda z: z["strength"], reverse=True)
    return zones


def calc_sr_zones(hourly_candles, lookback=80, swing_window=2, tolerance_pct=0.002):
    swing_highs, swing_lows = detect_swings(hourly_candles, lookback, swing_window)
    resistance_zones = cluster_levels(swing_highs, tolerance_pct)
    support_zones = cluster_levels(swing_lows, tolerance_pct)
    return support_zones, resistance_zones


# ============================================
# NOWE S/R Z META (D1 + H1)
# ============================================

def detect_swings_with_meta(candles, lookback=200, swing_window=2):
    """
    Zwraca swing_highs/swing_lows jako dicty:
    { "price": ..., "ts": ..., "idx": indeks w 'candles' }.
    """
    if not candles:
        return [], []

    if len(candles) < lookback:
        recent = candles
    else:
        recent = candles[-lookback:]

    offset = len(candles) - len(recent)

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    swing_highs = []
    swing_lows = []

    for i in range(swing_window, len(recent) - swing_window):
        local_high = max(highs[i - swing_window:i + swing_window + 1])
        local_low = min(lows[i - swing_window:i + swing_window + 1])

        if highs[i] == local_high and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append({
                "price": highs[i],
                "ts": recent[i]["ts"],
                "idx": offset + i,
            })
        if lows[i] == local_low and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append({
                "price": lows[i],
                "ts": recent[i]["ts"],
                "idx": offset + i,
            })

    return swing_highs, swing_lows


def cluster_levels_with_meta(points, run_at_dt, tolerance_pct=0.002,
                             timeframe="H1", freshness_half_life_hours=72.0):
    if not points:
        return []

    points_sorted = sorted(points, key=lambda p: p["price"])
    clusters = []
    current_cluster = [points_sorted[0]]

    for pt in points_sorted[1:]:
        cluster_avg = sum(p["price"] for p in current_cluster) / len(current_cluster)
        if abs(pt["price"] - cluster_avg) / cluster_avg <= tolerance_pct:
            current_cluster.append(pt)
        else:
            clusters.append(current_cluster)
            current_cluster = [pt]
    if current_cluster:
        clusters.append(current_cluster)

    zones = []
    for cluster in clusters:
        level = sum(p["price"] for p in cluster) / len(cluster)
        strength = len(cluster)
        last_touch_ts = max(cluster, key=lambda p: p["ts"])["ts"]
        last_dt = parse_ts_utc(last_touch_ts)
        age_hours = None
        freshness = None
        if run_at_dt is not None and last_dt is not None:
            dt_diff = run_at_dt - last_dt
            age_hours = max(dt_diff.total_seconds() / 3600.0, 0.0)
            if freshness_half_life_hours and freshness_half_life_hours > 0:
                freshness = strength * math.exp(-age_hours / freshness_half_life_hours)
            else:
                freshness = float(strength)

        zones.append({
            "level": level,
            "strength": strength,
            "timeframe": timeframe,
            "last_touch_ts": last_touch_ts,
            "age_hours": age_hours,
            "freshness": freshness,
        })

    zones.sort(key=lambda z: (z["freshness"] if z["freshness"] is not None else 0.0),
               reverse=True)
    return zones


def calc_sr_multi_zones(daily_candles, hourly_candles, run_at_dt):
    d_sw_hi, d_sw_lo = detect_swings_with_meta(daily_candles, lookback=200, swing_window=2)
    d_res = cluster_levels_with_meta(
        d_sw_hi,
        run_at_dt=run_at_dt,
        tolerance_pct=0.002,
        timeframe="D1",
        freshness_half_life_hours=24.0 * 10.0,
    )
    d_sup = cluster_levels_with_meta(
        d_sw_lo,
        run_at_dt=run_at_dt,
        tolerance_pct=0.002,
        timeframe="D1",
        freshness_half_life_hours=24.0 * 10.0,
    )

    h_sw_hi, h_sw_lo = detect_swings_with_meta(hourly_candles, lookback=80, swing_window=2)
    h_res = cluster_levels_with_meta(
        h_sw_hi,
        run_at_dt=run_at_dt,
        tolerance_pct=0.002,
        timeframe="H1",
        freshness_half_life_hours=72.0,
    )
    h_sup = cluster_levels_with_meta(
        h_sw_lo,
        run_at_dt=run_at_dt,
        tolerance_pct=0.002,
        timeframe="H1",
        freshness_half_life_hours=72.0,
    )

    support_zones = d_sup + h_sup
    resistance_zones = d_res + h_res

    support_zones.sort(key=lambda z: (z["freshness"] if z["freshness"] is not None else 0.0),
                       reverse=True)
    resistance_zones.sort(key=lambda z: (z["freshness"] if z["freshness"] is not None else 0.0),
                          reverse=True)

    return support_zones, resistance_zones


def find_nearest_zone(zones, price, direction="any"):
    if not zones:
        return None
    if direction == "below":
        candidates = [z for z in zones if z["level"] <= price]
    elif direction == "above":
        candidates = [z for z in zones if z["level"] >= price]
    else:
        candidates = zones[:]
    if not candidates:
        candidates = zones[:]
    best = min(candidates, key=lambda z: abs(z["level"] - price))
    return best


# ============================================
# IMPULS NA H1 (do flag)
# ============================================

def detect_h1_impulse(hourly_candles, closes_h, atr_h, last_idx_h, h_sw_hi, h_sw_lo):
    """
    Prosta definicja impulsu na H1:
    - bierzemy ostatni swing (low / high),
    - jeśli ostatni swing to low -> impuls UP,
      start = ten low, koniec = ostatni close,
    - jeśli ostatni swing to high -> impuls DOWN,
      start = ten high, koniec = ostatni close.
    Zwraca dict z metadanymi albo None.
    """
    if last_idx_h is None or last_idx_h < 0:
        return None
    if (not h_sw_hi) and (not h_sw_lo):
        return None

    recent_low = h_sw_lo[-1] if h_sw_lo else None
    recent_high = h_sw_hi[-1] if h_sw_hi else None

    if recent_low and recent_high:
        if recent_low["idx"] > recent_high["idx"]:
            direction = "UP"
            start = recent_low
        else:
            direction = "DOWN"
            start = recent_high
    elif recent_low:
        direction = "UP"
        start = recent_low
    else:
        direction = "DOWN"
        start = recent_high

    start_idx = start.get("idx")
    if start_idx is None:
        return None
    if start_idx >= len(hourly_candles):
        return None
    if start_idx >= last_idx_h:
        return None

    start_price = start["price"]
    end_price = closes_h[last_idx_h]
    if start_price is None or end_price is None or start_price == 0:
        return None

    diff = end_price - start_price
    size_abs = abs(diff)
    atr_val = atr_h[last_idx_h] if atr_h and last_idx_h < len(atr_h) else None
    size_atr = size_abs / atr_val if atr_val not in (None, 0) else None
    size_pct = diff / start_price * 100.0
    bars = last_idx_h - start_idx

    return {
        "direction": direction,
        "start_idx": start_idx,
        "start_ts": start.get("ts"),
        "start_price": start_price,
        "end_idx": last_idx_h,
        "end_ts": hourly_candles[last_idx_h]["ts"],
        "end_price": end_price,
        "size_abs": size_abs,
        "size_atr": size_atr,
        "size_pct": size_pct,
        "bars": bars,
    }


# ============================================
# FLAGA / KANAŁ NA M5 (TU JUŻ NIE UŻYWAMY – TYLKO DLA KOMPATYBILNOŚCI)
# ============================================

def detect_flag_channel(
    m5_candles,
    impulse_direction,
    atr_ref,
    max_bars=60,
    min_bars=20,
    min_width_atr=0.3,
    max_width_atr=2.5,
    max_trend_atr=1.5,
):
    """
    Zostawione tylko dla kompatybilności kodu – NIE UŻYWAMY w wersji fast.
    """
    return None


# ============================================
# WCZYTYWANIE ŚWIEC Z CSV
# ============================================

def load_candles_from_csv(path, limit=None):
    """
    CSV:
    DD/MM/YYYY;HH:MM:SS;Open;High;Low;Close;Volume
    => [{ts, open, high, low, close}]
    """
    candles = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row or len(row) < 6:
                continue
            date_str = row[0].strip()
            time_str = row[1].strip()
            dt = datetime.strptime(date_str + " " + time_str, "%d/%m/%Y %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            o = float(row[2])
            h = float(row[3])
            l = float(row[4])
            c = float(row[5])
            candles.append({
                "ts": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            })
    candles.sort(key=lambda c: c["ts"])
    if limit is not None and len(candles) > limit:
        candles = candles[-limit:]
    return candles


# ============================================
# PRECOMPUTE M5 – GŁÓWNA OPTYMALIZACJA
# ============================================

def precompute_m5_indicators(m5_candles):
    """
    Liczymy RSI/ATR/EMA dla M5 tylko raz na całą serię.
    Zwracamy dict z tablicami, żeby w pętli po barach tylko odczytywać.
    """
    closes = [c["close"] for c in m5_candles]
    highs = [c["high"] for c in m5_candles]
    lows = [c["low"] for c in m5_candles]

    rsi14_all = rsi(closes, 14)
    atr14_all = atr(highs, lows, closes, 14)
    ema20_all = ema(closes, 20)
    ema50_all = ema(closes, 50)

    return {
        "candles": m5_candles,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "rsi14": rsi14_all,
        "atr14": atr14_all,
        "ema20": ema20_all,
        "ema50": ema50_all,
    }


# ============================================
# SNAPSHOT DLA "STANU" DO DANEGO CZASU
# ============================================

def build_snapshot_from_slices(
    daily_candles,
    hourly_candles,
    m5_pre,
    last_m5_index,
    now_dt,
    instrument="BZ=F",
):
    run_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = {
        "instrument": instrument,
        "run_at": run_at,
    }

    # --- D1 ---
    daily_part = {"candles_count": len(daily_candles)}
    if len(daily_candles) >= 1:
        closes_d = [c["close"] for c in daily_candles]
        highs_d = [c["high"] for c in daily_candles]
        lows_d = [c["low"] for c in daily_candles]

        ema20_d = ema(closes_d, 20)
        ema50_d = ema(closes_d, 50)
        rsi14_d = rsi(closes_d, 14)
        atr14_d = atr(highs_d, lows_d, closes_d, 14)

        idx = len(closes_d) - 2 if len(closes_d) >= 2 else len(closes_d) - 1
        if idx >= 0:
            last_close_d = closes_d[idx]
            daily_part.update({
                "idx": idx,
                "ts": daily_candles[idx]["ts"],
                "close": last_close_d,
                "ema20": ema20_d[idx],
                "ema50": ema50_d[idx],
                "rsi14": rsi14_d[idx],
            })

            atr_val_d = atr14_d[idx] if idx < len(atr14_d) else None
            daily_part["atr14"] = atr_val_d

            rsi_sample = [v for i, v in enumerate(rsi14_d) if v is not None and i <= idx]
            atr_sample = [v for i, v in enumerate(atr14_d) if v is not None and i <= idx]

            if rsi_sample and len(rsi_sample) >= 20:
                rsi_tail = rsi_sample[-200:]
                daily_part["rsi14_pct_rank"] = percentile_rank(rsi14_d[idx], rsi_tail)
                daily_part["rsi14_q20"] = percentile(rsi_tail, 20)
                daily_part["rsi14_q50"] = percentile(rsi_tail, 50)
                daily_part["rsi14_q80"] = percentile(rsi_tail, 80)
            else:
                daily_part["rsi14_pct_rank"] = None
                daily_part["rsi14_q20"] = None
                daily_part["rsi14_q50"] = None
                daily_part["rsi14_q80"] = None

            if atr_sample and len(atr_sample) >= 20:
                atr_tail = atr_sample[-200:]
                daily_part["atr14_pct_rank"] = percentile_rank(atr_val_d, atr_tail)
            else:
                daily_part["atr14_pct_rank"] = None

            ema50_sample = [v for i, v in enumerate(ema50_d) if v is not None and i <= idx]
            if ema50_sample and len(ema50_sample) >= 10:
                ema50_tail = ema50_sample[-50:]
                slope_ema50 = linear_regression_slope(ema50_tail)
                daily_part["ema50_slope"] = slope_ema50
                if last_close_d not in (None, 0):
                    daily_part["ema50_slope_pct"] = slope_ema50 / last_close_d
                else:
                    daily_part["ema50_slope_pct"] = None
            else:
                daily_part["ema50_slope"] = None
                daily_part["ema50_slope_pct"] = None

    snapshot["daily"] = daily_part

    # --- H1 ---
    h1_part = {"candles_count": len(hourly_candles)}
    support_zones_old, resistance_zones_old = calc_sr_zones(hourly_candles)
    snapshot["sr"] = {
        "support_zones": support_zones_old,
        "resistance_zones": resistance_zones_old,
    }

    if len(hourly_candles) >= 1:
        closes_h = [c["close"] for c in hourly_candles]
        highs_h = [c["high"] for c in hourly_candles]
        lows_h = [c["low"] for c in hourly_candles]

        ema_period = min(20, max(10, len(closes_h) - 2))
        ema_h = ema(closes_h, ema_period)
        rsi_h = rsi(closes_h, 14)
        atr_h = atr(highs_h, lows_h, closes_h, 14)

        last_idx_h = len(closes_h) - 2 if len(closes_h) >= 2 else len(closes_h) - 1
        if last_idx_h >= 0:
            last_close_h = closes_h[last_idx_h]
            e = ema_h[last_idx_h]
            r = rsi_h[last_idx_h]
            a = atr_h[last_idx_h] if last_idx_h < len(atr_h) else None

            h1_part.update({
                "idx": last_idx_h,
                "ts": hourly_candles[last_idx_h]["ts"],
                "close": last_close_h,
                "ema_period": ema_period,
                "ema": e,
                "rsi14": r,
                "atr14": a,
            })

            # Swings z meta (z idx)
            h_sw_hi, h_sw_lo = detect_swings_with_meta(
                hourly_candles,
                lookback=80,
                swing_window=2,
            )

            last_low = prev_low = last_high = prev_high = None
            last_low_ts = prev_low_ts = last_high_ts = prev_high_ts = None

            if len(h_sw_lo) >= 2:
                last_low = h_sw_lo[-1]["price"]
                last_low_ts = h_sw_lo[-1]["ts"]
                prev_low = h_sw_lo[-2]["price"]
                prev_low_ts = h_sw_lo[-2]["ts"]

            if len(h_sw_hi) >= 2:
                last_high = h_sw_hi[-1]["price"]
                last_high_ts = h_sw_hi[-1]["ts"]
                prev_high = h_sw_hi[-2]["price"]
                prev_high_ts = h_sw_hi[-2]["ts"]

            h1_part.update({
                "last_low": last_low,
                "prev_low": prev_low,
                "last_high": last_high,
                "prev_high": prev_high,
                "last_low_ts": last_low_ts,
                "prev_low_ts": prev_low_ts,
                "last_high_ts": last_high_ts,
                "prev_high_ts": prev_high_ts,
            })

            # Impuls H1 (meta do flag)
            impulse_info = detect_h1_impulse(
                hourly_candles,
                closes_h,
                atr_h,
                last_idx_h,
                h_sw_hi,
                h_sw_lo,
            )
            if impulse_info:
                h1_part.update({
                    "impulse_direction": impulse_info["direction"],
                    "impulse_start_ts": impulse_info["start_ts"],
                    "impulse_start_price": impulse_info["start_price"],
                    "impulse_end_ts": impulse_info["end_ts"],
                    "impulse_end_price": impulse_info["end_price"],
                    "impulse_size_abs": impulse_info["size_abs"],
                    "impulse_size_atr": impulse_info["size_atr"],
                    "impulse_size_pct": impulse_info["size_pct"],
                    "impulse_bars": impulse_info["bars"],
                })
            else:
                h1_part.update({
                    "impulse_direction": None,
                    "impulse_start_ts": None,
                    "impulse_start_price": None,
                    "impulse_end_ts": None,
                    "impulse_end_price": None,
                    "impulse_size_abs": None,
                    "impulse_size_atr": None,
                    "impulse_size_pct": None,
                    "impulse_bars": None,
                })

            atr_h_sample = [v for i, v in enumerate(atr_h) if v is not None and i <= last_idx_h]
            if atr_h_sample and len(atr_h_sample) >= 20:
                atr_tail_h = atr_h_sample[-200:]
                h1_part["atr14_pct_rank"] = percentile_rank(a, atr_tail_h)
            else:
                h1_part["atr14_pct_rank"] = None

            ema_h_sample = [v for i, v in enumerate(ema_h) if v is not None and i <= last_idx_h]
            if ema_h_sample and len(ema_h_sample) >= 10:
                ema_h_tail = ema_h_sample[-50:]
                slope_ema_h = linear_regression_slope(ema_h_tail)
                h1_part["ema_slope"] = slope_ema_h
                if last_close_h not in (None, 0):
                    h1_part["ema_slope_pct"] = slope_ema_h / last_close_h
                else:
                    h1_part["ema_slope_pct"] = None
            else:
                h1_part["ema_slope"] = None
                h1_part["ema_slope_pct"] = None

            rsi_h_sample = [v for i, v in enumerate(rsi_h) if v is not None and i <= last_idx_h]
            if rsi_h_sample and len(rsi_h_sample) >= 20:
                rsi_h_tail = rsi_h_sample[-200:]
                h1_part["rsi14_pct_rank"] = percentile_rank(r, rsi_h_tail)
            else:
                h1_part["rsi14_pct_rank"] = None

    snapshot["h1"] = h1_part

    # --- SR MULTI-TF ---
    sup_multi, res_multi = calc_sr_multi_zones(daily_candles, hourly_candles, now_dt)
    snapshot["sr_multi"] = {
        "support_zones": sup_multi,
        "resistance_zones": res_multi,
    }

    # --- M5 (z precomputed) ---
    m5_candles = m5_pre["candles"]
    closes_m5 = m5_pre["closes"]
    highs_m5 = m5_pre["highs"]
    lows_m5 = m5_pre["lows"]
    rsi_5_all = m5_pre["rsi14"]
    atr_5_all = m5_pre["atr14"]
    ema20_5_all = m5_pre["ema20"]
    ema50_5_all = m5_pre["ema50"]

    m5_part = {"candles_count": last_m5_index + 1}

    if last_m5_index >= 0:
        last_idx_5 = last_m5_index
        last_candle_5 = m5_candles[last_idx_5]
        last_close_5 = closes_m5[last_idx_5]
        last_rsi_5 = rsi_5_all[last_idx_5] if last_idx_5 < len(rsi_5_all) else None
        last_atr_5 = atr_5_all[last_idx_5] if last_idx_5 < len(atr_5_all) else None
        e20_5 = ema20_5_all[last_idx_5] if last_idx_5 < len(ema20_5_all) else None
        e50_5 = ema50_5_all[last_idx_5] if last_idx_5 < len(ema50_5_all) else None

        prev_rsi_5 = rsi_5_all[last_idx_5 - 1] if last_idx_5 - 1 >= 0 else None

        window = 5
        if last_idx_5 >= window:
            slice_start = last_idx_5 - window
            slice_end = last_idx_5
            highs_slice = highs_m5[slice_start:slice_end]
            lows_slice = lows_m5[slice_start:slice_end]

            recent_high = max(highs_slice) if highs_slice else None
            recent_low = min(lows_slice) if lows_slice else None

            recent_high_age_bars = None
            recent_low_age_bars = None

            if highs_slice and recent_high is not None:
                for i in range(len(highs_slice) - 1, -1, -1):
                    if highs_slice[i] == recent_high:
                        recent_high_age_bars = window - i
                        break

            if lows_slice and recent_low is not None:
                for i in range(len(lows_slice) - 1, -1, -1):
                    if lows_slice[i] == recent_low:
                        recent_low_age_bars = window - i
                        break
        else:
            recent_high = None
            recent_low = None
            recent_high_age_bars = None
            recent_low_age_bars = None

        # SR z sr_multi
        sr_multi_sup = snapshot.get("sr_multi", {}).get("support_zones") or []
        sr_multi_res = snapshot.get("sr_multi", {}).get("resistance_zones") or []

        nearest_support = None
        nearest_resistance = None
        support_dist_atr = None
        resistance_dist_atr = None
        nearest_support_tf = None
        nearest_support_freshness = None
        nearest_support_age_hours = None
        nearest_resistance_tf = None
        nearest_resistance_freshness = None
        nearest_resistance_age_hours = None

        if sr_multi_sup:
            nearest_support = find_nearest_zone(sr_multi_sup, last_close_5, direction="below")
        if sr_multi_res:
            nearest_resistance = find_nearest_zone(sr_multi_res, last_close_5, direction="above")

        if nearest_support:
            nearest_support_tf = nearest_support.get("timeframe")
            nearest_support_freshness = nearest_support.get("freshness")
            nearest_support_age_hours = nearest_support.get("age_hours")

        if nearest_resistance:
            nearest_resistance_tf = nearest_resistance.get("timeframe")
            nearest_resistance_freshness = nearest_resistance.get("freshness")
            nearest_resistance_age_hours = nearest_resistance.get("age_hours")

        if last_atr_5 not in (None, 0):
            if nearest_support:
                support_dist_atr = abs(last_close_5 - nearest_support["level"]) / last_atr_5
            if nearest_resistance:
                resistance_dist_atr = abs(last_close_5 - nearest_resistance["level"]) / last_atr_5

        rsi5_sample = [v for i, v in enumerate(rsi_5_all) if v is not None and i <= last_idx_5]
        atr5_sample = [v for i, v in enumerate(atr_5_all) if v is not None and i <= last_idx_5]

        if rsi5_sample and len(rsi5_sample) >= 20:
            rsi5_tail = rsi5_sample[-200:]
            m5_part["rsi14_pct_rank"] = percentile_rank(last_rsi_5, rsi5_tail)
            if prev_rsi_5 is not None:
                m5_part["prev_rsi14_pct_rank"] = percentile_rank(prev_rsi_5, rsi5_tail)
            else:
                m5_part["prev_rsi14_pct_rank"] = None
        else:
            m5_part["rsi14_pct_rank"] = None
            m5_part["prev_rsi14_pct_rank"] = None

        if atr5_sample and len(atr5_sample) >= 20:
            atr5_tail = atr5_sample[-200:]
            m5_part["atr14_pct_rank"] = percentile_rank(last_atr_5, atr5_tail)
        else:
            m5_part["atr14_pct_rank"] = None

        m5_part.update({
            "idx": last_idx_5,
            "ts": last_candle_5["ts"],
            "close": last_close_5,
            "rsi14": last_rsi_5,
            "prev_rsi14": prev_rsi_5,
            "atr14": last_atr_5,
            "ema20": e20_5,
            "ema50": e50_5,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "recent_high_age_bars": recent_high_age_bars,
            "recent_low_age_bars": recent_low_age_bars,
            "nearest_support_level": nearest_support["level"] if nearest_support else None,
            "nearest_support_strength": nearest_support["strength"] if nearest_support else None,
            "nearest_support_dist_atr": support_dist_atr,
            "nearest_support_timeframe": nearest_support_tf,
            "nearest_support_freshness": nearest_support_freshness,
            "nearest_support_age_hours": nearest_support_age_hours,
            "nearest_resistance_level": nearest_resistance["level"] if nearest_resistance else None,
            "nearest_resistance_strength": nearest_resistance["strength"] if nearest_resistance else None,
            "nearest_resistance_dist_atr": resistance_dist_atr,
            "nearest_resistance_timeframe": nearest_resistance_tf,
            "nearest_resistance_freshness": nearest_resistance_freshness,
            "nearest_resistance_age_hours": nearest_resistance_age_hours,
        })

        # --- FLAGA NA M5 – WYŁĄCZONA, ALE KOLUMNY ZOSTAJĄ ---
        m5_part.update({
            "flag_active": False,
            "flag_impulse_direction": h1_part.get("impulse_direction"),
            "flag_upper": None,
            "flag_lower": None,
            "flag_width": None,
            "flag_width_atr": None,
            "flag_slope": None,
            "flag_age_bars": None,
            "flag_position": None,
        })

    snapshot["m5"] = m5_part

    return snapshot


# ============================================
# FLATTEN SNAPSHOT -> PŁASKI WIERSZ
# ============================================

def flatten_snapshot(snapshot):
    row = {}
    row["instrument"] = snapshot.get("instrument")
    row["run_at"] = snapshot.get("run_at")

    for section_name in ["daily", "h1", "m5"]:
        section = snapshot.get(section_name, {}) or {}
        for k, v in section.items():
            col_name = f"{section_name}_{k}"
            row[col_name] = v

    # pełne sr_multi (D1+H1) jako JSON w jednej kolumnie
    sr_multi = snapshot.get("sr_multi") or {}
    row["sr_multi_json"] = json.dumps(sr_multi, default=str)

    return row


# ============================================
# MAIN – ITERACJA PO KAŻDEJ ŚWIECY 5M
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_5m", default="brent_5m.csv")
    parser.add_argument("--file_1h", default="brent_1h.csv")
    parser.add_argument("--file_1d", default="brent_1d.csv")
    parser.add_argument("--output", default="brent_snapshots_5m_fast.csv")
    parser.add_argument("--instrument", default="BZ=F")
    parser.add_argument(
        "--days_back",
        type=int,
        default=None,
        help="ile dni wstecz od ostatniego bara M5 generować snapshoty (np. 30). "
             "Jeśli nie podane, użyje całej historii."
    )
    parser.add_argument(
        "--skip_days",
        type=int,
        default=0,
        help="ile dni od najnowszej świecy M5 pominąć (cofnąć się w czasie). "
             "Np. --days_back 90 --skip_days 90 => zakres od -180d do -90d."
    )

    args = parser.parse_args()

    daily_all = load_candles_from_csv(args.file_1d)
    hourly_all = load_candles_from_csv(args.file_1h)
    m5_all = load_candles_from_csv(args.file_5m)

    if not m5_all:
        print("Brak danych M5, przerywam.")
        return

    # --- FILTROWANIE M5 DO OSTATNICH N DNI (OD OSTATNIEGO BARA) ---
    if args.days_back is not None and args.days_back > 0:
        last_dt = parse_ts_utc(m5_all[-1]["ts"])

        if args.skip_days is not None and args.skip_days > 0:
            end_dt = last_dt - timedelta(days=args.skip_days)
        else:
            end_dt = last_dt

        start_dt = end_dt - timedelta(days=args.days_back)

        start_ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        m5_all = [c for c in m5_all if start_ts <= c["ts"] <= end_ts]

        if not m5_all:
            print(
                f"Po odfiltrowaniu do okna days_back={args.days_back}, "
                f"skip_days={args.skip_days} brak danych M5 – przerywam."
            )
            return

        first_dt = parse_ts_utc(m5_all[0]["ts"])
        last_filtered_dt = parse_ts_utc(m5_all[-1]["ts"])
        print(
            f"Restricting M5 to window of {args.days_back} days "
            f"(skip {args.skip_days} days from latest): "
            f"from {first_dt} to {last_filtered_dt}"
        )

    print(f"Loaded candles: D1={len(daily_all)}, H1={len(hourly_all)}, M5(filtered)={len(m5_all)}")

    # --- PRECOMPUTE M5 ---
    print("Precomputing M5 indicators (RSI/ATR/EMA)...")
    m5_pre = precompute_m5_indicators(m5_all)
    print("  Done precompute M5.")

    total = len(m5_all)

    # --- pierwsza świeca: budujemy nagłówek CSV ---
    first_ts = m5_all[0]["ts"]
    first_now_dt = parse_ts_utc(first_ts)

    daily_slice0 = [c for c in daily_all if c["ts"] <= first_ts]
    hourly_slice0 = [c for c in hourly_all if c["ts"] <= first_ts]

    first_snapshot = build_snapshot_from_slices(
        daily_candles=daily_slice0,
        hourly_candles=hourly_slice0,
        m5_pre=m5_pre,
        last_m5_index=0,
        now_dt=first_now_dt,
        instrument=args.instrument,
    )
    first_row = flatten_snapshot(first_snapshot)

    fieldnames = list(first_row.keys())
    fieldnames.sort()

    # --- zapis na bieżąco do pliku ---
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        # pierwszy bar
        progress0 = 1 / total * 100.0
        print(
            f"[{progress0:6.2f}%] M5 bar ts={m5_all[0]['ts']}, "
            f"open={m5_all[0]['open']}, high={m5_all[0]['high']}, "
            f"low={m5_all[0]['low']}, close={m5_all[0]['close']}",
            flush=True,
        )
        writer.writerow(first_row)
        f.flush()

        # kolejne bary
        for i in range(1, total):
            current_ts = m5_all[i]["ts"]
            now_dt = parse_ts_utc(current_ts)

            progress = (i + 1) / total * 100.0
            print(
                f"[{progress:6.2f}%] M5 bar ts={current_ts}, "
                f"open={m5_all[i]['open']}, high={m5_all[i]['high']}, "
                f"low={m5_all[i]['low']}, close={m5_all[i]['close']}",
                flush=True,
            )

            daily_slice = [c for c in daily_all if c["ts"] <= current_ts]
            hourly_slice = [c for c in hourly_all if c["ts"] <= current_ts]

            snapshot = build_snapshot_from_slices(
                daily_candles=daily_slice,
                hourly_candles=hourly_slice,
                m5_pre=m5_pre,
                last_m5_index=i,
                now_dt=now_dt,
                instrument=args.instrument,
            )
            row = flatten_snapshot(snapshot)
            writer.writerow(row)
            f.flush()

    print(f"Zapisano {total} snapshotów do pliku: {args.output}")


if __name__ == "__main__":
    main()
