import os
import json
import decimal
from decimal import Decimal
from datetime import datetime, timezone
import boto3

dynamodb = boto3.resource("dynamodb")

# --- konfiguracja z env ---
OHLC_TABLE_NAME = os.getenv("OHLC_TABLE_NAME", "brent_ohlc")
SIGNALS_TABLE_NAME = os.getenv("SIGNALS_TABLE_NAME", "brent_signals")
DEBUG_TABLE_NAME = os.getenv("DEBUG_TABLE_NAME", "brent_signals_debug")
INSTRUMENT = os.getenv("INSTRUMENT", "BZ=F")

TIMEFRAME_1D = "1d"
TIMEFRAME_1H = "1h"
TIMEFRAME_5M = "5m"


# --------- Pomocnicze: konwersja typów ---------
def _to_float(x):
    if isinstance(x, decimal.Decimal):
        return float(x)
    return x


def _to_decimal(x):
    # UWAGA: bool jest podklasą int, więc musimy go wykluczyć
    if isinstance(x, bool):
        return x
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _to_decimal_deep(obj):
    """
    Rekurencyjna konwersja liczb (float/int/Decimal) w całej strukturze na Decimal,
    z pominięciem booli.
    """
    # bool zostawiamy tak jak jest
    if isinstance(obj, bool):
        return obj

    # liczby -> Decimal
    if isinstance(obj, (float, int, Decimal, decimal.Decimal)):
        return _to_decimal(obj)

    # listy -> rekurencyjnie po elementach
    if isinstance(obj, list):
        return [_to_decimal_deep(v) for v in obj]

    # słowniki -> rekurencyjnie po wartościach
    if isinstance(obj, dict):
        return {k: _to_decimal_deep(v) for k, v in obj.items()}

    # inne typy (str, None, itp.) zostawiamy
    return obj


# --------- Pobieranie danych z DynamoDB ---------
def fetch_candles(pk_value: str, limit: int = 300):
    """
    Pobiera ostatnie `limit` świec dla danego pk (np. BZ=F#1d)
    Zwraca listę posortowaną rosnąco po ts.
    """
    print(f"[DEBUG] Fetching candles for pk={pk_value} from table={OHLC_TABLE_NAME}, limit={limit}")
    table = dynamodb.Table(OHLC_TABLE_NAME)

    resp = table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": pk_value},
        ScanIndexForward=False,  # DESC – od najnowszej
        Limit=limit,
    )

    items = resp.get("Items", [])
    print(f"[DEBUG] Got {len(items)} raw items for pk={pk_value}")

    # odwracamy, żeby mieć rosnąco po czasie (najstarsza -> najnowsza)
    items = list(reversed(items))

    candles = []
    for it in items:
        candles.append({
            "ts": it["ts"],
            "open": _to_float(it["open"]),
            "high": _to_float(it["high"]),
            "low": _to_float(it["low"]),
            "close": _to_float(it["close"]),
            # volume ignorujemy
        })

    if candles:
        print(f"[DEBUG] First candle {pk_value}: {candles[0]}")
        print(f"[DEBUG] Last  candle {pk_value}: {candles[-1]}")
    else:
        print(f"[DEBUG] No candles returned for pk={pk_value}")

    return candles


# --------- Wskaźniki techniczne ---------
def ema(values, period):
    if len(values) < period:
        return [None] * len(values)
    k = 2 / (period + 1)
    ema_vals = [None] * len(values)
    # start od SMA
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
    # pierwsze RSI pojawi się na indeksie period
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
    # pierwszy ATR = SMA z TR
    first_atr = sum(trs[1:period + 1]) / period
    atr_vals[period] = first_atr

    for i in range(period + 1, len(trs)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + trs[i]) / period

    return atr_vals


# --------- Logika trendu na 1D (na zamkniętej świecy) ---------
def determine_daily_bias(daily_candles):
    """
    Zwraca (bias, debug_dict).
    Bias liczony na PRZEDOSTATNIEJ świecy (ostatnia może być w trakcie).
      - LONG: EMA20 > EMA50 i RSI > 50
      - SHORT: EMA20 < EMA50 i RSI < 50
    """
    print(f"[DEBUG] determine_daily_bias: candles_1d_count={len(daily_candles)}")
    debug = {"candles_count": len(daily_candles)}

    closes = [c["close"] for c in daily_candles]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    rsi14 = rsi(closes, 14)

    if len(closes) < 2:
        print("[DEBUG] determine_daily_bias: not enough candles -> BIAS=NONE")
        debug["reason"] = "not_enough_candles"
        return "NONE", debug

    # używamy przedostatniej świecy – zamkniętej
    idx = len(closes) - 2
    c = closes[idx]
    e20 = ema20[idx]
    e50 = ema50[idx]
    r = rsi14[idx]

    debug.update({
        "idx": idx,
        "ts": daily_candles[idx]["ts"],
        "close": c,
        "ema20": e20,
        "ema50": e50,
        "rsi14": r,
    })

    print(f"[DEBUG] 1D (closed) idx={idx}, close={c}, ema20={e20}, ema50={e50}, rsi14={r}")

    if e20 is None or e50 is None or r is None:
        print("[DEBUG] determine_daily_bias: some indicator is None -> BIAS=NONE")
        debug["reason"] = "indicator_none"
        return "NONE", debug

    if e20 > e50 and r > 50:
        print("[DEBUG] determine_daily_bias: conditions for LONG met (ema20>ema50, rsi>50)")
        debug["reason"] = "long_conditions"
        return "LONG", debug
    elif e20 < e50 and r < 50:
        print("[DEBUG] determine_daily_bias: conditions for SHORT met (ema20<ema50, rsi<50)")
        debug["reason"] = "short_conditions"
        return "SHORT", debug
    else:
        print("[DEBUG] determine_daily_bias: no clear ema20/ema50 + rsi trend -> BIAS=NONE")
        debug["reason"] = "no_clear_trend"
        return "NONE", debug


# --------- Wsparcie / opór z 1H (swings + clustering) ---------
def detect_swings(candles, lookback=80, swing_window=2):
    """
    Detekcja swing high / swing low na 1H.
    Zwraca listy swing_highs, swing_lows (poziomy).
    """
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

    print(f"[DEBUG] detect_swings: found {len(swing_lows)} swing_lows, {len(swing_highs)} swing_highs")
    return swing_highs, swing_lows


def cluster_levels(levels, tolerance_pct=0.002):
    """
    Grupuje poziomy w strefy (level + strength).
    """
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

    print(f"[DEBUG] cluster_levels: input={len(levels)}, zones={zones}")
    return zones


def calc_sr_zones(hourly_candles, lookback=80, swing_window=2, tolerance_pct=0.002):
    """
    Wyznacza strefy wsparcia/oporu na 1H.
    Zwraca (support_zones, resistance_zones).
    """
    swing_highs, swing_lows = detect_swings(hourly_candles, lookback, swing_window)
    resistance_zones = cluster_levels(swing_highs, tolerance_pct)
    support_zones = cluster_levels(swing_lows, tolerance_pct)
    print(f"[DEBUG] calc_sr_zones: support_zones={support_zones}, resistance_zones={resistance_zones}")
    return support_zones, resistance_zones


def find_nearest_zone(zones, price, direction="any"):
    """
    Najbliższa strefa względem ceny.
    """
    if not zones:
        return None

    if direction == "below":
        candidates = [z for z in zones if z["level"] <= price]
    elif direction == "above":
        candidates = [z for z in zones if z["level"] >= price]
    else:
        candidates = zones[:]

    if not candidates:
        candidates = zones[:]  # fallback

    best = min(candidates, key=lambda z: abs(z["level"] - price))
    print(f"[DEBUG] find_nearest_zone: price={price}, dir={direction}, best={best}")
    return best


# --------- Logika 1H (miękka, zamknięta świeca) ---------
def check_1h_setup(hourly_candles, bias):
    """
    Zwraca (setup_ok, debug_dict).
    """
    print(f"[DEBUG] check_1h_setup: candles_1h_count={len(hourly_candles)}, bias={bias}")
    debug = {"candles_count": len(hourly_candles), "bias": bias}

    if len(hourly_candles) < 10:
        print("[DEBUG] check_1h_setup: not enough 1H candles (<10)")
        debug["reason"] = "not_enough_candles"
        return False, debug

    closes = [c["close"] for c in hourly_candles]

    ema_period = min(20, max(10, len(closes) - 2))
    print(f"[DEBUG] check_1h_setup: using ema_period={ema_period}")
    debug["ema_period"] = ema_period

    ema_h = ema(closes, ema_period)
    rsi_h = rsi(closes, 14)

    # używamy przedostatniej świecy 1H
    last_idx = len(closes) - 2
    c = closes[last_idx]
    e = ema_h[last_idx]
    r = rsi_h[last_idx]

    debug.update({
        "idx": last_idx,
        "ts": hourly_candles[last_idx]["ts"],
        "close": c,
        "ema": e,
        "rsi14": r,
    })

    print(f"[DEBUG] 1H (closed) idx={last_idx}, close={c}, ema{ema_period}={e}, rsi14={r}")

    if e is None or r is None:
        print("[DEBUG] check_1h_setup: ema or rsi is None")
        debug["reason"] = "indicator_none"
        return False, debug

    last_low = min(hourly_candles[last_idx - 2:last_idx + 1], key=lambda x: x["low"])["low"]
    prev_low = min(hourly_candles[last_idx - 3:last_idx],   key=lambda x: x["low"])["low"]

    last_high = max(hourly_candles[last_idx - 2:last_idx + 1], key=lambda x: x["high"])["high"]
    prev_high = max(hourly_candles[last_idx - 3:last_idx],     key=lambda x: x["high"])["high"]

    debug.update({
        "last_low": last_low,
        "prev_low": prev_low,
        "last_high": last_high,
        "prev_high": prev_high,
    })

    print(f"[DEBUG] 1H last_low={last_low}, prev_low={prev_low}, last_high={last_high}, prev_high={prev_high}")

    if bias == "LONG":
        cond_price = c > e
        cond_rsi = r > 50
        cond_structure = last_low >= prev_low
        ok = cond_price or cond_rsi or cond_structure
        debug.update({
            "cond_price": cond_price,
            "cond_rsi": cond_rsi,
            "cond_structure": cond_structure,
        })
        print(f"[DEBUG] check_1h_setup LONG -> {ok} (price>{e}? {cond_price}, rsi>50? {cond_rsi}, HL? {cond_structure})")
        debug["reason"] = "long_conditions" if ok else "long_not_met"
        return ok, debug

    elif bias == "SHORT":
        cond_price = c < e
        cond_rsi = r < 50
        cond_structure = last_high <= prev_high
        ok = cond_price or cond_rsi or cond_structure
        debug.update({
            "cond_price": cond_price,
            "cond_rsi": cond_rsi,
            "cond_structure": cond_structure,
        })
        print(f"[DEBUG] check_1h_setup SHORT -> {ok} (price<{e}? {cond_price}, rsi<50? {cond_rsi}, LH? {cond_structure})")
        debug["reason"] = "short_conditions" if ok else "short_not_met"
        return ok, debug

    else:
        print("[DEBUG] check_1h_setup: bias NONE -> False")
        debug["reason"] = "bias_none"
        return False, debug


# --------- Logika wejścia na 5M (z S/R) ---------
def check_5m_trigger(candles_5m, bias, support_zones=None, resistance_zones=None):
    """
    Zwraca (signal_or_None, debug_dict).
    """
    print(f"[DEBUG] check_5m_trigger: candles_5m_count={len(candles_5m)}, bias={bias}")
    debug = {"candles_count": len(candles_5m), "bias": bias}

    if len(candles_5m) < 30:
        print("[DEBUG] check_5m_trigger: not enough 5M candles")
        debug["reason"] = "not_enough_candles"
        return None, debug

    closes = [c["close"] for c in candles_5m]
    highs = [c["high"] for c in candles_5m]
    lows = [c["low"] for c in candles_5m]

    rsi_5 = rsi(closes, 14)
    atr_5 = atr(highs, lows, closes, 14)
    ema20_5 = ema(closes, 20)
    ema50_5 = ema(closes, 50)

    last_idx = len(candles_5m) - 1
    last_candle = candles_5m[last_idx]
    last_close = closes[last_idx]
    last_rsi = rsi_5[last_idx]
    last_atr = atr_5[last_idx]
    e20 = ema20_5[last_idx]
    e50 = ema50_5[last_idx]

    debug.update({
        "idx": last_idx,
        "ts": last_candle["ts"],
        "last_close": last_close,
        "last_rsi": last_rsi,
        "atr": last_atr,
        "ema20": e20,
        "ema50": e50,
    })

    print(f"[DEBUG] 5M last candle={last_candle}")
    print(f"[DEBUG] 5M last_close={last_close}, rsi={last_rsi}, atr={last_atr}, ema20={e20}, ema50={e50}")

    if last_rsi is None or last_atr is None or e20 is None or e50 is None:
        print("[DEBUG] check_5m_trigger: some indicator is None -> no signal")
        debug["reason"] = "indicator_none"
        return None, debug

    window = 5
    recent_high = max(highs[last_idx - window:last_idx])
    recent_low = min(lows[last_idx - window:last_idx])

    debug.update({
        "recent_high": recent_high,
        "recent_low": recent_low,
    })

    print(f"[DEBUG] 5M recent_high={recent_high}, recent_low={recent_low}")

    signal = None

    def is_near_zone(zone, max_atr_multiple=1.0):
        if zone is None or last_atr is None or last_atr == 0:
            return False, None
        dist = abs(last_close - zone["level"])
        multiple = dist / last_atr
        return multiple <= max_atr_multiple, multiple

    if bias == "LONG":
        prev_rsi = rsi_5[last_idx - 1]
        debug["prev_rsi"] = prev_rsi
        print(f"[DEBUG] 5M LONG prev_rsi={prev_rsi}, last_rsi={last_rsi}")
        if prev_rsi is not None and prev_rsi < 40 and last_rsi > 40:
            if last_close > recent_high and last_close > e20 and last_close > e50:
                nearest_support = find_nearest_zone(support_zones or [], last_close, "below")
                near, multiple = is_near_zone(nearest_support, max_atr_multiple=1.0)
                debug["nearest_support"] = nearest_support
                debug["nearest_support_near"] = near
                debug["nearest_support_dist_atr"] = multiple
                print(f"[DEBUG] 5M LONG nearest_support={nearest_support}, near={near}, dist_atr={multiple}")
                if nearest_support and near:
                    print("[DEBUG] 5M LONG trigger conditions + S/R met")
                    signal = {
                        "side": "BUY",
                        "timestamp": last_candle["ts"],
                        "price": last_close,
                        "atr": last_atr,
                        "sr_level": nearest_support["level"],
                        "sr_strength": nearest_support["strength"],
                        "sr_distance_atr": multiple,
                    }
                    debug["reason"] = "trigger_long_sr_ok"
                else:
                    print("[DEBUG] 5M LONG: dobre wybicie, ale daleko od 1H wsparcia -> brak sygnału")
                    debug["reason"] = "long_far_from_support"
            else:
                print("[DEBUG] 5M LONG: RSI ok, ale brak wybicia high / powyżej EMA")
                debug["reason"] = "long_no_breakout"
        else:
            print("[DEBUG] 5M LONG: RSI pattern not met")
            debug["reason"] = "long_rsi_pattern_not_met"

    elif bias == "SHORT":
        prev_rsi = rsi_5[last_idx - 1]
        debug["prev_rsi"] = prev_rsi
        print(f"[DEBUG] 5M SHORT prev_rsi={prev_rsi}, last_rsi={last_rsi}")
        if prev_rsi is not None and prev_rsi > 60 and last_rsi < 60:
            if last_close < recent_low and last_close < e20 and last_close < e50:
                nearest_res = find_nearest_zone(resistance_zones or [], last_close, "above")
                near, multiple = is_near_zone(nearest_res, max_atr_multiple=1.0)
                debug["nearest_resistance"] = nearest_res
                debug["nearest_resistance_near"] = near
                debug["nearest_resistance_dist_atr"] = multiple
                print(f"[DEBUG] 5M SHORT nearest_res={nearest_res}, near={near}, dist_atr={multiple}")
                if nearest_res and near:
                    print("[DEBUG] 5M SHORT trigger conditions + S/R met")
                    signal = {
                        "side": "SELL",
                        "timestamp": last_candle["ts"],
                        "price": last_close,
                        "atr": last_atr,
                        "sr_level": nearest_res["level"],
                        "sr_strength": nearest_res["strength"],
                        "sr_distance_atr": multiple,
                    }
                    debug["reason"] = "trigger_short_sr_ok"
                else:
                    print("[DEBUG] 5M SHORT: dobre wybicie, ale daleko od 1H oporu -> brak sygnału")
                    debug["reason"] = "short_far_from_resistance"
            else:
                print("[DEBUG] 5M SHORT: RSI ok, ale brak wybicia low / poniżej EMA")
                debug["reason"] = "short_no_breakout"
        else:
            print("[DEBUG] 5M SHORT: RSI pattern not met")
            debug["reason"] = "short_rsi_pattern_not_met"

    return signal, debug


# --------- Scoring + TP/SL ---------
def build_signal_with_risk(signal, bias):
    """
    Dodaje SL/TP i score.
    """
    atr_val = signal["atr"]
    price = signal["price"]
    side = signal["side"]

    if atr_val is None:
        print("[DEBUG] build_signal_with_risk: ATR is None -> no signal")
        return None

    rr = 2.0  # RR 1:2

    if side == "BUY":
        sl = price - atr_val
        tp = price + atr_val * 2
    else:  # SELL
        sl = price + atr_val
        tp = price - atr_val * 2

    base_score = 60
    rr_bonus = 10 if rr >= 2.0 else 0

    sr_bonus = 0
    if "sr_strength" in signal:
        sr_bonus += min(signal["sr_strength"] * 2, 10)
    if "sr_distance_atr" in signal and signal["sr_distance_atr"] is not None:
        if signal["sr_distance_atr"] <= 0.5:
            sr_bonus += 5

    final_score = base_score + rr_bonus + sr_bonus

    print(f"[DEBUG] build_signal_with_risk: side={side}, price={price}, sl={sl}, tp={tp}, rr={rr}, base={base_score}, sr_bonus={sr_bonus}, score={final_score}")

    signal_out = {
        "instrument": INSTRUMENT,
        "side": side,
        "timestamp": signal["timestamp"],   # czas świecy 5m
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "score": final_score,
        "bias": bias,
        "timeframe_entry": TIMEFRAME_5M,
    }

    if "sr_level" in signal:
        signal_out["sr_level"] = signal["sr_level"]
    if "sr_strength" in signal:
        signal_out["sr_strength"] = signal["sr_strength"]
    if "sr_distance_atr" in signal:
        signal_out["sr_distance_atr"] = signal["sr_distance_atr"]

    return signal_out


# --------- Zapis sygnału do tabeli sygnałów ---------
def persist_signal_to_dynamodb(signal_obj):
    print(f"[DEBUG] persist_signal_to_dynamodb: saving signal={signal_obj}")
    table = dynamodb.Table(SIGNALS_TABLE_NAME)

    pk_value = f"{signal_obj['instrument']}#signal"
    ts_value = signal_obj["timestamp"]

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    item = {
        "pk": pk_value,
        "ts": ts_value,
        "instrument": signal_obj["instrument"],
        "side": signal_obj["side"],
        "bias": signal_obj["bias"],
        "timeframe_entry": signal_obj["timeframe_entry"],
        "entry_price": _to_decimal(signal_obj["entry_price"]),
        "sl": _to_decimal(signal_obj["sl"]),
        "tp": _to_decimal(signal_obj["tp"]),
        "rr": _to_decimal(signal_obj["rr"]),
        "score": decimal.Decimal(signal_obj["score"]),
        "created_at": now_iso,
    }

    if "sr_level" in signal_obj:
        item["sr_level"] = _to_decimal(signal_obj["sr_level"])
    if "sr_strength" in signal_obj:
        item["sr_strength"] = decimal.Decimal(signal_obj["sr_strength"])
    if "sr_distance_atr" in signal_obj and signal_obj["sr_distance_atr"] is not None:
        item["sr_distance_atr"] = _to_decimal(signal_obj["sr_distance_atr"])

    print(f"[DEBUG] DynamoDB put_item to {SIGNALS_TABLE_NAME}: {item}")
    table.put_item(Item=item)
    print("[DEBUG] persist_signal_to_dynamodb: put_item done")


# --------- Zapis debug info do osobnej tabeli ---------
def persist_debug_to_dynamodb(debug_obj):
    """
    Zapisuje pełny snapshot obliczeń do tabeli debugowej.
    """
    print("[DEBUG] persist_debug_to_dynamodb: saving debug snapshot")
    table = dynamodb.Table(DEBUG_TABLE_NAME)

    instrument = debug_obj.get("instrument", INSTRUMENT)
    pk_value = f"{instrument}#debug"

    # ts – preferuj timestamp świecy 5m, jeśli jest
    ts = None
    m5 = debug_obj.get("m5") or {}
    ts = m5.get("ts")
    if not ts:
        ts = debug_obj.get("run_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    item = {
        "pk": pk_value,
        "ts": ts,
        "debug": _to_decimal_deep(debug_obj),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print(f"[DEBUG] DynamoDB put_item to {DEBUG_TABLE_NAME}: {item}")
    table.put_item(Item=item)
    print("[DEBUG] persist_debug_to_dynamodb: put_item done")


# --------- Główny handler Lambdy ---------
def handler(event, context):
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[DEBUG] Lambda started, event={event}")
    print(f"[DEBUG] Using tables: OHLC={OHLC_TABLE_NAME}, SIGNALS={SIGNALS_TABLE_NAME}, DEBUG={DEBUG_TABLE_NAME}, instrument={INSTRUMENT}")

    debug_info = {
        "instrument": INSTRUMENT,
        "run_at": run_at,
        "event": event,
    }

    daily_pk = f"{INSTRUMENT}#{TIMEFRAME_1D}"
    hourly_pk = f"{INSTRUMENT}#{TIMEFRAME_1H}"
    m5_pk = f"{INSTRUMENT}#{TIMEFRAME_5M}"

    daily_candles = fetch_candles(daily_pk, limit=300)
    hourly_candles = fetch_candles(hourly_pk, limit=300)
    m5_candles = fetch_candles(m5_pk, limit=300)

    # 1D bias (na zamkniętej świecy)
    bias, daily_debug = determine_daily_bias(daily_candles)
    debug_info["daily"] = daily_debug
    debug_info["daily"]["bias"] = bias

    print(f"[DEBUG] Daily bias={bias}")
    if bias == "NONE":
        debug_info["final"] = {
            "has_signal": False,
            "reason": "No daily trend (BIAS = NONE).",
            "daily_rsi": daily_debug.get("rsi14"),
            "daily_close": daily_debug.get("close"),
            "daily_ema20": daily_debug.get("ema20"),
            "daily_ema50": daily_debug.get("ema50"),
        }
        persist_debug_to_dynamodb(debug_info)
        print("[DEBUG] Exiting: BIAS = NONE -> no signal, nothing saved")
        return {
            "statusCode": 200,
            "body": json.dumps(debug_info["final"]),
        }

    # S/R z 1H
    support_zones, resistance_zones = calc_sr_zones(hourly_candles)
    debug_info["sr"] = {
        "support_zones": support_zones,
        "resistance_zones": resistance_zones,
    }

    # 1H filtr (na zamkniętej świecy)
    setup_ok, h1_debug = check_1h_setup(hourly_candles, bias)
    debug_info["h1"] = h1_debug
    debug_info["h1"]["setup_ok"] = setup_ok

    print(f"[DEBUG] 1H setup_ok={setup_ok}")
    if not setup_ok:
        debug_info["final"] = {
            "has_signal": False,
            "reason": f"1H conditions not met: {h1_debug.get('reason', 'unknown')}",
            "h1_rsi": h1_debug.get("rsi14"),
            "h1_close": h1_debug.get("close"),
            "h1_ema": h1_debug.get("ema"),
            "h1_last_low": h1_debug.get("last_low"),
            "h1_prev_low": h1_debug.get("prev_low"),
            "h1_last_high": h1_debug.get("last_high"),
            "h1_prev_high": h1_debug.get("prev_high"),
        }
        persist_debug_to_dynamodb(debug_info)
        print("[DEBUG] Exiting: 1H conditions not met")
        return {
            "statusCode": 200,
            "body": json.dumps(debug_info["final"]),
        }

    # 5M trigger + S/R
    raw_signal, m5_debug = check_5m_trigger(m5_candles, bias, support_zones, resistance_zones)
    debug_info["m5"] = m5_debug
    debug_info["m5"]["has_raw_signal"] = raw_signal is not None

    print(f"[DEBUG] raw_signal={raw_signal}")
    if raw_signal is None:
        m5_reason = m5_debug.get("reason", "unknown")

        nearest_support = m5_debug.get("nearest_support") or {}
        nearest_resistance = m5_debug.get("nearest_resistance") or {}

        debug_info["final"] = {
            "has_signal": False,
            "reason": f"No 5M trigger: {m5_reason}",

            # dodatkowe pola do tabelki
            "m5_last_close": m5_debug.get("last_close"),
            "m5_last_rsi": m5_debug.get("last_rsi"),
            "m5_prev_rsi": m5_debug.get("prev_rsi"),
            "m5_atr": m5_debug.get("atr"),
            "m5_ema20": m5_debug.get("ema20"),
            "m5_ema50": m5_debug.get("ema50"),
            "m5_recent_high": m5_debug.get("recent_high"),
            "m5_recent_low": m5_debug.get("recent_low"),

            "m5_nearest_support_level": nearest_support.get("level"),
            "m5_nearest_support_strength": nearest_support.get("strength"),
            "m5_nearest_support_dist_atr": m5_debug.get("nearest_support_dist_atr"),

            "m5_nearest_resistance_level": nearest_resistance.get("level"),
            "m5_nearest_resistance_strength": nearest_resistance.get("strength"),
            "m5_nearest_resistance_dist_atr": m5_debug.get("nearest_resistance_dist_atr"),
        }
        persist_debug_to_dynamodb(debug_info)
        print("[DEBUG] Exiting: No 5M trigger")
        return {
            "statusCode": 200,
            "body": json.dumps(debug_info["final"]),
        }

    # RR, SL/TP, score
    full_signal = build_signal_with_risk(raw_signal, bias)
    print(f"[DEBUG] full_signal={full_signal}")
    debug_info["signal"] = full_signal

    if full_signal is None or full_signal["score"] < 70:
        debug_info["final"] = {
            "has_signal": False,
            "reason": "Signal score too low.",
            "score": None if full_signal is None else full_signal["score"],
        }
        persist_debug_to_dynamodb(debug_info)
        print("[DEBUG] Exiting: signal is None or score too low")
        return {
            "statusCode": 200,
            "body": json.dumps(debug_info["final"]),
        }

    # Zapis sygnału
    persist_signal_to_dynamodb(full_signal)

    debug_info["final"] = {
        "has_signal": True,
        "reason": "Signal generated and stored.",
    }
    persist_debug_to_dynamodb(debug_info)

    print("[DEBUG] Signal generated and stored successfully")
    return {
        "statusCode": 200,
        "body": json.dumps({
            "signal": full_signal,
            "reason": "Signal generated and stored.",
        }),
    }
