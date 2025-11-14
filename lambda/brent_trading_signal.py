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
    # bezpieczna konwersja float -> Decimal
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


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


# --------- Logika trendu na 1D ---------
def determine_daily_bias(daily_candles):
    """
    Zwraca 'LONG', 'SHORT' albo 'NONE'.
    """
    print(f"[DEBUG] determine_daily_bias: candles_1d_count={len(daily_candles)}")
    closes = [c["close"] for c in daily_candles]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    rsi14 = rsi(closes, 14)

    if len(closes) == 0:
        print("[DEBUG] determine_daily_bias: no closes -> BIAS=NONE")
        return "NONE"

    last_idx = len(closes) - 1
    c = closes[last_idx]
    e20 = ema20[last_idx]
    e50 = ema50[last_idx]
    e200 = ema200[last_idx]
    r = rsi14[last_idx]

    print(f"[DEBUG] 1D last close={c}, ema20={e20}, ema50={e50}, ema200={e200}, rsi14={r}")

    # Jeśli za mało danych na EMA200/EMA50 itd.
    if e20 is None or e50 is None or e200 is None or r is None:
        print("[DEBUG] determine_daily_bias: some indicator is None -> BIAS=NONE")
        return "NONE"

    # proste reguły trendu
    if c > e200 and e20 > e50 and r > 55:
        print("[DEBUG] determine_daily_bias: conditions for LONG met")
        return "LONG"
    elif c < e200 and e20 < e50 and r < 45:
        print("[DEBUG] determine_daily_bias: conditions for SHORT met")
        return "SHORT"
    else:
        print("[DEBUG] determine_daily_bias: no strong trend -> BIAS=NONE")
        return "NONE"


# --------- Logika 1H ---------
def check_1h_setup(hourly_candles, bias):
    """
    Sprawdza, czy 1H wspiera bias z 1D.
    Zwraca True/False.
    """
    print(f"[DEBUG] check_1h_setup: candles_1h_count={len(hourly_candles)}, bias={bias}")
    if len(hourly_candles) < 5:
        print("[DEBUG] check_1h_setup: not enough 1H candles")
        return False

    closes = [c["close"] for c in hourly_candles]
    ema50_h = ema(closes, 50)
    rsi_h = rsi(closes, 14)
    last_idx = len(closes) - 1

    c = closes[last_idx]
    e50 = ema50_h[last_idx]
    r = rsi_h[last_idx]

    print(f"[DEBUG] 1H last close={c}, ema50={e50}, rsi14={r}")

    if e50 is None or r is None:
        print("[DEBUG] check_1h_setup: ema50 or rsi is None")
        return False

    # prosta detekcja HL / LH – bardzo uproszczona
    last_low = min(hourly_candles[last_idx - 2:last_idx + 1], key=lambda x: x["low"])["low"]
    prev_low = min(hourly_candles[last_idx - 3:last_idx], key=lambda x: x["low"])["low"]

    last_high = max(hourly_candles[last_idx - 2:last_idx + 1], key=lambda x: x["high"])["high"]
    prev_high = max(hourly_candles[last_idx - 3:last_idx], key=lambda x: x["high"])["high"]

    print(f"[DEBUG] 1H last_low={last_low}, prev_low={prev_low}, last_high={last_high}, prev_high={prev_high}")

    if bias == "LONG":
        ok = c > e50 and r > 50 and last_low > prev_low
        print(f"[DEBUG] check_1h_setup LONG -> {ok}")
        return ok
    elif bias == "SHORT":
        ok = c < e50 and r < 50 and last_high < prev_high
        print(f"[DEBUG] check_1h_setup SHORT -> {ok}")
        return ok
    else:
        print("[DEBUG] check_1h_setup: bias NONE -> False")
        return False


# --------- Logika wejścia na 5M ---------
def check_5m_trigger(candles_5m, bias):
    """
    Uproszczony trigger:
    - dla LONG: RSI wychodzi z wyprzedania, wybicie lokalnego szczytu
    - dla SHORT: RSI wychodzi z wykupienia, wybicie lokalnego dołka
    Zwraca dict z informacją o sygnale lub None.
    """
    print(f"[DEBUG] check_5m_trigger: candles_5m_count={len(candles_5m)}, bias={bias}")
    if len(candles_5m) < 30:
        print("[DEBUG] check_5m_trigger: not enough 5M candles")
        return None

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

    print(f"[DEBUG] 5M last candle={last_candle}")
    print(f"[DEBUG] 5M last_close={last_close}, rsi={last_rsi}, atr={last_atr}, ema20={e20}, ema50={e50}")

    if last_rsi is None or last_atr is None or e20 is None or e50 is None:
        print("[DEBUG] check_5m_trigger: some indicator is None -> no signal")
        return None

    # lokalne HH/LL na ostatnich kilku świecach
    window = 5
    recent_high = max(highs[last_idx - window:last_idx])
    recent_low = min(lows[last_idx - window:last_idx])

    print(f"[DEBUG] 5M recent_high={recent_high}, recent_low={recent_low}")

    signal = None

    if bias == "LONG":
        prev_rsi = rsi_5[last_idx - 1]
        print(f"[DEBUG] 5M LONG prev_rsi={prev_rsi}, last_rsi={last_rsi}")
        if prev_rsi is not None and prev_rsi < 40 and last_rsi > 40:
            # wybicie lokalnego high + powyżej EMA
            if last_close > recent_high and last_close > e20 and last_close > e50:
                print("[DEBUG] 5M LONG trigger conditions met")
                signal = {
                    "side": "BUY",
                    "timestamp": last_candle["ts"],
                    "price": last_close,
                    "atr": last_atr,
                }
            else:
                print("[DEBUG] 5M LONG: RSI ok, ale brak wybicia high / powyżej EMA")
        else:
            print("[DEBUG] 5M LONG: RSI pattern not met")

    elif bias == "SHORT":
        prev_rsi = rsi_5[last_idx - 1]
        print(f"[DEBUG] 5M SHORT prev_rsi={prev_rsi}, last_rsi={last_rsi}")
        if prev_rsi is not None and prev_rsi > 60 and last_rsi < 60:
            # wybicie lokalnego low + poniżej EMA
            if last_close < recent_low and last_close < e20 and last_close < e50:
                print("[DEBUG] 5M SHORT trigger conditions met")
                signal = {
                    "side": "SELL",
                    "timestamp": last_candle["ts"],
                    "price": last_close,
                    "atr": last_atr,
                }
            else:
                print("[DEBUG] 5M SHORT: RSI ok, ale brak wybicia low / poniżej EMA")
        else:
            print("[DEBUG] 5M SHORT: RSI pattern not met")

    return signal


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

    base_score = 70
    rr_bonus = 10 if rr >= 2.0 else 0
    final_score = base_score + rr_bonus

    print(f"[DEBUG] build_signal_with_risk: side={side}, price={price}, sl={sl}, tp={tp}, rr={rr}, score={final_score}")

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
    return signal_out


# --------- Zapis sygnału do nowej tabeli ---------
def persist_signal_to_dynamodb(signal_obj):
    """
    Zapisuje sygnał do tabeli `brent_signals`.
    """
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

    print(f"[DEBUG] DynamoDB put_item to {SIGNALS_TABLE_NAME}: {item}")
    table.put_item(Item=item)
    print("[DEBUG] persist_signal_to_dynamodb: put_item done")


# --------- Główny handler Lambdy ---------
def handler(event, context):
    print(f"[DEBUG] Lambda started, event={event}")
    print(f"[DEBUG] Using tables: OHLC={OHLC_TABLE_NAME}, SIGNALS={SIGNALS_TABLE_NAME}, instrument={INSTRUMENT}")

    # 1. pobierz świece
    daily_pk = f"{INSTRUMENT}#{TIMEFRAME_1D}"
    hourly_pk = f"{INSTRUMENT}#{TIMEFRAME_1H}"
    m5_pk = f"{INSTRUMENT}#{TIMEFRAME_5M}"

    daily_candles = fetch_candles(daily_pk, limit=300)
    hourly_candles = fetch_candles(hourly_pk, limit=300)
    m5_candles = fetch_candles(m5_pk, limit=300)

    # 2. bias z 1D
    bias = determine_daily_bias(daily_candles)
    print(f"[DEBUG] Daily bias={bias}")
    if bias == "NONE":
        print("[DEBUG] Exiting: BIAS = NONE -> no signal, nothing saved")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "signal": None,
                "reason": "No daily trend (BIAS = NONE).",
            }),
        }

    # 3. setup na 1H
    setup_ok = check_1h_setup(hourly_candles, bias)
    print(f"[DEBUG] 1H setup_ok={setup_ok}")
    if not setup_ok:
        print("[DEBUG] Exiting: 1H conditions not met")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "signal": None,
                "reason": "1H conditions not met.",
            }),
        }

    # 4. trigger na 5M
    raw_signal = check_5m_trigger(m5_candles, bias)
    print(f"[DEBUG] raw_signal={raw_signal}")
    if raw_signal is None:
        print("[DEBUG] Exiting: No 5M trigger")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "signal": None,
                "reason": "No 5M trigger.",
            }),
        }

    # 5. RR, SL/TP, score
    full_signal = build_signal_with_risk(raw_signal, bias)
    print(f"[DEBUG] full_signal={full_signal}")
    if full_signal is None or full_signal["score"] < 70:
        print("[DEBUG] Exiting: signal is None or score too low")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "signal": None,
                "reason": "Signal score too low.",
            }),
        }

    # 6. Zapis sygnału do tabeli brent_signals
    persist_signal_to_dynamodb(full_signal)

    # 7. Zwróć sygnał w odpowiedzi
    print("[DEBUG] Signal generated and stored successfully")
    return {
        "statusCode": 200,
        "body": json.dumps({
            "signal": full_signal,
            "reason": "Signal generated and stored.",
        }),
    }
