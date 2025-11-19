import argparse
import csv
import json
import hashlib
import base64
from decimal import Decimal


# ============================================================
#                  Pomocnicze konwersje / hash
# ============================================================

def _from_decimal_deep(obj):
    """Helper – konwersja Decimal -> float (na wszelki wypadek)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [_from_decimal_deep(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _from_decimal_deep(v) for k, v in obj.items()}
    return obj


def compute_strategy_hash(config: dict) -> str:
    """
    Krótki hash strategii na podstawie configu:
    SHA-256 + base32, obcięte do ~8 znaków.
    """
    config_for_hash = dict(config)
    canon = json.dumps(config_for_hash, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).digest()
    short = base64.b32encode(h[:5]).decode("utf-8").rstrip("=")
    return short


def _parse_field_value(v):
    """
    Parser wartości z CSV:
    - "" / "none" / "null" -> None
    - liczby -> float
    - "true"/"false" -> bool
    - reszta -> string (np. UP/DOWN)
    """
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None

    # bool
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # liczba
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return float(int(s))
    except Exception:
        return s


# ============================================================
#         Parsowanie snapshotu z CSV → daily / h1 / m5 / SR
# ============================================================

def split_parts_from_row(row: dict):
    """
    Zakładamy kolumny:
      daily_xxx, h1_xxx, m5_xxx
    lub z kropką: daily.xxx itd.
    """
    daily = {}
    h1 = {}
    m5 = {}

    for k, v in row.items():
        if k.startswith("daily_") or k.startswith("daily."):
            key = k.split("_", 1)[-1] if "_" in k else k.split(".", 1)[-1]
            if key in ("ts", "time", "timestamp"):
                daily["ts"] = v
            else:
                daily[key] = _parse_field_value(v)

        elif k.startswith("h1_") or k.startswith("h1."):
            key = k.split("_", 1)[-1] if "_" in k else k.split(".", 1)[-1]
            if key in ("ts", "time", "timestamp"):
                h1["ts"] = v
            else:
                h1[key] = _parse_field_value(v)

        elif k.startswith("m5_") or k.startswith("m5."):
            key = k.split("_", 1)[-1] if "_" in k else k.split(".", 1)[-1]
            if key in ("ts", "time", "timestamp"):
                m5["ts"] = v
            else:
                m5[key] = _parse_field_value(v)

    # fallback – jeśli nie było prefiksów, spróbuj kilka nazw top-level
    if "ts" not in m5 and "ts" in row:
        m5["ts"] = row["ts"]
    if "close" not in m5 and "close" in row:
        m5["close"] = _parse_field_value(row["close"])

    return daily, h1, m5


def parse_sr_multi_from_row(row: dict):
    """
    Odczytujemy sr_multi_json (pełny dict z support_zones/resistance_zones).
    """
    sr_multi = {"support_zones": [], "resistance_zones": []}

    sr_multi_raw = row.get("sr_multi_json")
    if not sr_multi_raw:
        return sr_multi

    s = sr_multi_raw.strip()

    # CSV może otoczyć całość dodatkowymi cudzysłowami
    if s and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]

    # podwójne "" -> pojedyncze "
    s = s.replace('""', '"')

    try:
        data = json.loads(s)
        if isinstance(data, dict):
            sr_multi["support_zones"] = data.get("support_zones") or []
            sr_multi["resistance_zones"] = data.get("resistance_zones") or []
    except Exception as e:
        print(f"[WARN] failed to parse sr_multi_json: {e}")

    return sr_multi


# ============================================================
#                     DAILY – prosty filtr
# ============================================================

def is_daily_ok(daily_part: dict, cfg_daily: dict):
    """
    Prosty filtr DAILY (opcjonalny).
    Zwraca: (ok: bool, reason: str)
    """
    if not cfg_daily or not cfg_daily.get("enabled", True):
        return True, "DAILY_DISABLED"

    if not isinstance(daily_part, dict) or not daily_part:
        return False, "NO_DAILY_PART"

    atr_pct = daily_part.get("atr14_pct_rank")
    slope_pct = daily_part.get("ema50_slope_pct")

    if atr_pct is None and slope_pct is None:
        return False, "MISSING_DAILY_FIELDS"

    max_atr_pct = cfg_daily.get("max_atr_pct")
    max_abs_slope_pct = cfg_daily.get("max_abs_slope_pct")

    if max_atr_pct is not None and atr_pct is not None and atr_pct > max_atr_pct:
        return False, "DAILY_ATR_TOO_HIGH"

    if max_abs_slope_pct is not None and slope_pct is not None:
        if abs(slope_pct) > max_abs_slope_pct:
            return False, "DAILY_TREND_TOO_STRONG"

    return True, "OK_DAILY"


# ============================================================
#               H1 – wykrywanie impulsu
# ============================================================

def check_h1_impulse(h1_part: dict, cfg_h1: dict):
    """
    Sprawdza, czy jest sensowny impuls na H1.
    Zwraca: (ok: bool, direction: 'UP'/'DOWN'|None, reason: str)
    """
    if not cfg_h1 or not cfg_h1.get("enabled", True):
        return True, None, "H1_DISABLED"

    if not isinstance(h1_part, dict) or not h1_part:
        return False, None, "NO_H1_PART"

    direction = h1_part.get("impulse_direction")
    bars = h1_part.get("impulse_bars")
    size_atr = h1_part.get("impulse_size_atr")
    size_pct = h1_part.get("impulse_size_pct")

    if direction not in ("UP", "DOWN"):
        return False, None, "NO_IMPULSE_DIRECTION"

    if size_atr is None or bars is None:
        return False, None, "MISSING_IMPULSE_FIELDS"

    min_impulse_atr = cfg_h1.get("min_impulse_atr", 0.5)
    max_impulse_atr = cfg_h1.get("max_impulse_atr", 5.0)
    min_impulse_bars = cfg_h1.get("min_impulse_bars", 3)
    max_impulse_bars = cfg_h1.get("max_impulse_bars", 30)
    max_impulse_pct = cfg_h1.get("max_impulse_pct")  # opcjonalnie

    if size_atr < min_impulse_atr:
        return False, None, "IMPULSE_TOO_SMALL_ATR"
    if size_atr > max_impulse_atr:
        return False, None, "IMPULSE_TOO_LARGE_ATR"

    if bars < min_impulse_bars:
        return False, None, "IMPULSE_TOO_SHORT"
    if bars > max_impulse_bars:
        return False, None, "IMPULSE_TOO_LONG"

    if max_impulse_pct is not None and size_pct is not None and size_pct > max_impulse_pct:
        return False, None, "IMPULSE_TOO_LARGE_PCT"

    return True, direction, "OK_H1_IMPULSE"


# ============================================================
#      M5 – trigger w kanale / fladze po impulsie
# ============================================================

def check_flag_trigger(m5_part: dict, cfg_flag: dict, impulse_direction: str):
    """
    Wejście w kanale flagi (mean-reversion wewnątrz flagi).
    Zwraca: (raw_signal: dict | None, debug: dict)
    """
    debug = {
        "enabled": bool(cfg_flag and cfg_flag.get("enabled", True)),
        "reason": None,
        "side": None,
    }

    if not cfg_flag or not cfg_flag.get("enabled", True):
        debug["reason"] = "M5_FLAG_DISABLED"
        return None, debug

    if not isinstance(m5_part, dict) or not m5_part:
        debug["reason"] = "NO_M5_PART"
        return None, debug

    candles_count = m5_part.get("candles_count", 0)
    min_candles = cfg_flag.get("min_candles", 30)
    if candles_count is None or candles_count < min_candles:
        debug["reason"] = "NOT_ENOUGH_M5_CANDLES"
        return None, debug

    flag_active = m5_part.get("flag_active")
    flag_lower = m5_part.get("flag_lower")
    flag_upper = m5_part.get("flag_upper")
    flag_width_atr = m5_part.get("flag_width_atr")
    flag_age_bars = m5_part.get("flag_age_bars")
    flag_position = m5_part.get("flag_position")
    flag_slope = m5_part.get("flag_slope")
    flag_impulse_dir = m5_part.get("flag_impulse_direction")

    price = m5_part.get("close")
    atr_pct = m5_part.get("atr14_pct_rank")
    atr_val = m5_part.get("atr14")
    rsi_pct = m5_part.get("rsi14_pct_rank")

    debug.update({
        "flag_active": flag_active,
        "flag_lower": flag_lower,
        "flag_upper": flag_upper,
        "flag_width_atr": flag_width_atr,
        "flag_age_bars": flag_age_bars,
        "flag_position": flag_position,
        "flag_slope": flag_slope,
        "flag_impulse_dir": flag_impulse_dir,
        "price": price,
        "atr_pct": atr_pct,
        "atr_val": atr_val,
        "rsi_pct": rsi_pct,
    })

    if not flag_active:
        debug["reason"] = "FLAG_NOT_ACTIVE"
        return None, debug

    if flag_lower is None or flag_upper is None or flag_width_atr is None:
        debug["reason"] = "FLAG_MISSING_LEVELS"
        return None, debug

    if flag_upper <= flag_lower:
        debug["reason"] = "FLAG_INVALID_BOUNDS"
        return None, debug

    if flag_age_bars is None:
        debug["reason"] = "FLAG_MISSING_AGE"
        return None, debug

    if flag_position is None:
        debug["reason"] = "FLAG_MISSING_POSITION"
        return None, debug

    # kierunek impulsu
    eff_impulse_dir = flag_impulse_dir or impulse_direction
    if eff_impulse_dir not in ("UP", "DOWN"):
        debug["reason"] = "NO_VALID_IMPULSE_DIR"
        return None, debug

    # filtry flagi
    min_flag_age = cfg_flag.get("min_flag_age_bars", 20)
    max_flag_age = cfg_flag.get("max_flag_age_bars", 200)
    min_flag_width_atr = cfg_flag.get("min_flag_width_atr", 0.3)
    max_flag_width_atr = cfg_flag.get("max_flag_width_atr", 2.5)
    max_m5_atr_pct = cfg_flag.get("max_m5_atr_pct", 100.0)

    if flag_age_bars < min_flag_age:
        debug["reason"] = "FLAG_TOO_YOUNG"
        return None, debug
    if flag_age_bars > max_flag_age:
        debug["reason"] = "FLAG_TOO_OLD"
        return None, debug

    if flag_width_atr < min_flag_width_atr:
        debug["reason"] = "FLAG_TOO_NARROW"
        return None, debug
    if flag_width_atr > max_flag_width_atr:
        debug["reason"] = "FLAG_TOO_WIDE"
        return None, debug

    if atr_pct is not None and atr_pct > max_m5_atr_pct:
        debug["reason"] = "M5_ATR_TOO_HIGH"
        return None, debug

    # pozycja w kanale
    max_buy_flag_pos = cfg_flag.get("max_buy_flag_pos", 0.3)
    min_sell_flag_pos = cfg_flag.get("min_sell_flag_pos", 0.7)

    side = None

    if eff_impulse_dir == "UP":
        # szukamy BUY przy dolnej krawędzi kanału
        if flag_position <= max_buy_flag_pos:
            side = "BUY"
        else:
            debug["reason"] = "FLAG_POSITION_NOT_GOOD_FOR_BUY"
            return None, debug

    elif eff_impulse_dir == "DOWN":
        # szukamy SELL przy górnej krawędzi kanału
        if flag_position >= min_sell_flag_pos:
            side = "SELL"
        else:
            debug["reason"] = "FLAG_POSITION_NOT_GOOD_FOR_SELL"
            return None, debug

    # ewentualne proste RSI
    if side == "BUY":
        max_rsi_buy = cfg_flag.get("max_rsi_buy_pct", 60.0)
        if rsi_pct is not None and rsi_pct > max_rsi_buy:
            debug["reason"] = "RSI_NOT_LOW_FOR_BUY"
            return None, debug
    elif side == "SELL":
        min_rsi_sell = cfg_flag.get("min_rsi_sell_pct", 40.0)
        if rsi_pct is not None and rsi_pct < min_rsi_sell:
            debug["reason"] = "RSI_NOT_HIGH_FOR_SELL"
            return None, debug

    debug["side"] = side
    debug["reason"] = "OK_FLAG_TRIGGER"

    raw_signal = {
        "side": side,
        "timestamp": m5_part.get("ts"),
        "price": price,
        "impulse_direction": eff_impulse_dir,
        "flag_lower": flag_lower,
        "flag_upper": flag_upper,
        "flag_width_atr": flag_width_atr,
        "flag_age_bars": flag_age_bars,
        "flag_position": flag_position,
        "flag_slope": flag_slope,
    }

    return raw_signal, debug


# ============================================================
#      Pomocnicze – ATR / SR_MULTI
# ============================================================

def _compute_atr_for_risk(cfg_risk, raw_signal, h1_part, m5_part):
    atr_source = (cfg_risk.get("atr_source") or "h1").lower()
    atr_m5 = None
    if isinstance(m5_part, dict):
        atr_m5 = m5_part.get("atr14")
    atr_h1 = None
    if isinstance(h1_part, dict):
        atr_h1 = h1_part.get("atr14")

    if atr_source == "h1":
        return atr_h1 or atr_m5
    elif atr_source == "m5":
        return atr_m5 or atr_h1
    else:
        return atr_h1 or atr_m5


# ============================================================
#               RISK / SL+TP – flaga + SR
# ============================================================

def build_flag_signal_with_risk(
    raw_signal: dict,
    cfg_risk: dict,
    instrument: str,
    strategy_hash: str,
    strategy_name: str,
    config_json: str,
    sr_multi: dict | None,
    h1_part: dict | None,
    m5_part: dict | None,
):
    """
    Ustalenie SL/TP na bazie flagi + najbliższych SR.
    """
    if not cfg_risk or not cfg_risk.get("enabled", True):
        raw_signal["_reject_reason"] = "RISK_DISABLED"
        return None

    side = raw_signal["side"]
    price = raw_signal["price"]
    ts = raw_signal["timestamp"]

    if price is None:
        raw_signal["_reject_reason"] = "NO_PRICE"
        return None

    atr_val = _compute_atr_for_risk(cfg_risk, raw_signal, h1_part, m5_part)
    if atr_val is None or atr_val <= 0:
        raw_signal["_reject_reason"] = "NO_ATR"
        return None

    flag_lower = raw_signal.get("flag_lower")
    flag_upper = raw_signal.get("flag_upper")
    flag_width_atr = raw_signal.get("flag_width_atr")

    if flag_lower is None or flag_upper is None or flag_width_atr is None:
        raw_signal["_reject_reason"] = "NO_FLAG_LEVELS"
        return None

    # --- SL: na bazie flagi ---
    sl_cfg = cfg_risk.get("sl", {})
    sl_buffer_flag_atr = sl_cfg.get("buffer_flag_atr", 0.1)
    max_sl_atr = sl_cfg.get("max_sl_atr")  # opcjonalnie

    min_sl_distance_atr = cfg_risk.get("min_sl_distance_atr", 0.2)

    if side == "BUY":
        sl = flag_lower - sl_buffer_flag_atr * atr_val
        sl_dist_abs = price - sl
    else:
        sl = flag_upper + sl_buffer_flag_atr * atr_val
        sl_dist_abs = sl - price

    if sl_dist_abs <= 0:
        raw_signal["_reject_reason"] = "SL_ON_WRONG_SIDE"
        return None

    sl_dist_atr = sl_dist_abs / atr_val

    # minimalna odległość SL w ATR – jeśli zbyt blisko, przesuwamy go dalej
    if sl_dist_atr < min_sl_distance_atr:
        needed_abs = min_sl_distance_atr * atr_val
        if side == "BUY":
            sl = price - needed_abs
        else:
            sl = price + needed_abs
        sl_dist_abs = abs(price - sl)
        sl_dist_atr = sl_dist_abs / atr_val

    if max_sl_atr is not None and sl_dist_atr > max_sl_atr:
        raw_signal["_reject_reason"] = "SL_TOO_FAR_ATR"
        raw_signal["_reject_sl_atr"] = sl_dist_atr
        return None

    # --- TP: flaga + najbliższe SR (z m5_part) ---
    tp_cfg = cfg_risk.get("tp", {})
    use_flag_tp = tp_cfg.get("use_flag", True)
    use_sr_tp = tp_cfg.get("use_sr", True)

    tp_candidates = []

    tp_flag_buffer_atr = tp_cfg.get("buffer_flag_atr", 0.2)
    tp_sr_buffer_atr = tp_cfg.get("buffer_sr_atr", 0.1)
    sr_max_tp_atr = tp_cfg.get("sr_max_tp_atr", 5.0)

    # 1) TP wg flagi
    if use_flag_tp:
        if side == "BUY":
            tp_flag = flag_upper - tp_flag_buffer_atr * atr_val
            if tp_flag > price:
                tp_candidates.append(("FLAG", tp_flag))
        else:
            tp_flag = flag_lower + tp_flag_buffer_atr * atr_val
            if tp_flag < price:
                tp_candidates.append(("FLAG", tp_flag))

    # 2) TP wg najbliższego SR (już policzonego w snapshotach)
    if use_sr_tp and isinstance(m5_part, dict):
        if side == "BUY":
            sr_level = m5_part.get("nearest_resistance_level")
            sr_dist_atr = m5_part.get("nearest_resistance_dist_atr")
            if sr_level is not None and sr_dist_atr is not None and sr_dist_atr <= sr_max_tp_atr:
                tp_sr = sr_level - tp_sr_buffer_atr * atr_val
                if tp_sr > price:
                    tp_candidates.append(("SR", tp_sr))
        else:
            sr_level = m5_part.get("nearest_support_level")
            sr_dist_atr = m5_part.get("nearest_support_dist_atr")
            if sr_level is not None and sr_dist_atr is not None and sr_dist_atr <= sr_max_tp_atr:
                tp_sr = sr_level + tp_sr_buffer_atr * atr_val
                if tp_sr < price:
                    tp_candidates.append(("SR", tp_sr))

    if not tp_candidates:
        raw_signal["_reject_reason"] = "NO_VALID_TP_CANDIDATES"
        return None

    # wybieramy „bliższy” TP (konserwatywnie)
    if side == "BUY":
        tp_type, tp = min(tp_candidates, key=lambda x: x[1])  # najniższy > price
    else:
        tp_type, tp = max(tp_candidates, key=lambda x: x[1])  # najwyższy < price

    # sanity check
    if side == "BUY" and tp <= price:
        raw_signal["_reject_reason"] = "TP_BELOW_ENTRY"
        return None
    if side == "SELL" and tp >= price:
        raw_signal["_reject_reason"] = "TP_ABOVE_ENTRY"
        return None

    # --- RR / minimalne wymagania ---
    if side == "BUY":
        tp_abs = tp - price
        sl_abs = price - sl
    else:
        tp_abs = price - tp
        sl_abs = sl - price

    if sl_abs <= 0:
        raw_signal["_reject_reason"] = "SL_DISTANCE_ZERO"
        return None

    rr = tp_abs / sl_abs

    min_tp_abs = cfg_risk.get("min_tp_abs")
    if min_tp_abs is not None and tp_abs < min_tp_abs:
        raw_signal["_reject_reason"] = "TP_TOO_CLOSE_ABS"
        raw_signal["_reject_tp_abs"] = tp_abs
        return None

    min_rr = cfg_risk.get("min_rr", 1.5)
    if rr < min_rr:
        raw_signal["_reject_reason"] = "RR_TOO_LOW"
        raw_signal["_reject_rr"] = rr
        return None

    # prosty „score” – nie bawimy się w punkty M5
    base_score = cfg_risk.get("base_score", 60.0)
    score = base_score

    # bonus za RR
    if rr >= 2.0:
        score += 10.0
    if rr >= 3.0:
        score += 5.0

    # bonus za bliskość SR (im bliżej, tym lepiej)
    if isinstance(m5_part, dict):
        if side == "BUY":
            sr_dist_atr = m5_part.get("nearest_resistance_dist_atr")
        else:
            sr_dist_atr = m5_part.get("nearest_support_dist_atr")
        if sr_dist_atr is not None:
            if sr_dist_atr <= 1.0:
                score += 10.0
            elif sr_dist_atr <= 2.0:
                score += 5.0

    min_score = cfg_risk.get("min_score", 60.0)
    if score < min_score:
        raw_signal["_reject_reason"] = "GLOBAL_SCORE_TOO_LOW"
        raw_signal["_reject_score"] = score
        return None

    atr_h1 = None
    if isinstance(h1_part, dict):
        atr_h1 = h1_part.get("atr14")

    # kilka pól z M5 do debugowania
    m5_near_sup_level = None
    m5_near_sup_dist = None
    m5_near_sup_strength = None
    m5_near_res_level = None
    m5_near_res_dist = None
    m5_near_res_strength = None

    if isinstance(m5_part, dict):
        m5_near_sup_level = m5_part.get("nearest_support_level")
        m5_near_sup_dist = m5_part.get("nearest_support_dist_atr")
        m5_near_sup_strength = m5_part.get("nearest_support_strength")
        m5_near_res_level = m5_part.get("nearest_resistance_level")
        m5_near_res_dist = m5_part.get("nearest_resistance_dist_atr")
        m5_near_res_strength = m5_part.get("nearest_resistance_strength")

    full_signal = {
        "instrument": instrument,
        "strategy_hash": strategy_hash,
        "strategy_name": strategy_name,
        "strategy_config_json": config_json,
        "side": side,
        "timestamp": ts,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "score": score,
        "bias": "NEUTRAL_FLAG",
        "timeframe_entry": "5m",
        "risk_mode": f"FLAG_{tp_type}",
        "flag_lower": flag_lower,
        "flag_upper": flag_upper,
        "flag_width_atr": flag_width_atr,
        "flag_age_bars": raw_signal.get("flag_age_bars"),
        "flag_position": raw_signal.get("flag_position"),
        "flag_slope": raw_signal.get("flag_slope"),
        "impulse_direction": raw_signal.get("impulse_direction"),
        "atr_m5": m5_part.get("atr14") if isinstance(m5_part, dict) else None,
        "atr_higher_tf": atr_h1,
        "atr_used": atr_val,
        "sl_dist_atr": sl_dist_atr,
        "tp_type": tp_type,
        "nearest_support_level": m5_near_sup_level,
        "nearest_support_dist_atr": m5_near_sup_dist,
        "nearest_support_strength": m5_near_sup_strength,
        "nearest_resistance_level": m5_near_res_level,
        "nearest_resistance_dist_atr": m5_near_res_dist,
        "nearest_resistance_strength": m5_near_res_strength,
    }

    return full_signal


# ============================================================
#                       main() – offline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", required=True,
                        help="CSV ze snapshotami M5/D1/H1 (z flagą/impulsem)")
    parser.add_argument("--config_json", required=True,
                        help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="neutral_flag_signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None,
                        help="opcjonalny override instrumentu (np. BZ=F)")
    parser.add_argument("--debug_csv", default="neutral_flag_debug_log.csv",
                        help="plik CSV z powodami braku sygnału / debugiem")
    parser.add_argument("--stats_csv", default="neutral_flag_debug_stats.csv",
                        help="plik CSV z agregacją powodów SKIP")

    args = parser.parse_args()

    # config strategii
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "neutral_flag_strategy")
    strategy_hash = compute_strategy_hash(config)

    cfg_daily = config.get("daily", {})
    cfg_h1 = config.get("h1", {})
    cfg_m5_flag = config.get("m5_flag", {})
    cfg_risk = config.get("risk", {})

    print(f"Strategy name: {strategy_name}, hash={strategy_hash}")

    # snapshoty
    with open(args.snapshots_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} snapshots from {args.snapshots_csv}")

    signals = []
    debug_rows = []
    processed = 0
    generated = 0

    # cooldown po ostatnim sygnale (w barach M5)
    # Uwaga: technicznie "po sygnale", a nie "po SL" – żeby użyć "po SL",
    # trzeba by mieć wyniki trade'ów z backtestu.
    cooldown_bars = cfg_risk.get("cooldown_bars_after_signal", 0)
    last_signal_index_by_instrument = {}

    def add_debug(idx, snap_ts, price, stage, result, reason, extra=""):
        debug_rows.append({
            "snap_index": idx,
            "snap_ts": snap_ts,
            "price": price,
            "bias": "NEUTRAL_FLAG",
            "stage": stage,
            "result": result,
            "reason": reason,
            "extra": extra,
        })

    for i, row in enumerate(rows):
        processed += 1
        progress = (i + 1) / max(total, 1) * 100.0

        snap_instrument = row.get("instrument")
        instrument = args.instrument or snap_instrument or "BZ=F"
        run_at = row.get("run_at")

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = (
            m5_part.get("ts")
            or h1_part.get("ts")
            or daily_part.get("ts")
            or run_at
            or row.get("ts")
        )
        price = m5_part.get("close")

        print(
            f"[{progress:6.2f}%] snap_ts={snap_ts}, price={price}",
            flush=True,
        )

        # --- COOLDOWN po ostatnim sygnale (per instrument) ---
        if cooldown_bars and instrument in last_signal_index_by_instrument:
            last_idx = last_signal_index_by_instrument[instrument]
            if (i - last_idx) < cooldown_bars:
                reason = f"COOLDOWN_ACTIVE({i - last_idx}/{cooldown_bars})"
                add_debug(i, snap_ts, price, "COOLDOWN", "SKIP", reason, extra="")
                print(f"  -> SKIP: {reason}")
                continue

        # 1) DAILY – prosty filtr
        daily_ok, daily_reason = is_daily_ok(daily_part, cfg_daily)
        if not daily_ok:
            extra = (
                f"daily_atr_pct={daily_part.get('atr14_pct_rank')}, "
                f"daily_slope_pct={daily_part.get('ema50_slope_pct')}"
            )
            print(f"  -> SKIP: DAILY_FILTER ({daily_reason}; {extra})")
            add_debug(i, snap_ts, price, "DAILY_FILTER", "SKIP", daily_reason, extra)
            continue
        else:
            add_debug(i, snap_ts, price, "DAILY_FILTER", "OK", daily_reason, "")

        # 2) H1 – impuls
        h1_ok, impulse_dir, h1_reason = check_h1_impulse(h1_part, cfg_h1)
        if not h1_ok:
            extra = (
                f"imp_dir={h1_part.get('impulse_direction')}, "
                f"imp_bars={h1_part.get('impulse_bars')}, "
                f"imp_atr={h1_part.get('impulse_size_atr')}, "
                f"imp_pct={h1_part.get('impulse_size_pct')}"
            )
            print(f"  -> SKIP: H1_IMPULSE ({h1_reason}; {extra})")
            add_debug(i, snap_ts, price, "H1_IMPULSE", "SKIP", h1_reason, extra)
            continue
        else:
            add_debug(i, snap_ts, price, "H1_IMPULSE", "OK", h1_reason, f"imp_dir={impulse_dir}")

        # 3) M5 – flaga / kanał
        raw_signal, flag_debug = check_flag_trigger(m5_part, cfg_m5_flag, impulse_dir)
        if not raw_signal:
            reason = (flag_debug or {}).get("reason", "FLAG_TRIGGER_FAILED")
            extra = (
                f"flag_active={m5_part.get('flag_active')}, "
                f"flag_pos={m5_part.get('flag_position')}, "
                f"flag_width_atr={m5_part.get('flag_width_atr')}, "
                f"flag_age_bars={m5_part.get('flag_age_bars')}, "
                f"m5_rsi_pct={m5_part.get('rsi14_pct_rank')}, "
                f"m5_atr_pct={m5_part.get('atr14_pct_rank')}"
            )
            print(f"  -> SKIP: M5_FLAG_TRIGGER ({reason}; {extra})")
            add_debug(i, snap_ts, price, "M5_FLAG", "SKIP", reason, extra)
            continue
        else:
            extra = (
                f"side={raw_signal['side']}, "
                f"flag_pos={raw_signal.get('flag_position')}, "
                f"flag_width_atr={raw_signal.get('flag_width_atr')}"
            )
            add_debug(i, snap_ts, price, "M5_FLAG", "OK", "OK_FLAG_TRIGGER", extra)

        # 4) RISK / SL+TP – flaga + SR
        full_signal = build_flag_signal_with_risk(
            raw_signal=raw_signal,
            cfg_risk=cfg_risk,
            instrument=instrument,
            strategy_hash=strategy_hash,
            strategy_name=strategy_name,
            config_json=json.dumps(_from_decimal_deep(config), sort_keys=True),
            sr_multi=sr_multi,
            h1_part=h1_part,
            m5_part=m5_part,
        )

        if not full_signal:
            reject_reason = raw_signal.get("_reject_reason")
            extra_parts = []
            if "_reject_rr" in raw_signal:
                extra_parts.append(f"rr={raw_signal['_reject_rr']}")
            if "_reject_tp_abs" in raw_signal:
                extra_parts.append(f"tp_abs={raw_signal['_reject_tp_abs']}")
            if "_reject_score" in raw_signal:
                extra_parts.append(f"score={raw_signal['_reject_score']}")
            if "_reject_sl_atr" in raw_signal:
                extra_parts.append(f"sl_atr={raw_signal['_reject_sl_atr']}")
            extra_str = ", ".join(extra_parts)
            print(
                f"  -> SKIP: RISK_FLAG_REJECTED "
                f"(reason={reject_reason}{', ' if extra_str else ''}{extra_str})"
            )
            add_debug(i, snap_ts, price, "RISK_FLAG", "SKIP",
                      reject_reason or "RISK_REJECTED", extra_str)
            continue

        generated += 1
        extra_ok = (
            f"side={full_signal['side']}, "
            f"price={full_signal['entry_price']}, "
            f"rr={full_signal['rr']:.2f}, "
            f"score={full_signal['score']:.1f}, "
            f"flag_pos={full_signal.get('flag_position')}, "
            f"risk_mode={full_signal.get('risk_mode')}"
        )
        print("  -> OK SIGNAL (NEUTRAL_FLAG): " + extra_ok)
        add_debug(i, snap_ts, price, "FINAL_FLAG", "OK", "OK_SIGNAL", extra_ok)

        signals.append(full_signal)
        last_signal_index_by_instrument[instrument] = i

    print(f"Done. Processed snapshots={processed}, generated signals={generated}.")

    # --- sygnały do CSV ---
    if signals:
        fieldnames = sorted(signals[0].keys())
    else:
        fieldnames = [
            "instrument", "strategy_hash", "strategy_name",
            "side", "timestamp", "entry_price", "sl", "tp",
            "rr", "score", "bias", "timeframe_entry", "risk_mode",
        ]

    with open(args.output_signals_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for s in signals:
            writer.writerow(s)

    print(f"Signals saved to: {args.output_signals_csv}")

    # --- debug per snapshot ---
    debug_fieldnames = [
        "snap_index", "snap_ts", "price",
        "bias", "stage", "result", "reason", "extra"
    ]
    with open(args.debug_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fieldnames, delimiter=';')
        writer.writeheader()
        for r in debug_rows:
            writer.writerow(r)

    print(f"Debug log saved to: {args.debug_csv}")

    # --- stats z powodami SKIP ---
    reason_counts = {}
    for r in debug_rows:
        if r.get("result") != "SKIP":
            continue
        reason = r.get("reason") or "UNKNOWN"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    with open(args.stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            writer.writerow([reason, count])

    print(f"Stats saved to: {args.stats_csv}")


if __name__ == "__main__":
    main()
