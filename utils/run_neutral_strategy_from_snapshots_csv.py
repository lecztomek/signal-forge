import argparse
import csv
import json
import hashlib
import base64
from decimal import Decimal


# ============================================================
#                  Pomocnicze konwersje / hash
# ============================================================

COOLDOWN_BARS_PER_ZONE = 5  # minimalna liczba świec M5 przerwy dla tej samej strefy / strony


def _from_decimal_deep(obj):
    """W offline CSV nie używamy Decimal, ale zostawiam helper na wszelki wypadek."""
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


def _parse_number_or_none(v: str):
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() == "none" or v.lower() == "null":
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return float(int(v))
    except Exception:
        return None


# ============================================================
#         Parsowanie snapshotu z CSV → daily / h1 / m5 / SR
# ============================================================

def split_parts_from_row(row: dict):
    """
    Zakładamy, że generator snapshotów nazwał kolumny np.:
      daily_ts, daily_close, daily_ema20, ...
      h1_ts, h1_close, ...
      m5_ts, m5_close, ...
    Albo z kropką: daily.ts itd.
    Bierzemy oba warianty.
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
                daily[key] = _parse_number_or_none(v)
        elif k.startswith("h1_") or k.startswith("h1."):
            key = k.split("_", 1)[-1] if "_" in k else k.split(".", 1)[-1]
            if key in ("ts", "time", "timestamp"):
                h1["ts"] = v
            else:
                h1[key] = _parse_number_or_none(v)
        elif k.startswith("m5_") or k.startswith("m5."):
            key = k.split("_", 1)[-1] if "_" in k else k.split(".", 1)[-1]
            if key in ("ts", "time", "timestamp"):
                m5["ts"] = v
            else:
                m5[key] = _parse_number_or_none(v)

    # fallback – jeśli nie było prefiksów, spróbuj kilka nazw top-level
    if "ts" not in m5 and "ts" in row:
        m5["ts"] = row["ts"]
    if "close" not in m5 and "close" in row:
        m5["close"] = _parse_number_or_none(row["close"])

    return daily, h1, m5


def parse_sr_multi_from_row(row: dict):
    sr_multi = {"support_zones": [], "resistance_zones": []}

    # --- 1) nowy przypadek: jedna kolumna sr_multi_json ---
    sr_multi_raw = row.get("sr_multi_json")
    if sr_multi_raw:
        s = sr_multi_raw.strip()
        # jeżeli CSV owija całość w dodatkowe "
        if s and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        # podwójne "" -> pojedyncze "
        s = s.replace('""', '"')

        try:
            data = json.loads(s)
            if isinstance(data, dict):
                sr_multi["support_zones"] = data.get("support_zones") or []
                sr_multi["resistance_zones"] = data.get("resistance_zones") or []
                return sr_multi
        except Exception as e:
            print(f"[WARN] failed to parse sr_multi_json: {e}")

    # --- 2) fallback: stary format z dwoma kolumnami ---
    support_keys = [
        "sr_multi_support_zones",
        "sr_multi_support_zones_json",
        "support_zones",
    ]
    resistance_keys = [
        "sr_multi_resistance_zones",
        "sr_multi_resistance_zones_json",
        "resistance_zones",
    ]

    support_json = None
    resistance_json = None

    for k in support_keys:
        if k in row and row[k]:
            support_json = row[k]
            break
    for k in resistance_keys:
        if k in row and row[k]:
            resistance_json = row[k]
            break

    if support_json:
        try:
            sr_multi["support_zones"] = json.loads(support_json)
        except Exception:
            sr_multi["support_zones"] = []
    if resistance_json:
        try:
            sr_multi["resistance_zones"] = json.loads(resistance_json)
        except Exception:
            sr_multi["resistance_zones"] = []

    return sr_multi



# ============================================================
#                DAILY – wykrywanie NEUTRAL
# ============================================================

def is_neutral_daily(daily_part: dict, cfg_daily: dict):
    """
    Wykrywa reżim NEUTRAL na D1.
    Zwraca: (is_neutral: bool, reason: str)
    """
    if not cfg_daily or not cfg_daily.get("enabled", True):
        return False, "DAILY_DISABLED"

    if not isinstance(daily_part, dict):
        return False, "NO_DAILY_PART"

    ema20 = daily_part.get("ema20")
    ema50 = daily_part.get("ema50")
    atr_pct = daily_part.get("atr14_pct_rank")
    slope_pct = daily_part.get("ema50_slope_pct")
    rsi_pct = daily_part.get("rsi14_pct_rank")

    if any(v is None for v in [ema20, ema50, atr_pct, slope_pct, rsi_pct]):
        return False, "MISSING_DAILY_FIELDS"

    max_atr_pct = cfg_daily.get("max_atr_pct", 30.0)
    max_abs_slope_pct = cfg_daily.get("max_abs_slope_pct", 10.0)
    max_ema_distance_pct = cfg_daily.get("max_ema_distance_pct", 2.0)
    rsi_mid_min = cfg_daily.get("rsi_mid_min_pct", 40.0)
    rsi_mid_max = cfg_daily.get("rsi_mid_max_pct", 60.0)

    if atr_pct > max_atr_pct:
        return False, "ATR_TOO_HIGH"

    if abs(slope_pct) > max_abs_slope_pct:
        return False, "SLOPE_TOO_STRONG"

    if ema50 == 0:
        return False, "EMA50_ZERO"
    ema_dist_pct = abs(ema20 - ema50) / abs(ema50) * 100.0
    if ema_dist_pct > max_ema_distance_pct:
        return False, "EMA_DISTANCE_TOO_LARGE"

    if not (rsi_mid_min <= rsi_pct <= rsi_mid_max):
        return False, "RSI_NOT_MIDRANGE"

    return True, "OK_NEUTRAL"


# ============================================================
#                     H1 – filtr NEUTRAL
# ============================================================

def check_h1_neutral(h1_part: dict, cfg_h1: dict) -> (bool, str):
    """
    H1 filtr dla neutral-mode.
    Zwraca: (ok: bool, reason: str)
    """
    if not cfg_h1 or not cfg_h1.get("enabled", True):
        return True, "H1_DISABLED"

    if not isinstance(h1_part, dict):
        return False, "NO_H1_PART"

    close = h1_part.get("close")
    ema = h1_part.get("ema")
    slope_pct = h1_part.get("ema_slope_pct")
    atr_pct = h1_part.get("atr14_pct_rank")
    atr_val = h1_part.get("atr14")

    if close is None or ema is None or slope_pct is None:
        return False, "MISSING_H1_FIELDS"

    max_abs_slope_pct = cfg_h1.get("max_abs_slope_pct", 15.0)
    max_price_distance_ema_atr = cfg_h1.get("max_price_distance_ema_atr", 2.0)
    max_atr_pct = cfg_h1.get("max_atr_pct")

    if abs(slope_pct) > max_abs_slope_pct:
        return False, "H1_SLOPE_TOO_STRONG"

    if atr_val is not None and atr_val > 0 and max_price_distance_ema_atr is not None:
        dist_atr = abs(close - ema) / atr_val
        if dist_atr > max_price_distance_ema_atr:
            return False, "PRICE_TOO_FAR_FROM_EMA"

    if max_atr_pct is not None and atr_pct is not None:
        if atr_pct > max_atr_pct:
            return False, "H1_ATR_TOO_HIGH"

    return True, "OK_H1_NEUTRAL"


# ============================================================
#            5M – mean-reversion trigger w neutral
# ============================================================

def check_neutral_5m_trigger(m5_part: dict, cfg_m5: dict):
    """
    5M mean-reversion trigger dla NEUTRAL-mode.
    Zwraca: (raw_signal: dict | None, debug: dict)
    """
    debug = {
        "enabled": bool(cfg_m5 and cfg_m5.get("enabled", True)),
        "reason": None,
        "raw": {},
        "side": None,
        "score_long": 0.0,
        "score_short": 0.0,
    }

    if not cfg_m5 or not cfg_m5.get("enabled", True):
        debug["reason"] = "M5_DISABLED"
        return None, debug

    if not isinstance(m5_part, dict):
        debug["reason"] = "NO_M5_PART"
        return None, debug

    candles_count = m5_part.get("candles_count", 0)
    min_candles = cfg_m5.get("min_candles", 30)
    if candles_count < min_candles:
        debug["reason"] = "NOT_ENOUGH_CANDLES"
        return None, debug

    price = m5_part.get("close")
    ema20 = m5_part.get("ema20")
    last_rsi_pct = m5_part.get("rsi14_pct_rank")
    prev_rsi_pct = m5_part.get("prev_rsi14_pct_rank")
    atr_val = m5_part.get("atr14")
    atr_pct = m5_part.get("atr14_pct_rank")
    ts = m5_part.get("ts")

    debug["raw"] = {
        "price": price,
        "ema20": ema20,
        "rsi_pct": last_rsi_pct,
        "prev_rsi_pct": prev_rsi_pct,
        "atr": atr_val,
        "atr_pct": atr_pct,
        "candles_count": candles_count,
    }

    if any(v is None for v in [price, ema20, last_rsi_pct, prev_rsi_pct, atr_val]):
        debug["reason"] = "MISSING_M5_FIELDS"
        return None, debug

    atr_regime_cfg = cfg_m5.get("atr_regime", {})
    if atr_regime_cfg.get("enabled", False):
        min_pct = atr_regime_cfg.get("min_pct", 0.0)
        max_pct = atr_regime_cfg.get("max_pct", 100.0)
        if atr_pct is None or not (min_pct <= atr_pct <= max_pct):
            debug["reason"] = "M5_ATR_REGIME_FAILED"
            return None, debug

    score_cfg = cfg_m5.get("score", {})
    min_score = score_cfg.get("min_score", 40.0)
    rsi_points = score_cfg.get("rsi_points", 20.0)
    ema_points = score_cfg.get("ema_points", 10.0)
    sr_points = score_cfg.get("sr_points", 20.0)

    long_cfg = cfg_m5.get("long", {})
    short_cfg = cfg_m5.get("short", {})

    # ---------------- LONG candidate ----------------
    score_long = 0.0
    rsi_ok_long = False
    ema_ok_long = False
    sr_ok_long = False
    
    lp_prev_min = long_cfg.get("rsi_prev_min_pct", 30.0)
    lp_last_max = long_cfg.get("rsi_last_max_pct", 35.0)
    
    # Prostsza logika RSI: poprzednie "trochę wyżej", ostatnie "raczej nisko"
    if prev_rsi_pct is not None and last_rsi_pct is not None:
        if prev_rsi_pct >= lp_prev_min and last_rsi_pct <= lp_last_max:
            rsi_ok_long = True
            score_long += rsi_points

    max_above_ema_atr = long_cfg.get("max_price_above_ema_atr", 0.5)
    if atr_val > 0:
        dist_atr_long = (price - ema20) / atr_val
        # w NEUTRAL chcemy, żeby cena nie była za bardzo powyżej EMA (mean-reversion)
        if dist_atr_long <= max_above_ema_atr:
            ema_ok_long = True
            score_long += ema_points

    sr_min_strength_l = long_cfg.get("sr_min_strength", 1)
    sr_max_dist_atr_l = long_cfg.get("sr_max_dist_atr", 1.5)
    sr_min_fresh_l = long_cfg.get("sr_min_freshness", 0.0)
    sr_max_age_l = long_cfg.get("sr_max_age_hours")

    sr_strength = m5_part.get("nearest_support_strength")
    sr_dist_atr = m5_part.get("nearest_support_dist_atr")
    sr_fresh = m5_part.get("nearest_support_freshness")
    sr_age = m5_part.get("nearest_support_age_hours")
    sr_level = m5_part.get("nearest_support_level")

    if sr_strength is not None and sr_dist_atr is not None:
        if sr_strength >= sr_min_strength_l and sr_dist_atr <= sr_max_dist_atr_l:
            fresh_ok = (sr_fresh is None or sr_fresh >= sr_min_fresh_l)
            age_ok = (sr_max_age_l is None or (sr_age is not None and sr_age <= sr_max_age_l))
            if fresh_ok and age_ok:
                sr_ok_long = True
                score_long += sr_points

    # ---------------- SHORT candidate ----------------
    score_short = 0.0
    rsi_ok_short = False
    ema_ok_short = False
    sr_ok_short = False
    
    sp_prev_max = short_cfg.get("rsi_prev_max_pct", 65.0)
    sp_last_min = short_cfg.get("rsi_last_min_pct", 70.0)
    
    # Prostsza logika RSI: poprzednie "trochę niżej", ostatnie "raczej wysoko"
    if prev_rsi_pct is not None and last_rsi_pct is not None:
        if prev_rsi_pct <= sp_prev_max and last_rsi_pct >= sp_last_min:
            rsi_ok_short = True
            score_short += rsi_points

    max_below_ema_atr = short_cfg.get("max_price_below_ema_atr", 0.5)
    if atr_val > 0:
        dist_atr_short = (ema20 - price) / atr_val
        if dist_atr_short <= max_below_ema_atr:
            ema_ok_short = True
            score_short += ema_points

    sr_min_strength_s = short_cfg.get("sr_min_strength", 1)
    sr_max_dist_atr_s = short_cfg.get("sr_max_dist_atr", 1.5)
    sr_min_fresh_s = short_cfg.get("sr_min_freshness", 0.0)
    sr_max_age_s = short_cfg.get("sr_max_age_hours")

    sr_strength_r = m5_part.get("nearest_resistance_strength")
    sr_dist_atr_r = m5_part.get("nearest_resistance_dist_atr")
    sr_fresh_r = m5_part.get("nearest_resistance_freshness")
    sr_age_r = m5_part.get("nearest_resistance_age_hours")
    sr_level_r = m5_part.get("nearest_resistance_level")

    if sr_strength_r is not None and sr_dist_atr_r is not None:
        if sr_strength_r >= sr_min_strength_s and sr_dist_atr_r <= sr_max_dist_atr_s:
            fresh_ok = (sr_fresh_r is None or sr_fresh_r >= sr_min_fresh_s)
            age_ok = (sr_max_age_s is None or (sr_age_r is not None and sr_age_r <= sr_max_age_s))
            if fresh_ok and age_ok:
                sr_ok_short = True
                score_short += sr_points

    debug["score_long"] = float(score_long)
    debug["score_short"] = float(score_short)
    debug["rsi_ok_long"] = rsi_ok_long
    debug["rsi_ok_short"] = rsi_ok_short
    debug["ema_ok_long"] = ema_ok_long
    debug["ema_ok_short"] = ema_ok_short
    debug["sr_ok_long"] = sr_ok_long
    debug["sr_ok_short"] = sr_ok_short

    best_side = None
    best_score = 0.0
    best_sr_level = None
    best_sr_strength = None
    best_sr_dist = None

    # WYMAGAMY, żeby RSI BYŁO OK dla wybranego kierunku
    if score_long >= score_short and score_long >= min_score and sr_ok_long and rsi_ok_long:
        best_side = "BUY"
        best_score = score_long
        best_sr_level = sr_level
        best_sr_strength = sr_strength
        best_sr_dist = sr_dist_atr
    elif score_short > score_long and score_short >= min_score and sr_ok_short and rsi_ok_short:
        best_side = "SELL"
        best_score = score_short
        best_sr_level = sr_level_r
        best_sr_strength = sr_strength_r
        best_sr_dist = sr_dist_atr_r

    if not best_side:
        debug["reason"] = "NO_SIDE_ABOVE_MIN_SCORE"
        return None, debug

    debug["side"] = best_side
    debug["reason"] = "OK_TRIGGER"

    raw_signal = {
        "side": best_side,
        "timestamp": ts,
        "price": price,
        "atr": atr_val,
        "sr_level": best_sr_level,
        "sr_strength": best_sr_strength,
        "sr_distance_atr_m5": best_sr_dist,
        "m5_score": best_score,
        "m5_score_debug": debug,
    }

    return raw_signal, debug


# ============================================================
#      Pomocnicze – ATR / SR_MULTI (jak w lambdzie)
# ============================================================

def _compute_atr_for_risk(cfg_risk, raw_signal, h1_part):
    atr_source = (cfg_risk.get("atr_source") or "h1").lower()
    atr_m5 = raw_signal.get("atr")
    atr_h1 = None
    if isinstance(h1_part, dict):
        atr_h1 = h1_part.get("atr14")

    if atr_source == "h1":
        return atr_h1 or atr_m5
    elif atr_source == "m5":
        return atr_m5 or atr_h1
    else:
        return atr_h1 or atr_m5


def _pick_zones_for_side(sr_multi, side, kind):
    if not sr_multi:
        return []

    supports = sr_multi.get("support_zones") or []
    resistances = sr_multi.get("resistance_zones") or []

    if side == "BUY":
        return supports if kind == "sl" else resistances
    else:
        return resistances if kind == "sl" else supports


def _select_sr_zone(sr_zones, price, side, is_sl, atr_val, cfg):
    if not sr_zones or atr_val is None or atr_val <= 0:
        return None, None

    min_strength = cfg.get("min_strength", 1)
    min_freshness = cfg.get("min_freshness", 0.0)
    tf_weights = cfg.get("tf_weights", {})
    max_atr = cfg.get("max_sl_atr") if is_sl else cfg.get("max_tp_atr")

    candidates = []

    for z in sr_zones:
        level = z.get("level")
        strength = z.get("strength")
        freshness = z.get("freshness", 1.0)
        tf = z.get("timeframe", "H1")

        if level is None or strength is None:
            continue

        if side == "BUY":
            if is_sl and level > price:
                continue
            if not is_sl and level < price:
                continue
        else:
            if is_sl and level < price:
                continue
            if not is_sl and level > price:
                continue

        if strength < min_strength:
            continue
        if freshness < min_freshness:
            continue

        dist_abs = abs(price - level)
        dist_atr = dist_abs / atr_val

        if max_atr is not None and dist_atr > max_atr:
            continue

        w_tf = tf_weights.get(tf, 1.0)
        score = w_tf * strength * freshness / (1.0 + dist_atr)

        candidates.append((score, dist_atr, z))

    if not candidates:
        return None, None

    candidates.sort(key=lambda t: (-t[0], t[1]))
    best_score, best_dist_atr, best_zone = candidates[0]
    return best_zone, best_dist_atr


# ============================================================
#               RISK / SL+TP – SR_MULTI neutral
# ============================================================

def build_neutral_signal_with_risk(
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
    if not cfg_risk or not cfg_risk.get("enabled", True):
        raw_signal["_reject_reason"] = "RISK_DISABLED"
        return None

    side = raw_signal["side"]
    price = raw_signal["price"]
    m5_score = raw_signal.get("m5_score")
    atr_m5 = raw_signal.get("atr")

    atr_val = _compute_atr_for_risk(cfg_risk, raw_signal, h1_part)
    if atr_val is None or atr_val <= 0:
        raw_signal["_reject_reason"] = "NO_ATR"
        return None

    sr_sl_cfg = cfg_risk.get("sr_sl", {})
    sr_tp_cfg = cfg_risk.get("sr_tp", {})

    if not sr_sl_cfg.get("enabled", True) or not sr_tp_cfg.get("enabled", True):
        raw_signal["_reject_reason"] = "SR_SL_OR_TP_DISABLED"
        return None

    sl_zones = _pick_zones_for_side(sr_multi, side, kind="sl")
    sl_zone, sl_dist_atr = _select_sr_zone(sl_zones, price, side, is_sl=True, atr_val=atr_val, cfg=sr_sl_cfg)
    if sl_zone is None:
        raw_signal["_reject_reason"] = "SL_ZONE_NOT_FOUND"
        return None

    tp_zones = _pick_zones_for_side(sr_multi, side, kind="tp")
    tp_zone, tp_dist_atr = _select_sr_zone(tp_zones, price, side, is_sl=False, atr_val=atr_val, cfg=sr_tp_cfg)
    if tp_zone is None:
        raw_signal["_reject_reason"] = "TP_ZONE_NOT_FOUND"
        return None

    sl_buffer = sr_sl_cfg.get("sl_buffer_atr", 0.2)
    tp_buffer = sr_tp_cfg.get("tp_buffer_atr", 0.3)

    if side == "BUY":
        sl = sl_zone["level"] - sl_buffer * atr_val
        tp = tp_zone["level"] - tp_buffer * atr_val
        if sl >= price:
            raw_signal["_reject_reason"] = "SL_ABOVE_ENTRY"
            return None
        if tp <= price:
            raw_signal["_reject_reason"] = "TP_BELOW_ENTRY"
            return None
    else:
        sl = sl_zone["level"] + sl_buffer * atr_val
        tp = tp_zone["level"] + tp_buffer * atr_val
        if sl <= price:
            raw_signal["_reject_reason"] = "SL_BELOW_ENTRY"
            return None
        if tp >= price:
            raw_signal["_reject_reason"] = "TP_ABOVE_ENTRY"
            return None

    tp_abs = abs(tp - price)
    sl_abs = abs(price - sl)
    if sl_abs <= 0:
        raw_signal["_reject_reason"] = "SL_DISTANCE_ZERO"
        return None

    # minimalna odległość SL w ATR-ach (żeby nie było mikro-SL tuż pod strefą)
    sl_distance_atr = sl_abs / atr_val
    min_sl_dist_atr = sr_sl_cfg.get("min_sl_distance_atr")
    if min_sl_dist_atr is not None and sl_distance_atr < min_sl_dist_atr:
        raw_signal["_reject_reason"] = "SL_TOO_CLOSE_ATR"
        raw_signal["_reject_sl_dist_atr"] = sl_distance_atr
        return None

    rr = tp_abs / sl_abs

    min_tp_abs = cfg_risk.get("min_tp_abs")
    if min_tp_abs is not None and tp_abs < min_tp_abs:
        raw_signal["_reject_reason"] = "TP_TOO_CLOSE_ABS"
        raw_signal["_reject_tp_abs"] = tp_abs
        return None

    min_rr = cfg_risk.get("min_rr")
    if min_rr is not None and rr < min_rr:
        raw_signal["_reject_reason"] = "RR_TOO_LOW"
        raw_signal["_reject_rr"] = rr
        return None

    base_score = cfg_risk.get("base_score", 60.0)
    score = base_score

    score_components = cfg_risk.get("score_components", {})
    rr_bonus_if_ge = score_components.get("rr_bonus_if_ge", 2.0)
    rr_bonus_points = score_components.get("rr_bonus_points", 10.0)
    sr_strength_weight = score_components.get("sr_strength_weight", 2.0)
    sr_strength_cap = score_components.get("sr_strength_cap", 10.0)
    sr_close_bonus_points = score_components.get("sr_close_bonus_points", 5.0)
    sr_close_threshold_atr = score_components.get("sr_close_threshold_atr", 1.0)
    rr_penalty_if_gt = score_components.get("rr_penalty_if_gt")
    rr_penalty_points = score_components.get("rr_penalty_points", 0.0)

    if rr >= rr_bonus_if_ge:
        score += rr_bonus_points

    # opcjonalna kara za zbyt kosmiczny RR
    if rr_penalty_if_gt is not None and rr > rr_penalty_if_gt and rr_penalty_points > 0:
        score -= rr_penalty_points

    sl_strength = (sl_zone or {}).get("strength")
    tp_strength = (tp_zone or {}).get("strength")
    sr_strength_for_score = max(sl_strength or 0, tp_strength or 0)

    if sr_strength_for_score:
        score += min(sr_strength_for_score * sr_strength_weight, sr_strength_cap)

    dist_candidates = [d for d in [sl_dist_atr, tp_dist_atr] if d is not None]
    if dist_candidates:
        sr_dist_for_score = min(dist_candidates)
        if sr_dist_for_score <= sr_close_threshold_atr:
            score += sr_close_bonus_points

    min_score = cfg_risk.get("min_score", 70.0)
    if score < min_score:
        raw_signal["_reject_reason"] = "GLOBAL_SCORE_TOO_LOW"
        raw_signal["_reject_score"] = score
        return None

    atr_h1 = None
    if isinstance(h1_part, dict):
        atr_h1 = h1_part.get("atr14")

    sr_level = (sl_zone or {}).get("level")
    sr_strength = sl_strength
    sr_distance_atr = sl_dist_atr

    full_signal = {
        "instrument": instrument,
        "strategy_hash": strategy_hash,
        "strategy_name": strategy_name,
        "strategy_config_json": config_json,
        "side": side,
        "timestamp": raw_signal["timestamp"],
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "score": score,
        "bias": "NEUTRAL",
        "timeframe_entry": "5m",
        "risk_mode": "SR_NEUTRAL",
        "sr_level": sr_level,
        "sr_strength": sr_strength,
        "sr_distance_atr": sr_distance_atr,
        "sl_zone_level": (sl_zone or {}).get("level"),
        "sl_zone_strength": sl_strength,
        "sl_zone_freshness": (sl_zone or {}).get("freshness"),
        "sl_zone_timeframe": (sl_zone or {}).get("timeframe"),
        "sl_zone_dist_atr": sl_dist_atr,
        "tp_zone_level": (tp_zone or {}).get("level"),
        "tp_zone_strength": tp_strength,
        "tp_zone_freshness": (tp_zone or {}).get("freshness"),
        "tp_zone_timeframe": (tp_zone or {}).get("timeframe"),
        "tp_zone_dist_atr": tp_dist_atr,
        "m5_score": m5_score,
        "atr_m5": atr_m5,
        "atr_higher_tf": atr_h1,
        "atr_used": atr_val,
        "sl_distance_atr_abs": sl_distance_atr,
    }

    return full_signal


# ============================================================
#                       main() – offline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", required=True,
                        help="CSV ze snapshotami M5/D1/H1")
    parser.add_argument("--config_json", required=True,
                        help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="neutral_signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None,
                        help="opcjonalny override instrumentu (np. BZ=F)")
    parser.add_argument("--debug_csv", default="neutral_signals_debug_log.csv",
                        help="plik CSV z powodami braku sygnału / debugiem")
    parser.add_argument("--stats_csv", default="neutral_signals_debug_stats.csv",
                        help="plik CSV z agregacją powodów SKIP")

    args = parser.parse_args()

    # config strategii
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "neutral_strategy")
    strategy_hash = compute_strategy_hash(config)
    cfg_daily = config.get("daily", {})
    cfg_h1 = config.get("h1", {})
    cfg_m5 = config.get("m5", {})
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

    # cooldown per strefa / strona / instrument
    zone_last_index = {}

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
            or daily_part.get("ts")
            or run_at
            or row.get("ts")
        )
        price = m5_part.get("close")

        print(
            f"[{progress:6.2f}%] snap_ts={snap_ts}, price={price}",
            flush=True,
        )

        def add_debug(stage, result, reason, extra=""):
            debug_rows.append({
                "snap_index": i,
                "snap_ts": snap_ts,
                "price": price,
                "bias": "NEUTRAL",   # tu zawsze neutralny tryb
                "stage": stage,
                "result": result,
                "reason": reason,
                "extra": extra,
            })

        # 1) DAILY – neutralny reżim?
        is_neutral, daily_reason = is_neutral_daily(daily_part, cfg_daily)
        if not is_neutral:
            extra = (
                f"ema20={daily_part.get('ema20')}, "
                f"ema50={daily_part.get('ema50')}, "
                f"rsi_pct={daily_part.get('rsi14_pct_rank')}, "
                f"atr_pct={daily_part.get('atr14_pct_rank')}, "
                f"slope_pct={daily_part.get('ema50_slope_pct')}"
            )
            print(f"  -> SKIP: DAILY_NOT_NEUTRAL ({daily_reason}; {extra})")
            add_debug(stage="DAILY_NEUTRAL", result="SKIP",
                      reason=daily_reason, extra=extra)
            continue

        # 2) H1 – filtr neutralności
        h1_ok, h1_reason = check_h1_neutral(h1_part, cfg_h1)
        if not h1_ok:
            extra = (
                f"h1_close={h1_part.get('close')}, "
                f"h1_ema={h1_part.get('ema')}, "
                f"h1_rsi_pct={h1_part.get('rsi14_pct_rank')}, "
                f"h1_atr_pct={h1_part.get('atr14_pct_rank')}, "
                f"h1_slope_pct={h1_part.get('ema_slope_pct')}"
            )
            print(f"  -> SKIP: H1_NEUTRAL_FAILED ({h1_reason}; {extra})")
            add_debug(stage="H1_NEUTRAL", result="SKIP",
                      reason=h1_reason, extra=extra)
            continue

        # 3) 5M – neutral mean-reversion trigger
        raw_signal, m5_debug = check_neutral_5m_trigger(m5_part, cfg_m5)
        if not raw_signal:
            reason = (m5_debug or {}).get("reason")
            extra = (
                f"m5_reason={reason}, "
                f"score_long={m5_debug.get('score_long') if m5_debug else None}, "
                f"score_short={m5_debug.get('score_short') if m5_debug else None}, "
                f"m5_rsi_pct={m5_part.get('rsi14_pct_rank')}, "
                f"m5_prev_rsi_pct={m5_part.get('prev_rsi14_pct_rank')}, "
                f"m5_atr={m5_part.get('atr14')}, "
                f"m5_atr_pct={m5_part.get('atr14_pct_rank')}"
            )
            print(f"  -> SKIP: M5_NEUTRAL_TRIGGER_NOT_GENERATED ({extra})")
            add_debug(stage="M5_NEUTRAL", result="SKIP",
                      reason=reason or "M5_TRIGGER_FAILED", extra=extra)
            continue

        # 4) RISK / SR_MULTI
        full_signal = build_neutral_signal_with_risk(
            raw_signal=raw_signal,
            cfg_risk=cfg_risk,
            instrument=instrument,
            strategy_hash=strategy_hash,
            strategy_name=strategy_name,
            config_json=json.dumps(config, sort_keys=True),
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
            if "_reject_sl_dist_atr" in raw_signal:
                extra_parts.append(f"sl_dist_atr={raw_signal['_reject_sl_dist_atr']}")
            extra_str = ", ".join(extra_parts)
            print(
                f"  -> SKIP: RISK_NEUTRAL_REJECTED "
                f"(reason={reject_reason}{', ' if extra_str else ''}{extra_str})"
            )
            add_debug(stage="RISK_NEUTRAL", result="SKIP",
                      reason=reject_reason or "RISK_REJECTED",
                      extra=extra_str)
            continue

        # 5) COOLDOWN po strefie – nie spamuj sygnałów co świecę na tym samym poziomie
        zone_key = (
            full_signal.get("instrument"),
            full_signal.get("side"),
            full_signal.get("sr_level"),
            full_signal.get("sl_zone_timeframe"),
        )
        last_idx = zone_last_index.get(zone_key)
        if last_idx is not None and (i - last_idx) < COOLDOWN_BARS_PER_ZONE:
            reason = f"COOLDOWN_{COOLDOWN_BARS_PER_ZONE}_BARS_ACTIVE"
            extra = f"zone_key={zone_key}, last_index={last_idx}"
            print(f"  -> SKIP: {reason} ({extra})")
            add_debug(stage="COOLDOWN", result="SKIP", reason=reason, extra=extra)
            continue

        zone_last_index[zone_key] = i

        generated += 1
        extra_ok = (
            f"side={full_signal['side']}, "
            f"price={full_signal['entry_price']}, "
            f"rr={full_signal['rr']:.2f}, "
            f"score={full_signal['score']:.1f}, "
            f"m5_score={full_signal.get('m5_score')}"
        )
        print("  -> OK SIGNAL (NEUTRAL): " + extra_ok)
        add_debug(stage="FINAL_NEUTRAL", result="OK",
                  reason="OK_SIGNAL", extra=extra_ok)

        signals.append(full_signal)

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
