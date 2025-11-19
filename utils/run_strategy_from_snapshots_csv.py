import csv
import json
import hashlib
import base64
import argparse
from datetime import datetime, timezone
from typing import Optional


# ===========================
# Pomocnicze do CSV
# ===========================

def parse_value(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def split_parts_from_row(row):
    """
    Z wiersza CSV (DictReader) wyciąga:
    - daily_part (klucze bez prefiksu 'daily_')
    - h1_part   (klucze bez 'h1_')
    - m5_part   (klucze bez 'm5_')
    """
    daily = {}
    h1 = {}
    m5 = {}
    for k, v in row.items():
        if k.startswith("daily_"):
            daily[k[len("daily_"):]] = parse_value(v)
        elif k.startswith("h1_"):
            h1[k[len("h1_"):]] = parse_value(v)
        elif k.startswith("m5_"):
            m5[k[len("m5_"):]] = parse_value(v)
    return daily, h1, m5


def parse_sr_multi_from_row(row):
    raw = row.get("sr_multi_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ===========================
# Hash strategii
# ===========================

def compute_strategy_hash(config: dict) -> str:
    config_for_hash = dict(config)
    canon = json.dumps(config_for_hash, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).digest()
    short = base64.b32encode(h[:5]).decode("utf-8").rstrip("=")
    return short


# ===========================
# DAILY BIAS – jak w Lambdzie
# ===========================

def determine_daily_bias(daily_part: dict, cfg_daily: dict) -> str:
    if not cfg_daily or not cfg_daily.get("enabled", True):
        return "NONE"

    ema20 = daily_part.get("ema20")
    ema50 = daily_part.get("ema50")
    rsi_pct = daily_part.get("rsi14_pct_rank")
    atr_pct = daily_part.get("atr14_pct_rank")
    slope_pct = daily_part.get("ema50_slope_pct")

    if any(v is None for v in [ema20, ema50, rsi_pct, atr_pct, slope_pct]):
        return "NONE"

    adaptive_cfg = cfg_daily.get("adaptive", {}) or {}

    long_rsi_min_pct = adaptive_cfg.get(
        "long_rsi_min_pct",
        cfg_daily.get("long_rsi_min_pct", 55.0)
    )
    short_rsi_max_pct = adaptive_cfg.get(
        "short_rsi_max_pct",
        cfg_daily.get("short_rsi_max_pct", 45.0)
    )
    long_min_slope_pct = adaptive_cfg.get(
        "long_min_slope_pct",
        cfg_daily.get("long_min_slope_pct", 0.0)
    )
    short_max_slope_pct = adaptive_cfg.get(
        "short_max_slope_pct",
        cfg_daily.get("short_max_slope_pct", 0.0)
    )
    min_atr_pct = adaptive_cfg.get(
        "min_atr_pct",
        cfg_daily.get("min_atr_pct", 0.0)
    )
    max_atr_pct = adaptive_cfg.get(
        "max_atr_pct",
        cfg_daily.get("max_atr_pct", 100.0)
    )

    if not (min_atr_pct <= atr_pct <= max_atr_pct):
        return "NONE"

    trend_up = (ema20 > ema50) and (slope_pct >= long_min_slope_pct)
    trend_down = (ema20 < ema50) and (slope_pct <= short_max_slope_pct)

    momentum_long = rsi_pct >= long_rsi_min_pct
    momentum_short = rsi_pct <= short_rsi_max_pct

    if trend_up and momentum_long:
        return "LONG"
    if trend_down and momentum_short:
        return "SHORT"

    return "NONE"


# ===========================
# H1 FILTER – jak w Lambdzie
# ===========================

def check_h1_setup(h1_part: dict, cfg_h1: dict, bias: str) -> bool:
    if bias not in ("LONG", "SHORT"):
        return False

    if not cfg_h1 or not cfg_h1.get("enabled", True):
        return False

    if not h1_part:
        return False

    close = h1_part.get("close")
    ema = h1_part.get("ema")
    rsi_pct = h1_part.get("rsi14_pct_rank")
    last_low = h1_part.get("last_low")
    prev_low = h1_part.get("prev_low")
    last_high = h1_part.get("last_high")
    prev_high = h1_part.get("prev_high")
    atr_pct = h1_part.get("atr14_pct_rank")
    slope_pct = h1_part.get("ema_slope_pct")

    if close is None or ema is None or rsi_pct is None:
        return False

    rsi_long_min_pct = cfg_h1.get("rsi_long_min_pct", 55.0)
    rsi_short_max_pct = cfg_h1.get("rsi_short_max_pct", 45.0)

    struct_cfg = cfg_h1.get("structure", {})
    long_requires_HL = struct_cfg.get("long_requires_HL", True)
    short_requires_LH = struct_cfg.get("short_requires_LH", True)

    conds = []

    # EMA
    if bias == "LONG":
        conds.append(close > ema)
    else:
        conds.append(close < ema)

    # RSI pct
    if bias == "LONG":
        conds.append(rsi_pct >= rsi_long_min_pct)
    else:
        conds.append(rsi_pct <= rsi_short_max_pct)

    # Struktura HL/LH
    if bias == "LONG" and long_requires_HL:
        if last_low is not None and prev_low is not None:
            conds.append(last_low >= prev_low)
        else:
            conds.append(False)
    elif bias == "SHORT" and short_requires_LH:
        if last_high is not None and prev_high is not None:
            conds.append(last_high <= prev_high)
        else:
            conds.append(False)
    else:
        conds.append(True)

    min_true = cfg_h1.get("min_true", 2)
    min_true = min(min_true, len(conds))

    satisfied = sum(1 for c in conds if c)
    if satisfied < min_true:
        return False

    # regime_filter
    regime_cfg = cfg_h1.get("regime_filter", {})
    if not regime_cfg.get("enabled", False):
        return True

    if atr_pct is None or slope_pct is None:
        return False

    min_atr_pct = regime_cfg.get("min_atr_pct", 0.0)
    max_atr_pct = regime_cfg.get("max_atr_pct", 100.0)
    long_min_slope_pct = regime_cfg.get("long_min_slope_pct", 0.0)
    short_max_slope_pct = regime_cfg.get("short_max_slope_pct", 0.0)

    if not (min_atr_pct <= atr_pct <= max_atr_pct):
        return False

    if bias == "LONG":
        if slope_pct < long_min_slope_pct:
            return False
    else:
        if slope_pct > short_max_slope_pct:
            return False

    return True


# ===========================
# 5M TRIGGER – jak w Twojej strategii
# ===========================

def check_5m_trigger(m5_part: dict, cfg_m5: dict, bias: str):
    score_debug = {
        "enabled": bool(cfg_m5 and cfg_m5.get("enabled", True)),
        "bias": bias,
        "reason": None,
        "rsi_contrib": 0.0,
        "ema_price_contrib": 0.0,
        "ema_trend_contrib": 0.0,
        "breakout_contrib": 0.0,
        "sr_contrib": 0.0,
        "total_score": 0.0,
        "min_score": None,
        "atr_filter_passed": True,
        "atr_regime_passed": True,
        "raw": {},
    }

    if bias not in ("LONG", "SHORT"):
        score_debug["reason"] = "NO_BIAS"
        return None, score_debug

    if not cfg_m5 or not cfg_m5.get("enabled", True):
        score_debug["reason"] = "CFG_M5_DISABLED"
        return None, score_debug

    if not m5_part or m5_part.get("candles_count", 0) < 30:
        score_debug["reason"] = "NOT_ENOUGH_CANDLES"
        return None, score_debug

    price = m5_part.get("close")
    last_rsi = m5_part.get("rsi14")
    prev_rsi = m5_part.get("prev_rsi14")
    last_rsi_pct = m5_part.get("rsi14_pct_rank")
    prev_rsi_pct = m5_part.get("prev_rsi14_pct_rank")
    atr_val = m5_part.get("atr14")
    atr_pct = m5_part.get("atr14_pct_rank")
    e20 = m5_part.get("ema20")
    e50 = m5_part.get("ema50")
    recent_high = m5_part.get("recent_high")
    recent_low = m5_part.get("recent_low")
    recent_high_age_bars = m5_part.get("recent_high_age_bars")
    recent_low_age_bars = m5_part.get("recent_low_age_bars")
    ts = m5_part.get("ts")

    for key, val in [
        ("price", price),
        ("rsi", last_rsi),
        ("prev_rsi", prev_rsi),
        ("rsi_pct_rank", last_rsi_pct),
        ("prev_rsi_pct_rank", prev_rsi_pct),
        ("atr", atr_val),
        ("atr_pct_rank", atr_pct),
        ("recent_high", recent_high),
        ("recent_low", recent_low),
        ("recent_high_age_bars", recent_high_age_bars),
        ("recent_low_age_bars", recent_low_age_bars),
    ]:
        score_debug["raw"][key] = val

    if any(v is None for v in [price, last_rsi, prev_rsi, last_rsi_pct, prev_rsi_pct, atr_val, e20, e50]):
        score_debug["reason"] = "MISSING_M5_FIELDS"
        return None, score_debug

    # ATR filter
    atr_cfg = cfg_m5.get("atr_filter", {})
    if atr_cfg.get("enabled", False):
        atr_min = atr_cfg.get("min")
        atr_max = atr_cfg.get("max")
        if atr_min is not None and atr_val < atr_min:
            score_debug["atr_filter_passed"] = False
            score_debug["reason"] = "ATR_BELOW_MIN"
            return None, score_debug
        if atr_max is not None and atr_val > atr_max:
            score_debug["atr_filter_passed"] = False
            score_debug["reason"] = "ATR_ABOVE_MAX"
            return None, score_debug

    # ATR regime
    atr_regime_cfg = cfg_m5.get("atr_regime", {})
    if atr_regime_cfg.get("enabled", False):
        if atr_pct is None:
            score_debug["reason"] = "NO_ATR_PCT_FOR_REGIME"
            score_debug["atr_regime_passed"] = False
            return None, score_debug
        min_pct = atr_regime_cfg.get("min_pct", 0.0)
        max_pct = atr_regime_cfg.get("max_pct", 100.0)
        if not (min_pct <= atr_pct <= max_pct):
            score_debug["reason"] = "ATR_PCT_OUT_OF_REGIME"
            score_debug["atr_regime_passed"] = False
            return None, score_debug
        score_debug["atr_regime_passed"] = True

    score_cfg = cfg_m5.get("score", {})
    min_score = score_cfg.get("min_score", 40)
    score_debug["min_score"] = min_score

    rsi_points = score_cfg.get("rsi_points", 15)
    ema_price_points = score_cfg.get("ema_price_points", 10)
    ema_trend_points = score_cfg.get("ema_trend_points", 10)
    breakout_points = score_cfg.get("breakout_points", 15)
    sr_points = score_cfg.get("sr_points", 15)
    sr_close_threshold_atr = score_cfg.get("sr_close_threshold_atr", 1.5)

    score_5m = 0.0
    rsi_contrib = 0.0
    ema_price_contrib = 0.0
    ema_trend_contrib = 0.0
    breakout_contrib = 0.0
    sr_contrib = 0.0

    # RSI pattern
    if bias == "LONG":
        rsi_long_cfg = cfg_m5.get("rsi_pattern_long", {})
        if rsi_long_cfg.get("enabled", True):
            prev_max_pct = rsi_long_cfg.get("prev_max_pct", 40.0)
            last_min_pct = rsi_long_cfg.get("last_min_pct", 50.0)
            if (
                prev_rsi_pct is not None
                and last_rsi_pct is not None
                and prev_rsi_pct <= prev_max_pct
                and last_rsi_pct >= last_min_pct
            ):
                score_5m += rsi_points
                rsi_contrib = rsi_points
    else:
        rsi_short_cfg = cfg_m5.get("rsi_pattern_short", {})
        if rsi_short_cfg.get("enabled", True):
            prev_min_pct = rsi_short_cfg.get("prev_min_pct", 70.0)
            last_max_pct = rsi_short_cfg.get("last_max_pct", 50.0)
            if (
                prev_rsi_pct is not None
                and last_rsi_pct is not None
                and prev_rsi_pct >= prev_min_pct
                and last_rsi_pct <= last_max_pct
            ):
                score_5m += rsi_points
                rsi_contrib = rsi_points

    # EMA
    ema_long_cfg = cfg_m5.get("ema_filter_long", {})
    ema_short_cfg = cfg_m5.get("ema_filter_short", {})

    if bias == "LONG" and ema_long_cfg.get("enabled", True):
        if price > e20:
            score_5m += ema_price_points
            ema_price_contrib = ema_price_points
        if e20 > e50:
            score_5m += ema_trend_points
            ema_trend_contrib = ema_trend_points

    if bias == "SHORT" and ema_short_cfg.get("enabled", True):
        if price < e20:
            score_5m += ema_price_points
            ema_price_contrib = ema_price_points
        if e20 < e50:
            score_5m += ema_trend_points
            ema_trend_contrib = ema_trend_points

    # Breakout
    breakout_high_cfg = cfg_m5.get("breakout_high", {})
    breakout_low_cfg = cfg_m5.get("breakout_low", {})

    if bias == "LONG" and breakout_high_cfg.get("enabled", True) and recent_high is not None:
        breakout_ok = price > recent_high
        if breakout_ok:
            min_age_bars = breakout_high_cfg.get("min_age_bars")
            max_age_bars = breakout_high_cfg.get("max_age_bars")
            if min_age_bars is not None and recent_high_age_bars is not None and recent_high_age_bars < min_age_bars:
                breakout_ok = False
            if max_age_bars is not None and recent_high_age_bars is not None and recent_high_age_bars > max_age_bars:
                breakout_ok = False
        if breakout_ok:
            score_5m += breakout_points
            breakout_contrib = breakout_points

    if bias == "SHORT" and breakout_low_cfg.get("enabled", True) and recent_low is not None:
        breakout_ok = price < recent_low
        if breakout_ok:
            min_age_bars = breakout_low_cfg.get("min_age_bars")
            max_age_bars = breakout_low_cfg.get("max_age_bars")
            if min_age_bars is not None and recent_low_age_bars is not None and recent_low_age_bars < min_age_bars:
                breakout_ok = False
            if max_age_bars is not None and recent_low_age_bars is not None and recent_low_age_bars > max_age_bars:
                breakout_ok = False
        if breakout_ok:
            score_5m += breakout_points
            breakout_contrib = breakout_points

    # SR – tu tylko zapisujemy raw, risk użyje sr_multi
    score_debug["rsi_contrib"] = float(rsi_contrib)
    score_debug["ema_price_contrib"] = float(ema_price_contrib)
    score_debug["ema_trend_contrib"] = float(ema_trend_contrib)
    score_debug["breakout_contrib"] = float(breakout_contrib)
    score_debug["sr_contrib"] = float(sr_contrib)
    score_debug["total_score"] = float(score_5m)

    if score_5m < min_score:
        score_debug["reason"] = "SCORE_BELOW_MIN"
        return None, score_debug

    side = "BUY" if bias == "LONG" else "SELL"

    raw_signal = {
        "side": side,
        "timestamp": ts,
        "price": price,
        "atr": atr_val,
        "sr_level": None,
        "sr_strength": None,
        "sr_distance_atr_m5": None,
        "m5_score": score_5m,
        "m5_score_debug": score_debug,
    }

    score_debug["reason"] = "OK_TRIGGER"
    return raw_signal, score_debug


# ===========================
# RISK / SR_MULTI – jak w Lambdzie
# ===========================

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
    """
    kind: "sl" / "tp"
    dla LONG:
      sl -> support_zones (poniżej ceny)
      tp -> resistance_zones (powyżej ceny)
    dla SHORT:
      sl -> resistance_zones (powyżej ceny)
      tp -> support_zones (poniżej ceny)
    """
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

        # kierunek
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


def build_signal_with_risk(
    raw_signal: dict,
    bias: str,
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
        return None

    price = raw_signal["price"]
    side = raw_signal["side"]
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

    # SL
    sl_zones = _pick_zones_for_side(sr_multi, side, kind="sl")
    sl_zone, sl_dist_atr = _select_sr_zone(sl_zones, price, side, is_sl=True, atr_val=atr_val, cfg=sr_sl_cfg)
    if sl_zone is None:
        raw_signal["_reject_reason"] = "SL_ZONE_NOT_FOUND"
        return None

    # TP
    tp_zones = _pick_zones_for_side(sr_multi, side, kind="tp")
    tp_zone, tp_dist_atr = _select_sr_zone(tp_zones, price, side, is_sl=False, atr_val=atr_val, cfg=sr_tp_cfg)
    if tp_zone is None:
        raw_signal["_reject_reason"] = "TP_ZONE_NOT_FOUND"
        return None

    sl_buffer = sr_sl_cfg.get("sl_buffer_atr", 0.3)
    tp_buffer = sr_tp_cfg.get("tp_buffer_atr", 0.5)

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

    rr = tp_abs / sl_abs

    min_tp_abs = cfg_risk.get("min_tp_abs")
    if min_tp_abs is not None and tp_abs < min_tp_abs:
        raw_signal["_reject_reason"] = "TP_TOO_CLOSE_ABS"
        raw_signal["_reject_tp_abs"] = tp_abs
        return None

    min_rr = cfg_risk.get("min_rr") or cfg_risk.get("min_rr_final")
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

    if rr >= rr_bonus_if_ge:
        score += rr_bonus_points

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
        "bias": bias,
        "timeframe_entry": "5m",
        "risk_mode": "SR_MULTI",
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
    }

    return full_signal


# ===========================
# MAIN – czytanie snapshotów CSV i generowanie sygnałów CSV
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", default="brent_snapshots_5m.csv")
    parser.add_argument("--config_json", required=True, help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None, help="opcjonalny override instrumentu")
    parser.add_argument("--debug_csv", default="signals_debug_log.csv",
                        help="plik CSV z powodami braku sygnału / debugiem")
    parser.add_argument("--stats_csv", default="signals_debug_stats.csv",
                        help="plik CSV z agregacją powodów SKIP")

    args = parser.parse_args()

    # config strategii
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "")
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

    for i, row in enumerate(rows):
        processed += 1
        progress = (i + 1) / max(total, 1) * 100.0
        snap_instrument = row.get("instrument")
        instrument = args.instrument or snap_instrument or "BZ=F"
        run_at = row.get("run_at")

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = m5_part.get("ts") or daily_part.get("ts") or run_at
        price = m5_part.get("close")
        bias = None

        print(
            f"[{progress:6.2f}%] snap_ts={snap_ts}, price={price}",
            flush=True,
        )

        def add_debug(stage, result, reason, extra=""):
            debug_rows.append({
                "snap_index": i,
                "snap_ts": snap_ts,
                "price": price,
                "bias": bias,
                "stage": stage,
                "result": result,
                "reason": reason,
                "extra": extra,
            })

        # 1) DAILY BIAS
        bias = determine_daily_bias(daily_part, cfg_daily)
        if bias == "NONE":
            extra = (
                f"ema20={daily_part.get('ema20')}, "
                f"ema50={daily_part.get('ema50')}, "
                f"rsi_pct={daily_part.get('rsi14_pct_rank')}, "
                f"atr_pct={daily_part.get('atr14_pct_rank')}, "
                f"slope_pct={daily_part.get('ema50_slope_pct')}"
            )
            print("  -> SKIP: DAILY_BIAS_NONE (" + extra + ")")
            add_debug(stage="DAILY", result="SKIP", reason="DAILY_BIAS_NONE", extra=extra)
            continue

        # 2) H1 FILTER
        if not check_h1_setup(h1_part, cfg_h1, bias):
            extra = (
                f"h1_close={h1_part.get('close')}, "
                f"h1_ema={h1_part.get('ema')}, "
                f"h1_rsi_pct={h1_part.get('rsi14_pct_rank')}, "
                f"h1_last_low={h1_part.get('last_low')}, "
                f"h1_prev_low={h1_part.get('prev_low')}, "
                f"h1_last_high={h1_part.get('last_high')}, "
                f"h1_prev_high={h1_part.get('prev_high')}"
            )
            print("  -> SKIP: H1_FILTER_NOT_PASSED (" + extra + ")")
            add_debug(stage="H1", result="SKIP", reason="H1_FILTER_NOT_PASSED", extra=extra)
            continue

        # 3) 5M TRIGGER
        raw_signal, m5_debug = check_5m_trigger(m5_part, cfg_m5, bias)
        if not raw_signal:
            reason = (m5_debug or {}).get("reason")
            total_score = (m5_debug or {}).get("total_score")
            min_score = (m5_debug or {}).get("min_score")
            extra = (
                f"m5_reason={reason}, "
                f"score={total_score}, min_score={min_score}, "
                f"m5_rsi={m5_part.get('rsi14')}, "
                f"m5_rsi_pct={m5_part.get('rsi14_pct_rank')}, "
                f"m5_prev_rsi_pct={m5_part.get('prev_rsi14_pct_rank')}, "
                f"m5_atr={m5_part.get('atr14')}, "
                f"m5_atr_pct={m5_part.get('atr14_pct_rank')}"
            )
            print("  -> SKIP: M5_TRIGGER_NOT_GENERATED (" + extra + ")")
            add_debug(stage="M5", result="SKIP",
                      reason=reason or "M5_TRIGGER_NOT_GENERATED",
                      extra=extra)
            continue

        # 4) RISK / SR_MULTI
        full_signal = build_signal_with_risk(
            raw_signal=raw_signal,
            bias=bias,
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
            extra_str = ", ".join(extra_parts)
            print(
                f"  -> SKIP: RISK_REJECTED (reason={reject_reason}"
                f"{', ' if extra_str else ''}{extra_str})"
            )
            add_debug(stage="RISK", result="SKIP",
                      reason=reject_reason or "RISK_REJECTED",
                      extra=extra_str)
            continue

        generated += 1
        extra_ok = (
            f"side={full_signal['side']}, "
            f"price={full_signal['entry_price']}, "
            f"rr={full_signal['rr']:.2f}, "
            f"score={full_signal['score']:.1f}, "
            f"m5_score={full_signal.get('m5_score')}"
        )
        print("  -> OK SIGNAL: " + extra_ok)
        add_debug(stage="FINAL", result="OK", reason="OK_SIGNAL", extra=extra_ok)

        signals.append(full_signal)

    print(f"Done. Processed snapshots={processed}, generated signals={generated}.")

    # zapis sygnałów do CSV
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

    # zapis debug logu do CSV (per snapshot)
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

    # ======= STATS: agregacja powodów SKIP =======
    reason_counts = {}
    for r in debug_rows:
        if r.get("result") != "SKIP":
            continue
        reason = r.get("reason") or "UNKNOWN"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # zapisujemy w formacie:
    # REASON;COUNT
    with open(args.stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            writer.writerow([reason, count])

    print(f"Stats saved to: {args.stats_csv}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", default="brent_snapshots_5m.csv")
    parser.add_argument("--config_json", required=True, help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None, help="opcjonalny override instrumentu")
    parser.add_argument("--debug_csv", default="signals_debug_log.csv",
                        help="plik CSV z powodami braku sygnału / debugiem")

    args = parser.parse_args()

    # config strategii
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "")
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

    for i, row in enumerate(rows):
        processed += 1
        progress = (i + 1) / max(total, 1) * 100.0
        snap_instrument = row.get("instrument")
        instrument = args.instrument or snap_instrument or "BZ=F"
        run_at = row.get("run_at")

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = m5_part.get("ts") or daily_part.get("ts") or run_at
        price = m5_part.get("close")
        bias = None

        print(
            f"[{progress:6.2f}%] snap_ts={snap_ts}, price={price}",
            flush=True,
        )

        # helper do debug_rows
        def add_debug(stage, result, reason, extra=""):
            debug_rows.append({
                "snap_index": i,
                "snap_ts": snap_ts,
                "price": price,
                "bias": bias,
                "stage": stage,
                "result": result,
                "reason": reason,
                "extra": extra,
            })

        # 1) DAILY BIAS
        bias = determine_daily_bias(daily_part, cfg_daily)
        if bias == "NONE":
            extra = (
                f"ema20={daily_part.get('ema20')}, "
                f"ema50={daily_part.get('ema50')}, "
                f"rsi_pct={daily_part.get('rsi14_pct_rank')}, "
                f"atr_pct={daily_part.get('atr14_pct_rank')}, "
                f"slope_pct={daily_part.get('ema50_slope_pct')}"
            )
            print("  -> SKIP: DAILY_BIAS_NONE (" + extra + ")")
            add_debug(stage="DAILY", result="SKIP", reason="DAILY_BIAS_NONE", extra=extra)
            continue

        # 2) H1 FILTER
        if not check_h1_setup(h1_part, cfg_h1, bias):
            extra = (
                f"h1_close={h1_part.get('close')}, "
                f"h1_ema={h1_part.get('ema')}, "
                f"h1_rsi_pct={h1_part.get('rsi14_pct_rank')}, "
                f"h1_last_low={h1_part.get('last_low')}, "
                f"h1_prev_low={h1_part.get('prev_low')}, "
                f"h1_last_high={h1_part.get('last_high')}, "
                f"h1_prev_high={h1_part.get('prev_high')}"
            )
            print("  -> SKIP: H1_FILTER_NOT_PASSED (" + extra + ")")
            add_debug(stage="H1", result="SKIP", reason="H1_FILTER_NOT_PASSED", extra=extra)
            continue

        # 3) 5M TRIGGER
        raw_signal, m5_debug = check_5m_trigger(m5_part, cfg_m5, bias)
        if not raw_signal:
            reason = (m5_debug or {}).get("reason")
            total_score = (m5_debug or {}).get("total_score")
            min_score = (m5_debug or {}).get("min_score")
            extra = (
                f"m5_reason={reason}, "
                f"score={total_score}, min_score={min_score}, "
                f"m5_rsi={m5_part.get('rsi14')}, "
                f"m5_rsi_pct={m5_part.get('rsi14_pct_rank')}, "
                f"m5_prev_rsi_pct={m5_part.get('prev_rsi14_pct_rank')}, "
                f"m5_atr={m5_part.get('atr14')}, "
                f"m5_atr_pct={m5_part.get('atr14_pct_rank')}"
            )
            print("  -> SKIP: M5_TRIGGER_NOT_GENERATED (" + extra + ")")
            add_debug(stage="M5", result="SKIP",
                      reason=reason or "M5_TRIGGER_NOT_GENERATED",
                      extra=extra)
            continue

        # 4) RISK / SR_MULTI
        full_signal = build_signal_with_risk(
            raw_signal=raw_signal,
            bias=bias,
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
            extra_str = ", ".join(extra_parts)
            print(
                f"  -> SKIP: RISK_REJECTED (reason={reject_reason}"
                f"{', ' if extra_str else ''}{extra_str})"
            )
            add_debug(stage="RISK", result="SKIP",
                      reason=reject_reason or "RISK_REJECTED",
                      extra=extra_str)
            continue

        generated += 1
        extra_ok = (
            f"side={full_signal['side']}, "
            f"price={full_signal['entry_price']}, "
            f"rr={full_signal['rr']:.2f}, "
            f"score={full_signal['score']:.1f}, "
            f"m5_score={full_signal.get('m5_score')}"
        )
        print("  -> OK SIGNAL: " + extra_ok)
        add_debug(stage="FINAL", result="OK", reason="OK_SIGNAL", extra=extra_ok)

        signals.append(full_signal)

    print(f"Done. Processed snapshots={processed}, generated signals={generated}.")

    # zapis sygnałów do CSV
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

    # zapis debug logu do CSV
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", default="brent_snapshots_5m.csv")
    parser.add_argument("--config_json", required=True, help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None, help="opcjonalny override instrumentu")

    args = parser.parse_args()

    # config strategii
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "")
    strategy_hash = compute_strategy_hash(config)
    cfg_daily = config.get("daily", {})
    cfg_h1 = config.get("h1", {})
    cfg_m5 = config.get("m5", {})
    cfg_risk = config.get("risk", {})

    print(f"Strategy name: {strategy_name}, hash={strategy_hash}")

    # wczytujemy snapshoty
    with open(args.snapshots_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} snapshots from {args.snapshots_csv}")

    signals = []
    processed = 0
    generated = 0

    for i, row in enumerate(rows):
        processed += 1
        progress = (i + 1) / max(total, 1) * 100.0
        snap_instrument = row.get("instrument")
        instrument = args.instrument or snap_instrument or "BZ=F"
        run_at = row.get("run_at")

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = m5_part.get("ts") or daily_part.get("ts") or run_at
        price = m5_part.get("close")

        print(
            f"[{progress:6.2f}%] snap_ts={snap_ts}, price={price}",
            flush=True,
        )

        # 1) DAILY BIAS
        bias = determine_daily_bias(daily_part, cfg_daily)
        if bias == "NONE":
            print(
                f"  -> SKIP: DAILY_BIAS_NONE "
                f"(ema20={daily_part.get('ema20')}, "
                f"ema50={daily_part.get('ema50')}, "
                f"rsi_pct={daily_part.get('rsi14_pct_rank')}, "
                f"atr_pct={daily_part.get('atr14_pct_rank')}, "
                f"slope_pct={daily_part.get('ema50_slope_pct')})"
            )
            continue

        # 2) H1 FILTER
        if not check_h1_setup(h1_part, cfg_h1, bias):
            print(
                f"  -> SKIP: H1_FILTER_NOT_PASSED "
                f"(bias={bias}, "
                f"h1_close={h1_part.get('close')}, "
                f"h1_ema={h1_part.get('ema')}, "
                f"h1_rsi_pct={h1_part.get('rsi14_pct_rank')}, "
                f"h1_last_low={h1_part.get('last_low')}, "
                f"h1_prev_low={h1_part.get('prev_low')}, "
                f"h1_last_high={h1_part.get('last_high')}, "
                f"h1_prev_high={h1_part.get('prev_high')})"
            )
            continue

        # 3) 5M TRIGGER
        raw_signal, m5_debug = check_5m_trigger(m5_part, cfg_m5, bias)
        if not raw_signal:
            reason = (m5_debug or {}).get("reason")
            total_score = (m5_debug or {}).get("total_score")
            min_score = (m5_debug or {}).get("min_score")
            print(
                f"  -> SKIP: M5_TRIGGER_NOT_GENERATED "
                f"(bias={bias}, reason={reason}, "
                f"score={total_score}, min_score={min_score}, "
                f"m5_rsi={m5_part.get('rsi14')}, "
                f"m5_rsi_pct={m5_part.get('rsi14_pct_rank')}, "
                f"m5_prev_rsi_pct={m5_part.get('prev_rsi14_pct_rank')}, "
                f"m5_atr={m5_part.get('atr14')}, "
                f"m5_atr_pct={m5_part.get('atr14_pct_rank')})"
            )
            continue

        # 4) RISK / SR_MULTI
        full_signal = build_signal_with_risk(
            raw_signal=raw_signal,
            bias=bias,
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
            extra = []
            if "_reject_rr" in raw_signal:
                extra.append(f"rr={raw_signal['_reject_rr']}")
            if "_reject_tp_abs" in raw_signal:
                extra.append(f"tp_abs={raw_signal['_reject_tp_abs']}")
            if "_reject_score" in raw_signal:
                extra.append(f"score={raw_signal['_reject_score']}")
            extra_str = ", ".join(extra) if extra else ""
            print(
                f"  -> SKIP: RISK_REJECTED "
                f"(reason={reject_reason}{', ' if extra_str else ''}{extra_str})"
            )
            continue

        generated += 1
        print(
            f"  -> OK SIGNAL: side={full_signal['side']}, "
            f"price={full_signal['entry_price']}, "
            f"rr={full_signal['rr']:.2f}, "
            f"score={full_signal['score']:.1f}, "
            f"m5_score={full_signal.get('m5_score')}"
        )

        signals.append(full_signal)

    print(f"Done. Processed snapshots={processed}, generated signals={generated}.")

    # zapis sygnałów do CSV
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", default="brent_snapshots_5m.csv")
    parser.add_argument("--config_json", required=True, help="plik z configiem strategii (JSON)")
    parser.add_argument("--output_signals_csv", default="signals_from_snapshots.csv")
    parser.add_argument("--instrument", default=None, help="opcjonalny override instrumentu")

    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    strategy_name = config.get("name", "")
    strategy_hash = compute_strategy_hash(config)
    cfg_daily = config.get("daily", {})
    cfg_h1 = config.get("h1", {})
    cfg_m5 = config.get("m5", {})
    cfg_risk = config.get("risk", {})

    print(f"Strategy name: {strategy_name}, hash={strategy_hash}")

    with open(args.snapshots_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} snapshots from {args.snapshots_csv}")

    signals = []

    for i, row in enumerate(rows):
        progress = (i + 1) / max(total, 1) * 100.0
        snap_instrument = row.get("instrument")
        instrument = args.instrument or snap_instrument or "BZ=F"
        run_at = row.get("run_at")

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = m5_part.get("ts") or daily_part.get("ts") or run_at
        print(f"[{progress:6.2f}%] snap_ts={snap_ts}, price={m5_part.get('close')}", flush=True)

        # 1) daily bias
        bias = determine_daily_bias(daily_part, cfg_daily)
        if bias == "NONE":
            continue

        # 2) H1 filter
        if not check_h1_setup(h1_part, cfg_h1, bias):
            continue

        # 3) 5m trigger
        raw_signal, m5_debug = check_5m_trigger(m5_part, cfg_m5, bias)
        if not raw_signal:
            continue

        # 4) risk / SR_MULTI – identycznie jak w Lambdzie (sr_multi z CSV)
        full_signal = build_signal_with_risk(
            raw_signal=raw_signal,
            bias=bias,
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
            continue

        signals.append(full_signal)

    print(f"Generated {len(signals)} signals.")

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


if __name__ == "__main__":
    main()
