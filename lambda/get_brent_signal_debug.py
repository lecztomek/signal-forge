import os
import json
import decimal
from decimal import Decimal
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")

DEBUG_TABLE_NAME = os.getenv("DEBUG_TABLE_NAME", "brent_signals_debug")
INSTRUMENT = os.getenv("INSTRUMENT", "BZ=F")


def _to_float(x):
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return x


def handler(event, context):
    """
    Lambda wywoływana przez API Gateway (NON-proxy).
    Zwraca BEZPOŚREDNIO { "items": [...] }, tak żeby frontend mógł zrobić
    const json = await res.json(); json.items -> OK
    """
    print("[DEBUG] debug-api started, event=", event)

    try:
        params = (event.get("queryStringParameters") or {}) if isinstance(event, dict) else {}
        instrument = params.get("instrument") or INSTRUMENT
        limit = int(params.get("limit") or 20)

        table = dynamodb.Table(DEBUG_TABLE_NAME)
        pk_value = f"{instrument}#debug"

        resp = table.query(
            KeyConditionExpression=Key("pk").eq(pk_value),
            ScanIndexForward=False,  # najnowsze najpierw
            Limit=limit,
        )

        items = resp.get("Items", [])
        simplified = []

        for it in items:
            dbg = it.get("debug", it)

            daily = dbg.get("daily", {})
            h1    = dbg.get("h1", {})
            m5    = dbg.get("m5")
            sr    = dbg.get("sr", {})
            final = dbg.get("final", {})

            def _num(x):
                if isinstance(x, Decimal):
                    return float(x)
                if isinstance(x, (int, float)):
                    return float(x)
                return x

            support_levels = [_num(z.get("level")) for z in sr.get("support_zones", [])]
            resistance_levels = [_num(z.get("level")) for z in sr.get("resistance_zones", [])]

            run_at = dbg.get("run_at") or it.get("ts")

            simplified.append({
                "ts": run_at,
                "instrument": dbg.get("instrument", instrument),
                "has_signal": final.get("has_signal", False),
                "reason": final.get("reason", ""),
                "daily": {
                    "bias":  daily.get("bias"),
                    "close": _num(daily.get("close")),
                    "ema20": _num(daily.get("ema20")),
                    "ema50": _num(daily.get("ema50")),
                    "rsi14": _num(daily.get("rsi14")),
                },
                "h1": {
                    "setup_ok": h1.get("setup_ok", False),
                    "close":    _num(h1.get("close")),
                    "ema":      _num(h1.get("ema")),
                    "rsi14":    _num(h1.get("rsi14")),
                    "last_high": _num(h1.get("last_high")),
                    "prev_high": _num(h1.get("prev_high")),
                    "last_low":  _num(h1.get("last_low")),
                    "prev_low":  _num(h1.get("prev_low")),
                } if h1 else None,
                "sr": {
                    "support":    support_levels,
                    "resistance": resistance_levels,
                },
                "m5": {
                    "last_close": _num((m5 or {}).get("last_close")),
                    "last_rsi":   _num((m5 or {}).get("last_rsi")),
                    "prev_rsi":   _num((m5 or {}).get("prev_rsi")),
                    "atr":        _num((m5 or {}).get("atr")),
                    "recent_high": _num((m5 or {}).get("recent_high")),
                    "recent_low":  _num((m5 or {}).get("recent_low")),
                    "sr_level":     _num((m5 or {}).get("sr_level")),
                    "sr_dist_abs":  _num((m5 or {}).get("sr_dist_abs")),
                    "sr_dist_atr":  _num((m5 or {}).get("sr_dist_atr")),
                } if m5 else None,
            })

        simplified = list(reversed(simplified))

        # 👇 ZWRACAMY BEZPOŚREDNIO OBIEKT Z ITEMS
        return {
            "items": simplified,
        }

    except Exception as e:
        print("[ERROR] debug-api error:", e)
        # Front nadal może to ogarnąć: json.items || []
        return {
            "items": [],
            "error": str(e),
        }
