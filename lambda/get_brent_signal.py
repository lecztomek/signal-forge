import os
import json
import decimal
from decimal import Decimal
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")

SIGNALS_TABLE_NAME = os.getenv("SIGNALS_TABLE_NAME", "brent_signals")
INSTRUMENT = os.getenv("INSTRUMENT", "BZ=F")


def _num(x):
    """Konwersja Decimal -> float, int/float -> float, reszta bez zmian."""
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return x


def handler(event, context):
    """
    Lambda wywoływana przez API Gateway (NON-proxy).
    Zwraca BEZPOŚREDNIO { "items": [...] }, tak żeby frontend mógł zrobić:
      const json = await res.json();
      json.items -> OK
    """
    print("[DEBUG] signals-api started, event=", event)

    try:
        params = (event.get("queryStringParameters") or {}) if isinstance(event, dict) else {}
        instrument = params.get("instrument") or INSTRUMENT
        limit = int(params.get("limit") or 30)

        table = dynamodb.Table(SIGNALS_TABLE_NAME)
        pk_value = f"{instrument}#signal"

        # Najnowsze sygnały najpierw (ScanIndexForward=False)
        resp = table.query(
            KeyConditionExpression=Key("pk").eq(pk_value),
            ScanIndexForward=False,
            Limit=limit,
        )

        items = resp.get("Items", [])
        simplified = []

        for it in items:
            simplified.append({
                # sort key / czas świecy 5m – do sortowania i wyświetlania
                "ts": it.get("ts"),

                # podstawowe informacje o sygnale
                "instrument": it.get("instrument", instrument),
                "side": it.get("side"),                # BUY / SELL
                "bias": it.get("bias"),                # LONG / SHORT
                "timeframe_entry": it.get("timeframe_entry"),  # 5m itd.

                # poziomy cenowe (Decimal -> float)
                "entry_price": _num(it.get("entry_price")),
                "sl": _num(it.get("sl")),
                "tp": _num(it.get("tp")),
                "rr": _num(it.get("rr")),
                "score": _num(it.get("score")),

                # informacje o strefie S/R (opcjonalne)
                "sr_level": _num(it.get("sr_level")) if "sr_level" in it else None,
                "sr_strength": _num(it.get("sr_strength")) if "sr_strength" in it else None,
                "sr_distance_atr": _num(it.get("sr_distance_atr")) if "sr_distance_atr" in it else None,

                # meta
                "created_at": it.get("created_at"),
            })

        # jeśli wolisz rosnąco, możesz odkomentować:
        # simplified = list(reversed(simplified))

        return {
            "items": simplified,
        }

    except Exception as e:
        print("[ERROR] signals-api error:", e)
        return {
            "items": [],
            "error": str(e),
        }
