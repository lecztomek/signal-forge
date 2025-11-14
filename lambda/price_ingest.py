# app.py
import os, json, ssl, time, random, urllib.request, urllib.parse, datetime as dt
import logging, math
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd

# ===== LOGGING =====
log = logging.getLogger()
log.setLevel(logging.INFO)

# ===== ENV (domyślna konfiguracja) =====
TABLE_NAME        = os.environ.get("DDB_TABLE_OHLC", "brent_ohlc")   # OHLC (5m,1h,4h,1d)
TICKS_TABLE_NAME  = os.environ.get("DDB_TABLE_TICKS", "brent_ticks") # ticki

SYMBOL      = os.environ.get("SYMBOL", "BZ=F")
FMP_SYMBOL  = os.environ.get("FMP_SYMBOL", "BZUSD")
FMP_API_KEY = os.environ["FMP_API_KEY"]

dynamodb    = boto3.resource("dynamodb")
TABLE       = dynamodb.Table(TABLE_NAME)
TICKS_TABLE = dynamodb.Table(TICKS_TABLE_NAME)

# ===== Helpers: Decimal =====
def to_decimal(x, default: str = "0") -> Decimal:
    if x is None:
        return Decimal(default)
    try:
        if isinstance(x, (float, int)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return Decimal(default)
            return Decimal(str(x))
        if pd.isna(x):
            return Decimal(default)
        return Decimal(str(x))
    except Exception:
        return Decimal(default)

# ===== OHLC helpers =====
def last_ts_iso(symbol: str, tf: str) -> str | None:
    """
    Ostatni timestamp (ISO) dla danej ramy czasowej (5m,1h,4h,1d).
    """
    pk = f"{symbol}#{tf}"
    resp = TABLE.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=False,  # najnowsze
        Limit=1,
    )
    items = resp.get("Items", [])
    return items[0]["ts"] if items else None

def upsert_batch(symbol: str, tf: str, df: pd.DataFrame) -> int:
    """
    Batch upsert OHLC do tabeli OHLC.
    pk = SYMBOL#tf, sort key = ts (ISO, bez mikrosekund).
    """
    if df is None or df.empty:
        return 0

    pk = f"{symbol}#{tf}"
    n = 0
    with TABLE.batch_writer(overwrite_by_pkeys=["pk", "ts"]) as b:
        for _, row in df.iterrows():
            ts_iso = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            item = {
                "pk": pk,
                "ts": ts_iso,
                "open":   to_decimal(row.get("open")),
                "high":   to_decimal(row.get("high")),
                "low":    to_decimal(row.get("low")),
                "close":  to_decimal(row.get("close")),
                "volume": to_decimal(row.get("volume"), default="0"),
            }
            src = row.get("src")
            if src is not None:
                item["src"] = str(src)
            b.put_item(Item=item)
            n += 1
    return n

def fetch_all_ohlc(symbol: str, tf: str) -> pd.DataFrame:
    """
    Pobiera wszystkie świece dla danego tf (5m,1h,4h,1d) z tabeli OHLC.
    Uwaga: dla dużej historii trzeba będzie dodać paginację (LastEvaluatedKey),
    tutaj dla prostoty bierzemy tylko pierwszą stronę.
    """
    pk = f"{symbol}#{tf}"
    resp = TABLE.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=True,  # od najstarszych
    )
    items = resp.get("Items", [])
    if not items:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    rows = []
    for it in items:
        rows.append({
            "timestamp": pd.to_datetime(it["ts"], utc=True),
            "open": float(it["open"]),
            "high": float(it["high"]),
            "low": float(it["low"]),
            "close": float(it["close"]),
            "volume": float(it.get("volume", 0)),
        })
    return pd.DataFrame(rows).sort_values("timestamp")

def filter_new(df: pd.DataFrame, last_iso: str | None) -> pd.DataFrame:
    if not last_iso or df.empty:
        return df
    last_ts = pd.to_datetime(last_iso, utc=True)
    return df[df["timestamp"] >= last_ts]  # zamiast >

# ===== Tick helpers =====
def insert_tick(symbol: str, df: pd.DataFrame) -> int:
    """
    Zapisuje ticki do tabeli ticków.
    pk = SYMBOL#tick, sort key = ts (ISO z mikrosekundami).
    """
    if df is None or df.empty:
        return 0

    pk = f"{symbol}#tick"
    n = 0
    with TICKS_TABLE.batch_writer() as b:
        for _, row in df.iterrows():
            ts_iso = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            item = {
                "pk": pk,
                "ts": ts_iso,
                "price": to_decimal(row.get("price")),
                "volume": to_decimal(row.get("volume"), default="0"),
            }
            src = row.get("src")
            if src is not None:
                item["src"] = str(src)
            b.put_item(Item=item)
            n += 1
    return n

def fetch_all_ticks(symbol: str) -> pd.DataFrame:
    """
    Pobiera wszystkie ticki z tabeli ticków.
    Uwaga: dla dużej ilości danych trzeba będzie zrobić pętlę po LastEvaluatedKey,
    tutaj dla prostoty bierzemy jedną stronę.
    (Nie jest używane w handlerze, zostawione np. do backfilli.)
    """
    pk = f"{symbol}#tick"
    resp = TICKS_TABLE.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=True,  # od najstarszych
    )
    items = resp.get("Items", [])
    if not items:
        return pd.DataFrame(columns=["timestamp","price","volume"])

    rows = []
    for it in items:
        rows.append({
            "timestamp": pd.to_datetime(it["ts"], utc=True),
            "price": float(it["price"]),
            "volume": float(it.get("volume", 0)),
        })
    return pd.DataFrame(rows).sort_values("timestamp")

def fetch_ticks_since(symbol: str, since_iso: str | None) -> pd.DataFrame:
    """
    Pobiera ticki od zadanego timestampu (ts >= since_iso).
    Jeśli since_iso = None, pobiera wszystkie ticki (jak fetch_all_ticks).
    Uwaga: nadal bez paginacji – jedna strona wyniku z DynamoDB.
    """
    pk = f"{symbol}#tick"

    if since_iso:
        resp = TICKS_TABLE.query(
            KeyConditionExpression=Key("pk").eq(pk) & Key("ts").gte(since_iso),
            ScanIndexForward=True,
        )
    else:
        resp = TICKS_TABLE.query(
            KeyConditionExpression=Key("pk").eq(pk),
            ScanIndexForward=True,
        )

    items = resp.get("Items", [])
    if not items:
        return pd.DataFrame(columns=["timestamp", "price", "volume"])

    rows = []
    for it in items:
        rows.append({
            "timestamp": pd.to_datetime(it["ts"], utc=True),
            "price": float(it["price"]),
            "volume": float(it.get("volume", 0)),
        })
    return pd.DataFrame(rows).sort_values("timestamp")

# ===== HTTP helper =====
def _http_get_json(url: str, max_retries: int = 3, base_sleep: float = 0.6) -> dict | list:
    ctx = ssl.create_default_context()
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Lambda/BrentFMP",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                sleep = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
                log.info({"retry_http": e.code, "sleep": round(sleep, 2), "attempt": attempt + 1})
                time.sleep(sleep)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                sleep = base_sleep * (2 ** attempt)
                log.info({"retry_urlerr": str(e), "sleep": round(sleep, 2), "attempt": attempt + 1})
                time.sleep(sleep)
                continue
            raise
    return {}

# ===== FMP: tick z /quote =====
def fmp_fetch_last_quote_tick(symbol: str) -> pd.DataFrame:
    """
    Pobiera real-time quote i zwraca jeden tick:
      price, volume, timestamp = now UTC.
    """
    params = urllib.parse.urlencode({"symbol": symbol, "apikey": FMP_API_KEY})
    url = f"https://financialmodelingprep.com/stable/quote?{params}"
    data = _http_get_json(url)

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["timestamp","price","volume","src"])

    q = data[0]
    price = q.get("price")
    if price is None:
        return pd.DataFrame(columns=["timestamp","price","volume","src"])

    vol = q.get("volume") or 0.0
    now = dt.datetime.now(dt.timezone.utc)

    df = pd.DataFrame([{
        "timestamp": now,
        "price": float(price),
        "volume": float(vol),
        "src": "fmp_quote",
    }])

    df = df.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], None)
    return df

# ===== Resampling =====
def ticks_to_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Ticki (price, volume) -> OHLC wg rule, np. '5min'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    x = df.set_index("timestamp").sort_index()
    r = pd.DataFrame()
    r["open"]   = x["price"].resample(rule, label="right", closed="right").first()
    r["high"]   = x["price"].resample(rule, label="right", closed="right").max()
    r["low"]    = x["price"].resample(rule, label="right", closed="right").min()
    r["close"]  = x["price"].resample(rule, label="right", closed="right").last()
    r["volume"] = x["volume"].resample(rule, label="right", closed="right").sum()
    return r.dropna().reset_index()

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    OHLC -> OHLC (np. 5m->1h, 1h->4h, 1h->1d).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    x = df.set_index("timestamp").sort_index()
    r = pd.DataFrame()
    r["open"]   = x["open"].resample(rule, label="right", closed="right").first()
    r["high"]   = x["high"].resample(rule, label="right", closed="right").max()
    r["low"]    = x["low"].resample(rule, label="right", closed="right").min()
    r["close"]  = x["close"].resample(rule, label="right", closed="right").last()
    r["volume"] = x["volume"].resample(rule, label="right", closed="right").sum()
    return r.dropna().reset_index()

# ===== Handler =====
def handler(event, context):
    """
    Event może nadpisać konfigurację, np.:

    Prosto:
    {
      "SYMBOL": "CL=F",
      "FMP_SYMBOL": "CLUSD",
      "DDB_TABLE_OHLC": "cl_ohlc",
      "DDB_TABLE_TICKS": "cl_ticks",
      "FMP_API_KEY": "xxx"
    }

    albo z sekcją config:
    {
      "config": {
        "SYMBOL": "CL=F",
        "FMP_SYMBOL": "CLUSD",
        "FMP_API_KEY": "xxx"
      }
    }
    """
    global TABLE_NAME, TICKS_TABLE_NAME, SYMBOL, FMP_SYMBOL, FMP_API_KEY, TABLE, TICKS_TABLE, dynamodb

    # Event może być None/albo nie-dict (CloudWatch test, itp.)
    if not isinstance(event, dict):
        event = {}

    # Pozwalamy na opakowanie w "config"
    cfg = event.get("config", event) or {}

    # Nadpisanie nazw tabel (jeśli przyszły)
    TABLE_NAME       = cfg.get("DDB_TABLE_OHLC", TABLE_NAME)
    TICKS_TABLE_NAME = cfg.get("DDB_TABLE_TICKS", TICKS_TABLE_NAME)

    # Nadpisanie symboli (duże i małe nazwy)
    SYMBOL     = cfg.get("SYMBOL", cfg.get("symbol", SYMBOL))
    FMP_SYMBOL = cfg.get("FMP_SYMBOL", cfg.get("fmp_symbol", FMP_SYMBOL))

    # Nadpisanie klucza API (jeśli przyjdzie w evencie)
    FMP_API_KEY = cfg.get("FMP_API_KEY", FMP_API_KEY)

    # Przebindowanie tabel pod aktualne nazwy
    TABLE       = dynamodb.Table(TABLE_NAME)
    TICKS_TABLE = dynamodb.Table(TICKS_TABLE_NAME)

    # Uwaga: NIE logujemy API key'a
    log.info({
        "step": "effective_config",
        "TABLE_NAME": TABLE_NAME,
        "TICKS_TABLE_NAME": TICKS_TABLE_NAME,
        "SYMBOL": SYMBOL,
        "FMP_SYMBOL": FMP_SYMBOL,
        # "FMP_API_KEY": "hidden"
    })

    # 1) Pobierz tick z FMP i zapisz do brent_ticks
    df_tick_last = fmp_fetch_last_quote_tick(FMP_SYMBOL)
    log.info({
        "step": "fetch_tick_fmp_quote",
        "fmp_symbol": FMP_SYMBOL,
        "fetched_count": int(len(df_tick_last)),
        "ts_fetched": None if df_tick_last.empty else str(df_tick_last["timestamp"].iloc[-1]),
    })

    if df_tick_last.empty:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "inserted_tick": 0,
                "inserted_5m": 0,
                "inserted_1h": 0,
                "inserted_4h": 0,
                "inserted_1d": 0,
                "note": "empty_fmp_quote"
            })
        }

    n_tick = insert_tick(SYMBOL, df_tick_last)
    log.info({"step": "insert_tick", "inserted_tick": int(n_tick)})

    # 2) Zbuduj 5m tylko z najnowszych ticków (od ostatniej świeczki 5m - 5 minut)
    last5 = last_ts_iso(SYMBOL, "5m")

    if last5:
        last5_dt = pd.to_datetime(last5, utc=True)
        since_dt = last5_dt - pd.Timedelta(minutes=5)
        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        # brak świeczek 5m w DDB -> ładujemy wszystkie ticki
        since_iso = None

    df_ticks_recent = fetch_ticks_since(SYMBOL, since_iso)
    log.info({
        "step": "fetch_ticks_recent",
        "last5_in_ddb": last5,
        "since_iso": since_iso,
        "ticks_count": int(len(df_ticks_recent)),
        "ts_tick_min": None if df_ticks_recent.empty else str(df_ticks_recent["timestamp"].min()),
        "ts_tick_max": None if df_ticks_recent.empty else str(df_ticks_recent["timestamp"].max()),
    })

    df5_full = ticks_to_ohlc(df_ticks_recent, "5min")
    df5_new  = filter_new(df5_full, last5)
    n5 = upsert_batch(SYMBOL, "5m", df5_new)
    log.info({
        "step": "build_5m",
        "last5_in_ddb": last5,
        "bars_5m_total": int(len(df5_full)),
        "bars_5m_new": int(len(df5_new)),
        "inserted_5m": int(n5),
    })

    # 3) 5m -> 1h
    df5_all = fetch_all_ohlc(SYMBOL, "5m")
    last1h = last_ts_iso(SYMBOL, "1h")
    df1h_full = resample_ohlc(df5_all, "60min")
    df1h_new  = filter_new(df1h_full, last1h)
    n1 = upsert_batch(SYMBOL, "1h", df1h_new)
    log.info({
        "step": "build_1h",
        "last1h_in_ddb": last1h,
        "bars_1h_total": int(len(df1h_full)),
        "bars_1h_new": int(len(df1h_new)),
        "inserted_1h": int(n1),
    })

    # 4) 1h -> 4h
    df1h_all = fetch_all_ohlc(SYMBOL, "1h")
    last4h = last_ts_iso(SYMBOL, "4h")
    df4h_full = resample_ohlc(df1h_all, "4h")  # małe 'h' zamiast 'H'
    df4h_new  = filter_new(df4h_full, last4h)
    n4 = upsert_batch(SYMBOL, "4h", df4h_new)
    log.info({
        "step": "build_4h",
        "last4h_in_ddb": last4h,
        "bars_4h_total": int(len(df4h_full)),
        "bars_4h_new": int(len(df4h_new)),
        "inserted_4h": int(n4),
    })

    # 5) 1h -> 1d
    last1d = last_ts_iso(SYMBOL, "1d")
    df1d_full = resample_ohlc(df1h_all, "1D")
    df1d_new  = filter_new(df1d_full, last1d)
    n1d = upsert_batch(SYMBOL, "1d", df1d_new)
    log.info({
        "step": "build_1d",
        "last1d_in_ddb": last1d,
        "bars_1d_total": int(len(df1d_full)),
        "bars_1d_new": int(len(df1d_new)),
        "inserted_1d": int(n1d),
    })

    result = {
        "inserted_tick": int(n_tick),
        "inserted_5m": int(n5),
        "inserted_1h": int(n1),
        "inserted_4h": int(n4),
        "inserted_1d": int(n1d),
    }
    log.info({"result": result})
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
