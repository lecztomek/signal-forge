#!/usr/bin/env python
import argparse
import json
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import numpy as np


# ==========================
#  Pomocnicze walidacje
# ==========================

def ensure_columns(df: pd.DataFrame, required: List[str], what: str = "df"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w {what}: {missing}")


# ==========================
#  Wczytywanie danych
# ==========================

def load_signals(path: str) -> pd.DataFrame:
    """
    Oczekiwany format (separator ';'):

    atr_higher_tf;atr_m5;atr_used;bias;entry_price;instrument;m5_score;risk_mode;rr;score;side;sl;sl_zone_dist_atr;sl_zone_freshness;sl_zone_level;sl_zone_strength;sl_zone_timeframe;sr_distance_atr;sr_level;sr_strength;strategy_config_json;strategy_hash;strategy_name;timeframe_entry;timestamp;tp;tp_zone_dist_atr;tp_zone_freshness;tp_zone_level;tp_zone_strength;tp_zone_timeframe
    """
    print(f"[INFO] Wczytuję sygnały z: {path}")
    signals = pd.read_csv(path, sep=";")

    required_signals = [
        "timestamp",
        "side",
        "entry_price",
        "sl",
        "tp",
        "instrument",
        "rr",
        "score",
        "m5_score",
        "risk_mode",
    ]
    ensure_columns(signals, required_signals, what="signals")

    # timestamp -> datetime (UTC)
    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"], utc=True, errors="coerce"
    )
    before = len(signals)
    signals = signals.dropna(subset=["timestamp"])
    if len(signals) < before:
        print(f"[WARN] Usunięto {before - len(signals)} sygnałów z niepoprawnym timestampem.")

    # Na wszelki wypadek rzutujemy numeric
    for col in ["entry_price", "sl", "tp", "rr", "score", "m5_score"]:
        if col in signals.columns:
            signals[col] = pd.to_numeric(signals[col], errors="coerce")

    signals = signals.sort_values("timestamp").reset_index(drop=True)
    return signals


def load_candles_brent_5m(path: str) -> pd.DataFrame:
    """
    Oczekiwany format świece 5m (brak nagłówka):

    date;time;open;high;low;close;volume
    26/01/2009;21:05:00;46.96;46.96;46.96;46.96;14
    ...

    Daty w formacie DD/MM/YYYY.
    """
    print(f"[INFO] Wczytuję świece z: {path}")
    candles = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )

    # Zbuduj timestamp
    candles["timestamp"] = pd.to_datetime(
        candles["date"] + " " + candles["time"],
        dayfirst=True,
        utc=True,
        errors="coerce",
    )
    before = len(candles)
    candles = candles.dropna(subset=["timestamp"])
    if len(candles) < before:
        print(f"[WARN] Usunięto {before - len(candles)} świec z niepoprawnym timestampem.")

    # Rzut na float
    for col in ["open", "high", "low", "close"]:
        candles[col] = pd.to_numeric(candles[col], errors="coerce")

    candles = candles.dropna(subset=["open", "high", "low", "close"])
    candles = candles.sort_values("timestamp").reset_index(drop=True)
    return candles


# ==========================
#  Backtest pojedynczego trade’a
# ==========================

@dataclass
class TradeResult:
    signal_index: int
    instrument: str
    side: str
    entry_ts: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    exit_ts: Optional[pd.Timestamp]
    exit_price: Optional[float]
    outcome: str  # WIN / LOSS / LOSS_BOTH_HIT / NO_HIT / NO_CANDLES
    R: float
    bars_used: int
    rr_planned: float
    score: float
    m5_score: float
    risk_mode: str


def simulate_trade(
    idx: int,
    sig: pd.Series,
    candles: pd.DataFrame,
    max_bars: int = 500,
) -> TradeResult:
    side = str(sig["side"]).upper()
    entry_ts = sig["timestamp"]
    entry_price = float(sig["entry_price"])
    sl = float(sig["sl"])
    tp = float(sig["tp"])
    rr_planned = float(sig.get("rr", np.nan))
    score = float(sig.get("score", np.nan))
    m5_score = float(sig.get("m5_score", np.nan))
    risk_mode = str(sig.get("risk_mode", ""))

    # Świece po sygnale (łącznie z tą samą minutą)
    cdf = candles[candles["timestamp"] >= entry_ts]
    if cdf.empty:
        return TradeResult(
            signal_index=idx,
            instrument=str(sig.get("instrument", "")),
            side=side,
            entry_ts=entry_ts,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            exit_ts=None,
            exit_price=None,
            outcome="NO_CANDLES",
            R=0.0,
            bars_used=0,
            rr_planned=rr_planned,
            score=score,
            m5_score=m5_score,
            risk_mode=risk_mode,
        )

    if max_bars is not None:
        cdf = cdf.iloc[:max_bars]

    outcome = "NO_HIT"
    exit_ts = None
    exit_price = None
    R = 0.0
    bars_used = 0

    # Odległość SL w cenie => 1R
    if side == "BUY":
        risk_per_unit = abs(entry_price - sl)
    else:
        risk_per_unit = abs(sl - entry_price)
    if risk_per_unit <= 0:
        # Bezsensowny SL, traktujemy jako NO_HIT
        risk_per_unit = np.nan

    for i, (_, bar) in enumerate(cdf.iterrows(), start=1):
        low = float(bar["low"])
        high = float(bar["high"])

        if side == "BUY":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:  # SELL
            hit_sl = high >= sl
            hit_tp = low <= tp

        if hit_sl and hit_tp:
            # Konserwatywnie zakładamy, że SL poszedł pierwszy
            outcome = "LOSS_BOTH_HIT"
            exit_ts = bar["timestamp"]
            exit_price = sl
            R = -1.0 if risk_per_unit > 0 else 0.0
            bars_used = i
            break
        elif hit_tp:
            outcome = "WIN"
            exit_ts = bar["timestamp"]
            exit_price = tp
            if risk_per_unit > 0:
                if side == "BUY":
                    R = (tp - entry_price) / risk_per_unit
                else:
                    R = (entry_price - tp) / risk_per_unit
            bars_used = i
            break
        elif hit_sl:
            outcome = "LOSS"
            exit_ts = bar["timestamp"]
            exit_price = sl
            R = -1.0 if risk_per_unit > 0 else 0.0
            bars_used = i
            break

    if outcome in ("NO_HIT", "NO_CANDLES"):
        # Jeśli nie dotknęło TP/SL, liczymy R na ostatniej świecy (mark-to-market)
        last_bar = cdf.iloc[-1]
        last_price = float(last_bar["close"])
        exit_ts = last_bar["timestamp"]
        exit_price = last_price
        if risk_per_unit > 0:
            if side == "BUY":
                R = (last_price - entry_price) / risk_per_unit
            else:
                R = (entry_price - last_price) / risk_per_unit
        bars_used = len(cdf)

    return TradeResult(
        signal_index=idx,
        instrument=str(sig.get("instrument", "")),
        side=side,
        entry_ts=entry_ts,
        entry_price=entry_price,
        sl=sl,
        tp=tp,
        exit_ts=exit_ts,
        exit_price=exit_price,
        outcome=outcome,
        R=R,
        bars_used=bars_used,
        rr_planned=rr_planned,
        score=score,
        m5_score=m5_score,
        risk_mode=risk_mode,
    )


# ==========================
#  Agregacja statystyk
# ==========================

def summarize_results(trades_df: pd.DataFrame):
    print("\n========== PODSUMOWANIE ==========")
    n_all = len(trades_df)
    print(f"Liczba sygnałów: {n_all}")

    closed_mask = trades_df["outcome"].isin(["WIN", "LOSS", "LOSS_BOTH_HIT"])
    closed = trades_df[closed_mask]
    n_closed = len(closed)
    print(f"Zamknięte trady (TP/SL): {n_closed}")

    wins = closed[closed["outcome"] == "WIN"]
    losses = closed[closed["outcome"].isin(["LOSS", "LOSS_BOTH_HIT"])]

    n_wins = len(wins)
    n_losses = len(losses)

    print(f"Wygrane (TP): {n_wins}")
    print(f"Przegrane (SL): {n_losses}")

    if n_closed > 0:
        winrate = n_wins / n_closed * 100.0
        print(f"Winrate (TP / zamknięte): {winrate:.2f}%")

    print(f"NO_HIT: {len(trades_df[trades_df['outcome'] == 'NO_HIT'])}")
    print(f"NO_CANDLES: {len(trades_df[trades_df['outcome'] == 'NO_CANDLES'])}")

    if n_closed > 0:
        print("\n--- Statystyki R (tylko zamknięte) ---")
        print(f"Średnie R: {closed['R'].mean():.3f}")
        print(f"Mediana R: {closed['R'].median():.3f}")
        print(f"Suma R: {closed['R'].sum():.3f}")
        print(f"Max R: {closed['R'].max():.3f}")
        print(f"Min R: {closed['R'].min():.3f}")

    # Można też dorzucić breakdown po risk_mode
    if "risk_mode" in trades_df.columns:
        print("\n--- Winrate po risk_mode (zamknięte) ---")
        for rm, group in closed.groupby("risk_mode"):
            if len(group) == 0:
                continue
            w = len(group[group["outcome"] == "WIN"])
            wr = w / len(group) * 100.0
            print(f"{rm or 'NONE'}: n={len(group)}, winrate={wr:.2f}%")


# ==========================
#  Główny main
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest strategii z pliku sygnałów + świece 5m (brent_5m format)."
    )
    parser.add_argument("--signals", required=True, help="CSV z sygnałami")
    parser.add_argument("--candles", required=True, help="CSV ze świecami 5m (brent format)")
    parser.add_argument(
        "--out-trades",
        required=False,
        default="trades_results.csv",
        help="Plik wynikowy z trade'ami",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=500,
        help="Maksymalna liczba świec po sygnale (0 = bez limitu)",
    )

    args = parser.parse_args()

    signals = load_signals(args.signals)
    candles = load_candles_brent_5m(args.candles)

    if args.max_bars <= 0:
        max_bars = None
    else:
        max_bars = args.max_bars

    print(f"[INFO] Liczba sygnałów: {len(signals)}")
    print(f"[INFO] Liczba świec: {len(candles)}")

    trade_results: List[TradeResult] = []

    for idx, sig in signals.iterrows():
        tr = simulate_trade(idx, sig, candles, max_bars=max_bars)
        trade_results.append(tr)

        if (idx + 1) % 50 == 0 or idx == len(signals) - 1:
            print(f"[INFO] Przetworzono {idx + 1}/{len(signals)} sygnałów")

    trades_df = pd.DataFrame([t.__dict__ for t in trade_results])

    # Zapis wyników
    out_path = args.out_trades
    trades_df.to_csv(out_path, index=False, sep=";")
    print(f"[INFO] Zapisano wyniki trade'ów do: {out_path}")

    summarize_results(trades_df)


if __name__ == "__main__":
    main()
