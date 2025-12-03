#!/usr/bin/env python
"""
strategy_backtest.py

Backtest "na żywo" bar-po-barze na snapshotach M5:

Wejścia:
    --snapshots-enriched       : wzbogacone snapshoty TEST (z groupera)
    --entry-setups-long/short  : wybrane setupy wejścia (good)
    --ranked-long/short        : posortowane wyniki exit_lab_v3_*_ranked
    --signatures-long/short    : exit_signatures_* (opcjonalne, do signature gates)
    --atr-col, --risk-mult, --rsi-col

Wyjścia:
    - trades.csv
    - trades_summary.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpery
# ---------------------------------------------------------------------------

GROUP_COLS: List[str] = [
    "instrument",
    "daily_trend",
    "daily_vol_bucket",
    "daily_rsi_state",
    "h1_trend",
    "h1_slope_state",
    "h1_rsi_state",
    "h1_position_vs_swings",
    "m5_rsi_state",
    "m5_pos_in_recent_range",
    "m5_vol_bucket",
    "near_support",
    "near_resistance",
    "session_bucket",
]


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[strategy_backtest] WARN: plik {path} nie istnieje.")
        return None
    df = pd.read_csv(path, sep=";")
    if df.empty:
        print(f"[strategy_backtest] WARN: plik {path} jest pusty.")
        return None
    return df


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Brak kolumny czasu '{col}' w snapshots_enriched.")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col] = df[col].dt.tz_convert(None)
    return df


def build_setup_merge(df_snap: pd.DataFrame, setups: Optional[pd.DataFrame],
                      side: str) -> pd.DataFrame:
    """
    Dodaje do df_snap:
      - setup_id_{side}
      - entry_score_{side}
    na podstawie merge'u po GROUP_COLS (tak jak w grouperze).
    """
    if setups is None or setups.empty:
        return df_snap

    side = side.lower()
    setups = setups.copy()

    # wspólne kolumny grupujące
    merge_cols = [c for c in GROUP_COLS if c in setups.columns and c in df_snap.columns]
    if not merge_cols:
        print(
            f"[strategy_backtest] WARN: brak wspólnych kolumn grupujących dla setupów {side}. "
            f"Nie będzie żadnych wejść {side}."
        )
        return df_snap

    # prosty setup_id
    setups = setups.reset_index(drop=True)
    setups[f"setup_id_{side}"] = setups.index

    # entry_score – long_entry_score / short_entry_score, a jak nie ma, to 1.0
    score_col = f"{side}_entry_score"
    if score_col not in setups.columns:
        print(
            f"[strategy_backtest] INFO: w setupach {side} brak kolumny {score_col}, "
            f"ustawiam entry_score_{side} = 1.0."
        )
        setups[score_col] = 1.0

    use_cols = merge_cols + [f"setup_id_{side}", score_col]
    setups_small = setups[use_cols].drop_duplicates()

    df_merged = df_snap.merge(
        setups_small,
        on=merge_cols,
        how="left",
    )
    df_merged[f"entry_score_{side}"] = df_merged[score_col]
    return df_merged


def build_signature_flag(
    df_snap: pd.DataFrame,
    signatures: Optional[pd.DataFrame],
    side: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Buduje flagę sig_ok_{side} na DF snapshotów na podstawie exit_signatures_*:

    - wyszukuje wspólne kolumny kategoryczne (object/category) pomiędzy df_snap a signatures,
    - dla każdej kombinacji tych kolumn, która WYSTĘPUJE w signatures,
      ustawiamy sig_ok_{side} = True,
    - reszta = False.

    Jeśli nie ma wspólnych kolumn albo signatures jest None/puste -> zwraca df_snap bez flagi.
    """
    if signatures is None or signatures.empty:
        return df_snap, None

    side = side.lower()
    sig = signatures.copy()

    # wybieramy kolumny kandydujące na "stan rynku":
    # - są zarazem w snapshots i w signatures
    # - i są typu kategorycznego / stringowego
    candidate_cols = []
    for c in sig.columns:
        if c in df_snap.columns:
            if sig[c].dtype == "object" or isinstance(sig[c].dtype, pd.CategoricalDtype):
                candidate_cols.append(c)

    gate_cols = candidate_cols
    if not gate_cols:
        print(
            f"[strategy_backtest] INFO: exit_signatures_{side} nie ma wspólnych kolumn "
            f"kategorycznych z snapshots – signature gates dla {side} wyłączone."
        )
        return df_snap, None

    sig_keys = sig[gate_cols].dropna().drop_duplicates()
    if sig_keys.empty:
        print(
            f"[strategy_backtest] INFO: exit_signatures_{side} ma 0 kombinacji po gate_cols – "
            f"signature gates dla {side} wyłączone."
        )
        return df_snap, None

    flag_col = f"sig_ok_{side}"
    sig_keys[flag_col] = True

    df_merged = df_snap.merge(sig_keys, on=gate_cols, how="left")
    df_merged[flag_col] = df_merged[flag_col].fillna(False)

    print(
        f"[strategy_backtest] INFO: signature gating dla {side} używa kolumn: "
        f"{', '.join(gate_cols)}"
    )
    return df_merged, flag_col


@dataclass
class ExitRule:
    min_target_R: float
    trail_giveback_R: float
    stop_R: float
    max_bars: int
    rsi_exit_threshold: float
    rsi_min_R: float
    sr_dist_threshold_atr: float


def pick_exit_rule(ranked_df: Optional[pd.DataFrame],
                   rule_index: int,
                   side: str) -> Optional[ExitRule]:
    if ranked_df is None or ranked_df.empty:
        print(f"[strategy_backtest] WARN: brak ranked exits dla {side}.")
        return None

    if rule_index < 0 or rule_index >= len(ranked_df):
        print(
            f"[strategy_backtest] WARN: rule_index {rule_index} poza zakresem dla {side} "
            f"(n={len(ranked_df)}) – używam indeksu 0."
        )
        rule_index = 0

    row = ranked_df.iloc[rule_index]

    def get(col, default):
        return float(row[col]) if col in row and not pd.isna(row[col]) else default

    rule = ExitRule(
        min_target_R=get("min_target_R", 1.0),
        trail_giveback_R=get("trail_giveback_R", 1.0),
        stop_R=get("stop_R", -1.0),
        max_bars=int(get("max_bars", 100)),
        rsi_exit_threshold=get("rsi_exit_threshold", 50.0),
        rsi_min_R=get("rsi_min_R", 1.0),
        sr_dist_threshold_atr=get("sr_dist_threshold_atr", -1.0),
    )

    print(f"[strategy_backtest] Wybrana reguła exitu ({side}): {rule}")
    return rule


# ---------------------------------------------------------------------------
# Główna pętla backtestu
# ---------------------------------------------------------------------------

@dataclass
class Position:
    side: str  # "long" / "short"
    entry_ts: pd.Timestamp
    entry_price: float
    atr_entry: float
    setup_id: Optional[int]
    setup_entry_score: float
    rule: ExitRule

    R: float = 0.0
    MFE_R: float = 0.0
    MAE_R: float = 0.0
    bars_in_trade: int = 0


def compute_R(side: str, price: float, entry_price: float, atr_entry: float,
              risk_mult: float) -> float:
    if atr_entry <= 0 or not np.isfinite(atr_entry):
        return 0.0
    if side == "long":
        return (price - entry_price) / (atr_entry * risk_mult)
    else:  # short
        return (entry_price - price) / (atr_entry * risk_mult)


def backtest(
    df: pd.DataFrame,
    atr_col: str,
    risk_mult: float,
    rsi_col: str,
    long_rule: Optional[ExitRule],
    short_rule: Optional[ExitRule],
    sig_flag_long: Optional[str],
    sig_flag_short: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Główna pętla backtestu po DF snapshotów (TEST).
    """
    required_cols = ["m5_ts", "m5_close", atr_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Brak kolumny '{c}' w snapshots_enriched.")

    df = df.sort_values("m5_ts").reset_index(drop=True)

    trades = []
    pos: Optional[Position] = None

    for _, row in df.iterrows():
        ts = row["m5_ts"]
        price = row["m5_close"]
        atr_val = row[atr_col]

        # --- jeśli nie ma pozycji: sprawdź wejścia ---
        if pos is None:
            candidate_sides = []

            # LONG
            if (
                long_rule is not None
                and "setup_id_long" in row
                and not pd.isna(row["setup_id_long"])
            ):
                score = float(row.get("entry_score_long", 1.0))
                candidate_sides.append(("long", score, int(row["setup_id_long"])))

            # SHORT
            if (
                short_rule is not None
                and "setup_id_short" in row
                and not pd.isna(row["setup_id_short"])
            ):
                score = float(row.get("entry_score_short", 1.0))
                candidate_sides.append(("short", score, int(row["setup_id_short"])))

            if candidate_sides:
                # wybierz stronę z najwyższym entry_score, przy remisie preferuj long
                candidate_sides.sort(
                    key=lambda x: (x[1], x[0] == "long"), reverse=True
                )
                side, score, setup_id = candidate_sides[0]

                if atr_val > 0 and np.isfinite(atr_val):
                    rule = long_rule if side == "long" else short_rule
                    pos = Position(
                        side=side,
                        entry_ts=ts,
                        entry_price=price,
                        atr_entry=atr_val,
                        setup_id=setup_id,
                        setup_entry_score=score,
                        rule=rule,
                    )

            continue  # brak pozycji → kolejny bar

        # --- jeśli jest pozycja: update R(t), MFE, MAE ---
        pos.bars_in_trade += 1
        R_now = compute_R(pos.side, price, pos.entry_price, pos.atr_entry, risk_mult)
        pos.R = R_now
        pos.MFE_R = max(pos.MFE_R, R_now)
        pos.MAE_R = min(pos.MAE_R, R_now)

        # RSI
        rsi_val = row[rsi_col] if rsi_col in df.columns else np.nan

        # SR distance (min z support/resistance w ATR)
        sr_dist = np.nan
        d_s = row.get("m5_nearest_support_dist_atr", np.nan)
        d_r = row.get("m5_nearest_resistance_dist_atr", np.nan)
        if np.isfinite(d_s) or np.isfinite(d_r):
            sr_dist = np.nanmin([d_s, d_r])

        # signature gate flag
        sig_ok = True
        if pos.side == "long" and sig_flag_long is not None:
            sig_ok = bool(row.get(sig_flag_long, False))
        if pos.side == "short" and sig_flag_short is not None:
            sig_ok = bool(row.get(sig_flag_short, False))

        exit_now = False
        exit_reason = None

        # 1) STOP
        if R_now <= pos.rule.stop_R:
            exit_now = True
            exit_reason = "stop"

        # 2) TIMEOUT
        elif pos.bars_in_trade >= pos.rule.max_bars:
            exit_now = True
            exit_reason = "timeout"

        # 3) MIN TARGET strefa realizacji
        elif R_now >= pos.rule.min_target_R:
            # 3a) TRAIL
            if (
                pos.rule.trail_giveback_R > 0
                and R_now <= pos.MFE_R - pos.rule.trail_giveback_R
            ):
                exit_now = True
                exit_reason = "trail"
            else:
                # 3b) RSI/SR exit (tylko jeśli signature gate pozwala)
                rsi_cond = False
                if np.isfinite(rsi_val):
                    if pos.side == "long":
                        rsi_cond = (
                            rsi_val >= pos.rule.rsi_exit_threshold
                            and R_now >= pos.rule.rsi_min_R
                        )
                    else:  # short
                        rsi_cond = (
                            rsi_val <= (100 - pos.rule.rsi_exit_threshold)
                            and R_now >= pos.rule.rsi_min_R
                        )

                sr_cond = True
                if pos.rule.sr_dist_threshold_atr >= 0 and np.isfinite(sr_dist):
                    sr_cond = sr_dist <= pos.rule.sr_dist_threshold_atr

                sig_cond = sig_ok  # jeśli gating włączony, musi być True

                if rsi_cond and sr_cond and sig_cond:
                    exit_now = True
                    exit_reason = "rsi_sr"

        # --- finalizacja trade'a ---
        if exit_now:
            trades.append(
                {
                    "instrument": row.get("instrument", "UNKNOWN"),
                    "side": pos.side,
                    "entry_ts": pos.entry_ts,
                    "exit_ts": ts,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "R": pos.R,
                    "MFE_R": pos.MFE_R,
                    "MAE_R": pos.MAE_R,
                    "bars_in_trade": pos.bars_in_trade,
                    "exit_reason": exit_reason,
                    "setup_id": pos.setup_id,
                    "setup_entry_score": pos.setup_entry_score,
                    "min_target_R": pos.rule.min_target_R,
                    "trail_giveback_R": pos.rule.trail_giveback_R,
                    "stop_R": pos.rule.stop_R,
                    "max_bars": pos.rule.max_bars,
                    "rsi_exit_threshold": pos.rule.rsi_exit_threshold,
                    "rsi_min_R": pos.rule.rsi_min_R,
                    "sr_dist_threshold_atr": pos.rule.sr_dist_threshold_atr,
                }
            )
            pos = None

    # --- podsumowanie ---
    if trades:
        trades_df = pd.DataFrame(trades)
        n_trades = len(trades_df)
        avg_R = trades_df["R"].mean()
        median_R = trades_df["R"].median()
        sum_R = trades_df["R"].sum()
        winrate = (trades_df["R"] > 0).mean()

        equity = trades_df["R"].cumsum()
        running_max = equity.cummax()
        dd = equity - running_max
        max_dd = dd.min() if len(dd) > 0 else 0.0

        summary = pd.DataFrame(
            [
                {
                    "n_trades": n_trades,
                    "avg_R": avg_R,
                    "median_R": median_R,
                    "sum_R": sum_R,
                    "winrate": winrate,
                    "max_DD_R": max_dd,
                }
            ]
        )
    else:
        trades_df = pd.DataFrame(
            columns=[
                "instrument",
                "side",
                "entry_ts",
                "exit_ts",
                "entry_price",
                "exit_price",
                "R",
                "MFE_R",
                "MAE_R",
                "bars_in_trade",
                "exit_reason",
                "setup_id",
                "setup_entry_score",
                "min_target_R",
                "trail_giveback_R",
                "stop_R",
                "max_bars",
                "rsi_exit_threshold",
                "rsi_min_R",
                "sr_dist_threshold_atr",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "n_trades": 0,
                    "avg_R": 0.0,
                    "median_R": 0.0,
                    "sum_R": 0.0,
                    "winrate": 0.0,
                    "max_DD_R": 0.0,
                }
            ]
        )

    return trades_df, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bar-by-bar strategy backtester na podstawie entry setups + ranked exits "
            "+ (opcjonalnie) exit signatures."
        )
    )
    p.add_argument(
        "--snapshots-enriched",
        type=str,
        required=True,
        help="Plik CSV z snapshotami TEST (wzbogacony grouperem).",
    )
    p.add_argument(
        "--entry-setups-long",
        type=str,
        required=False,
        help="Plik CSV z dobrymi setupami LONG (entry_setups_long_good.csv).",
    )
    p.add_argument(
        "--entry-setups-short",
        type=str,
        required=False,
        help="Plik CSV z dobrymi setupami SHORT (entry_setups_short_good.csv).",
    )
    p.add_argument(
        "--ranked-long",
        type=str,
        required=False,
        help="Plik CSV z posortowanymi wynikami exit_lab_v3_long_ranked.csv.",
    )
    p.add_argument(
        "--ranked-short",
        type=str,
        required=False,
        help="Plik CSV z posortowanymi wynikami exit_lab_v3_short_ranked.csv.",
    )
    p.add_argument(
        "--long-rule-index",
        type=int,
        default=0,
        help="Który wiersz z ranked_long użyć jako regułę exitu (default: 0 = top-1).",
    )
    p.add_argument(
        "--short-rule-index",
        type=int,
        default=0,
        help="Który wiersz z ranked_short użyć jako regułę exitu (default: 0 = top-1).",
    )
    p.add_argument(
        "--signatures-long",
        type=str,
        required=False,
        help="Plik CSV z exit_signatures_long.csv (opcjonalny, do signature gates).",
    )
    p.add_argument(
        "--signatures-short",
        type=str,
        required=False,
        help="Plik CSV z exit_signatures_short.csv (opcjonalny, do signature gates).",
    )
    p.add_argument(
        "--atr-col",
        type=str,
        default="m5_atr14",
        help="Nazwa kolumny ATR na M5 (default: m5_atr14).",
    )
    p.add_argument(
        "--risk-mult",
        type=float,
        default=1.0,
        help="Mnożnik ATR dla 1R (default: 1.0).",
    )
    p.add_argument(
        "--rsi-col",
        type=str,
        default="m5_rsi14",
        help="Nazwa kolumny RSI na M5 (default: m5_rsi14).",
    )
    p.add_argument(
        "--out-trades",
        type=str,
        default="trades.csv",
        help="Wyjściowy plik CSV z listą tradów.",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        default="trades_summary.csv",
        help="Wyjściowy plik CSV z podsumowaniem backtestu.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- wczytaj snapshoty TEST ---
    df_snap = load_csv(args.snapshots_enriched)
    if df_snap is None:
        raise SystemExit("[strategy_backtest] Brak snapshots_enriched – przerywam.")
    df_snap = ensure_datetime(df_snap, "m5_ts")

    # --- setupy wejściowe ---
    setups_long = load_csv(args.entry_setups_long) if args.entry_setups_long else None
    setups_short = load_csv(args.entry_setups_short) if args.entry_setups_short else None

    df_snap = build_setup_merge(df_snap, setups_long, "long")
    df_snap = build_setup_merge(df_snap, setups_short, "short")

    # --- exit signatures (opcjonalnie) ---
    sig_long_df = load_csv(args.signatures_long) if args.signatures_long else None
    sig_short_df = load_csv(args.signatures_short) if args.signatures_short else None

    df_snap, sig_flag_long = build_signature_flag(df_snap, sig_long_df, "long")
    df_snap, sig_flag_short = build_signature_flag(df_snap, sig_short_df, "short")

    # --- ranked exits i wybór reguł ---
    ranked_long = load_csv(args.ranked_long) if args.ranked_long else None
    ranked_short = load_csv(args.ranked_short) if args.ranked_short else None

    long_rule = pick_exit_rule(ranked_long, args.long_rule_index, "long") if ranked_long is not None else None
    short_rule = pick_exit_rule(ranked_short, args.short_rule_index, "short") if ranked_short is not None else None

    # --- backtest ---
    trades_df, summary_df = backtest(
        df_snap,
        atr_col=args.atr_col,
        risk_mult=args.risk_mult,
        rsi_col=args.rsi_col,
        long_rule=long_rule,
        short_rule=short_rule,
        sig_flag_long=sig_flag_long,
        sig_flag_short=sig_flag_short,
    )

    # --- zapis ---
    trades_df.to_csv(args.out_trades, sep=";", index=False)
    summary_df.to_csv(args.out_summary, sep=";", index=False)
    print(f"[strategy_backtest] Zapisano trades -> {args.out_trades}")
    print(f"[strategy_backtest] Zapisano summary -> {args.out_summary}")
    print("[strategy_backtest] GOTOWE.")


if __name__ == "__main__":
    main()
