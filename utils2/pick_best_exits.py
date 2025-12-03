import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_results(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Plik {p} nie istnieje – pomijam.")
        return None
    df = pd.read_csv(p, sep=";")
    if df.empty:
        print(f"[WARN] Plik {p} jest pusty.")
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    """
    Prosta normalizacja [0,1] z obsługą stałej kolumny / NaN.
    """
    s = s.astype(float)
    s_min = s.min()
    s_max = s.max()
    if np.isfinite(s_min) and np.isfinite(s_max) and s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    else:
        # wszystko takie samo albo NaN -> zwracamy 0.5 (neutralnie)
        return pd.Series(np.full(len(s), 0.5), index=s.index)


def rank_exits(
    df: pd.DataFrame,
    side: str,
    min_n_trades: int,
    max_dd_abs: float,
    min_avg_R: float,
    min_winrate: float,
    min_sum_R: float,
    w_avg_R: float,
    w_winrate: float,
    w_sum_R: float,
    w_drawdown: float,
) -> pd.DataFrame:
    """
    Filtrowanie + ranking wyników exitu dla jednej strony (long/short).
    """
    df = df.copy()
    df["side"] = side

    required_cols = [
        "n_trades",
        "avg_R",
        "sum_R",
        "winrate",
        "max_drawdown_R",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Brak kolumny '{c}' w wynikach exit_lab_v3 ({side}).")

    # podstawowe filtry jakości
    mask = (
        df["n_trades"].fillna(0) >= min_n_trades
    ) & (
        df["avg_R"].fillna(-1e9) >= min_avg_R
    ) & (
        df["winrate"].fillna(0) >= min_winrate
    ) & (
        df["sum_R"].fillna(-1e9) >= min_sum_R
    )

    # max drawdown: df['max_drawdown_R'] jest ujemne / zero (np. -15R)
    # limit na |DD| -> np. 30R
    dd_abs = df["max_drawdown_R"].abs()
    mask &= dd_abs <= max_dd_abs

    df = df[mask].reset_index(drop=True)

    if df.empty:
        print(f"[INFO] Po filtrach nie ma żadnych wyników dla {side}.")
        return df

    # Normalizacja metryk do [0,1]
    norm_avg_R = normalize_series(df["avg_R"])
    norm_sum_R = normalize_series(df["sum_R"])
    norm_winrate = normalize_series(df["winrate"])

    # drawdown – im większy (bardziej ujemny), tym gorzej
    # bierzemy abs i normalizujemy: 0 = najmniejszy DD, 1 = największy DD
    norm_dd_abs = normalize_series(df["max_drawdown_R"].abs())

    # Score łączny (im więcej, tym lepiej)
    df["score_composite"] = (
        w_avg_R * norm_avg_R
        + w_winrate * norm_winrate
        + w_sum_R * norm_sum_R
        - w_drawdown * norm_dd_abs
    )

    # Sort: najpierw po score, potem po avg_R i n_trades
    df = df.sort_values(
        by=["score_composite", "avg_R", "n_trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # dodaj ranking
    df["rank"] = df.index + 1

    return df


def print_top(df: pd.DataFrame, side: str, top_k: int = 10):
    if df is None or df.empty:
        print(f"\n==== {side.upper()} – brak wyników po filtrach ====\n")
        return

    print(f"\n==== TOP {min(top_k, len(df))} EXITÓW ({side.upper()}) ====\n")
    cols_to_show = [
        "rank",
        "min_target_R",
        "trail_giveback_R",
        "stop_R",
        "max_bars",
        "rsi_exit_threshold",
        "rsi_min_R",
        "sr_dist_threshold_atr",
        "n_trades",
        "avg_R",
        "median_R",
        "sum_R",
        "winrate",
        "pct_ge_2R",
        "pct_ge_3R",
        "max_drawdown_R",
        "pct_exit_stop",
        "pct_exit_trail",
        "pct_exit_rsi_sr",
        "pct_exit_timeout",
        "score_composite",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    top = df[cols_to_show].head(top_k)
    print(top.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Wybór najbardziej sensownych exitów na podstawie wyników exit_lab_v3 "
            "(long + short). Filtrowanie + ranking wielokryterialny."
        )
    )

    p.add_argument(
        "--long-results",
        type=str,
        default="exit_lab_v3_long.csv",
        help="Plik wyników dla longa (exit_lab_v3_long.csv).",
    )
    p.add_argument(
        "--short-results",
        type=str,
        default="exit_lab_v3_short.csv",
        help="Plik wyników dla shorta (exit_lab_v3_short.csv).",
    )
    p.add_argument(
        "--out-long",
        type=str,
        default="exit_lab_v3_long_ranked.csv",
        help="Wyjściowy plik z posortowanymi wynikami longa.",
    )
    p.add_argument(
        "--out-short",
        type=str,
        default="exit_lab_v3_short_ranked.csv",
        help="Wyjściowy plik z posortowanymi wynikami shorta.",
    )
    p.add_argument(
        "--min-n-trades",
        type=int,
        default=80,
        help="Minimalna liczba tradów dla danego zestawu parametrów exitu.",
    )
    p.add_argument(
        "--max-dd-abs",
        type=float,
        default=30.0,
        help="Maksymalny dopuszczalny drawdown (absolutny) w R, np. 30.0 => |DD| <= 30R.",
    )
    p.add_argument(
        "--min-avg-R",
        type=float,
        default=0.3,
        help="Minimalne średnie R na trade po exit (avg_R).",
    )
    p.add_argument(
        "--min-winrate",
        type=float,
        default=0.45,
        help="Minimalny winrate (0–1).",
    )
    p.add_argument(
        "--min-sum-R",
        type=float,
        default=10.0,
        help="Minimalna łączna suma R, np. 10.0.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Ile TOP wyników wypisać w konsoli dla każdej strony.",
    )

    # wagi do score_composite
    p.add_argument(
        "--w-avg-R",
        type=float,
        default=0.4,
        help="Waga avg_R w score_composite.",
    )
    p.add_argument(
        "--w-winrate",
        type=float,
        default=0.3,
        help="Waga winrate w score_composite.",
    )
    p.add_argument(
        "--w-sum-R",
        type=float,
        default=0.2,
        help="Waga sum_R w score_composite.",
    )
    p.add_argument(
        "--w-drawdown",
        type=float,
        default=0.3,
        help="Waga (kara) za drawdown w score_composite (im większa, tym bardziej karze duże DD).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    long_df_raw = load_results(args.long_results)
    short_df_raw = load_results(args.short_results)

    ranked_long = None
    ranked_short = None

    if long_df_raw is not None and not long_df_raw.empty:
        print("==> Ranking LONG exitów...")
        ranked_long = rank_exits(
            long_df_raw,
            side="long",
            min_n_trades=args.min_n_trades,
            max_dd_abs=args.max_dd_abs,
            min_avg_R=args.min_avg_R,
            min_winrate=args.min_winrate,
            min_sum_R=args.min_sum_R,
            w_avg_R=args.w_avg_R,
            w_winrate=args.w_winrate,
            w_sum_R=args.w_sum_R,
            w_drawdown=args.w_drawdown,
        )
        if not ranked_long.empty:
            Path(args.out_long).parent.mkdir(parents=True, exist_ok=True)
            ranked_long.to_csv(args.out_long, sep=";", index=False)
            print(f"  Zapisano posortowane LONG do {args.out_long}")
            print_top(ranked_long, side="long", top_k=args.top_k)

    if short_df_raw is not None and not short_df_raw.empty:
        print("\n==> Ranking SHORT exitów...")
        ranked_short = rank_exits(
            short_df_raw,
            side="short",
            min_n_trades=args.min_n_trades,
            max_dd_abs=args.max_dd_abs,
            min_avg_R=args.min_avg_R,
            min_winrate=args.min_winrate,
            min_sum_R=args.min_sum_R,
            w_avg_R=args.w_avg_R,
            w_winrate=args.w_winrate,
            w_sum_R=args.w_sum_R,
            w_drawdown=args.w_drawdown,
        )
        if not ranked_short.empty:
            Path(args.out_short).parent.mkdir(parents=True, exist_ok=True)
            ranked_short.to_csv(args.out_short, sep=";", index=False)
            print(f"  Zapisano posortowane SHORT do {args.out_short}")
            print_top(ranked_short, side="short", top_k=args.top_k)

    print("\n==> GOTOWE.")
    if (ranked_long is None or ranked_long.empty) and (ranked_short is None or ranked_short.empty):
        print("Uwaga: po filtrach nie znaleziono żadnych rozsądnych kandydatów – "
              "poluzuj progi (min_n_trades, min_avg_R, max_dd_abs itd.).")


if __name__ == "__main__":
    main()
