import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_entry_setups(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Plik {p} nie istnieje.")
    df = pd.read_csv(p, sep=";")
    print(f"==> Wczytano entry_setups: {len(df)} wierszy z {p}")
    return df


def build_context_key(df: pd.DataFrame) -> pd.Series:
    """
    Buduje „kontekst wejścia” z kilku kluczowych kolumn opisujących stan rynku.
    Setupy z takim samym context_key traktujemy jako „podobne”.

    Jeśli jakiejś kolumny nie ma, po prostu ją ignorujemy.
    """
    context_cols_all = [
        "instrument",
        "daily_trend",
        "daily_vol_bucket",
        "daily_rsi_state",
        "h1_trend",
        "h1_rsi_state",
        "m5_rsi_state",
        "m5_pos_in_recent_range",
        "m5_vol_bucket",
        "near_support",
        "near_resistance",
        "session_bucket",
    ]
    cols = [c for c in context_cols_all if c in df.columns]
    if not cols:
        return pd.Series(["_NO_CONTEXT_"] * len(df), index=df.index)

    # sklejka stringów kolumna1|kolumna2|...
    return df[cols].astype(str).agg("|".join, axis=1)


def filter_side(
    df: pd.DataFrame,
    side: str,
    min_setup_trades: int,
    min_setup_clusters: int,
    min_entry_score: float,
    min_avg_max_R: float,
    min_pct_ge_2R: float,
    max_pct_min_le_minus1R: float | None,
    top_k: int | None,
    max_per_context: int | None = None,
) -> pd.DataFrame:
    """
    Filtrowanie setupów dla danej strony (long/short) na podstawie statystyk z entry_setups,
    z opcjonalnym ograniczeniem ilu setupów możemy mieć na 1 „kontekst” rynku.
    """
    prefix = "long" if side == "long" else "short"

    score_col = f"{prefix}_entry_score"
    avg_max_R_col = f"{prefix}_avg_max_R_weighted"
    pct_ge_2R_col = f"{prefix}_pct_ge_2R_weighted"
    pct_min_le_minus1R_col = f"{prefix}_pct_min_le_-1R_weighted"

    required = [
        "n_trades",
        "n_clusters",
        score_col,
        avg_max_R_col,
        pct_ge_2R_col,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Brak kolumny '{c}' w entry_setups (potrzebne dla {side}).")

    mask = (
        (df["n_trades"].fillna(0) >= min_setup_trades)
        & (df["n_clusters"].fillna(0) >= min_setup_clusters)
        & (df[score_col].fillna(-1e9) >= min_entry_score)
        & (df[avg_max_R_col].fillna(-1e9) >= min_avg_max_R)
        & (df[pct_ge_2R_col].fillna(0) >= min_pct_ge_2R)
    )

    if max_pct_min_le_minus1R is not None:
        if pct_min_le_minus1R_col not in df.columns:
            raise ValueError(f"Brak kolumny '{pct_min_le_minus1R_col}' w entry_setups.")
        mask &= df[pct_min_le_minus1R_col].fillna(1.0) <= max_pct_min_le_minus1R

    filtered = df[mask].copy()
    print(f"==> {side.upper()}: po filtrach zostało {len(filtered)} setupów.")

    if filtered.empty:
        return filtered

    # prosty ranking: entry_score + avg_max_R + pct_ge_2R
    # (normalizacja, żeby się nie „gryzły” skale)
    def norm(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        s_min, s_max = s.min(), s.max()
        if np.isfinite(s_min) and np.isfinite(s_max) and s_max > s_min:
            return (s - s_min) / (s_max - s_min)
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    score_norm = norm(filtered[score_col])
    avgR_norm = norm(filtered[avg_max_R_col])
    pct2R_norm = norm(filtered[pct_ge_2R_col])

    filtered["entry_quality_score"] = (
        0.5 * score_norm + 0.3 * avgR_norm + 0.2 * pct2R_norm
    )

    # sortujemy od najlepszych
    filtered = filtered.sort_values(
        by=["entry_quality_score", score_col, avg_max_R_col, "n_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # --- OGRANICZENIE NA KONTEKST ---
    if max_per_context is not None and max_per_context > 0 and "context_key" in filtered.columns:
        kept_idx = []
        ctx_counts: dict[str, int] = {}
        for idx, row in filtered.iterrows():
            key = row["context_key"]
            cnt = ctx_counts.get(key, 0)
            if cnt < max_per_context:
                kept_idx.append(idx)
                ctx_counts[key] = cnt + 1
        filtered = filtered.loc[kept_idx].reset_index(drop=True)
        print(
            f"==> {side.UPPER() if hasattr(str, 'UPPER') else side.upper()}: "
            f"po ograniczeniu do max {max_per_context} setupów na kontekst zostało {len(filtered)} setupów."
        )

    # ograniczenie do TOP K po rankingu
    if top_k is not None and top_k > 0 and len(filtered) > top_k:
        filtered = filtered.head(top_k).copy()
        print(f"==> {side.upper()}: ograniczam do TOP {top_k} setupów.")

    # dodaj info, że ten setup jest wybrany dla tej strony
    filtered[f"use_for_{side}"] = True
    return filtered


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Filtrowanie 'dobrych' setupów wejściowych przed exit-lab v3 "
            "na podstawie metryk z entry_setups."
        )
    )
    p.add_argument(
        "--entry-setups",
        type=str,
        default="my_entry_setups.csv",
        help="Wejściowy plik entry_setups (z groupera).",
    )
    p.add_argument(
        "--out-long",
        type=str,
        default="entry_setups_long_good.csv",
        help="Wyjściowy plik z dobrymi setupami dla longa.",
    )
    p.add_argument(
        "--out-short",
        type=str,
        default="entry_setups_short_good.csv",
        help="Wyjściowy plik z dobrymi setupami dla shorta.",
    )

    # wspólne progi
    p.add_argument(
        "--min-setup-trades",
        type=int,
        default=20,
        help="Minimalna liczba tradów w setupie (n_trades).",
    )
    p.add_argument(
        "--min-setup-clusters",
        type=int,
        default=2,
        help="Minimalna liczba klastrów (n_clusters).",
    )

    # LONG progi
    p.add_argument(
        "--long-min-entry-score",
        type=float,
        default=0.5,
        help="Minimalny long_entry_score, żeby setup był brany pod uwagę.",
    )
    p.add_argument(
        "--long-min-avg-max-R",
        type=float,
        default=2.0,
        help="Minimalne long_avg_max_R_weighted (średnia MFE w R).",
    )
    p.add_argument(
        "--long-min-pct-ge-2R",
        type=float,
        default=0.4,
        help="Minimalny odsetek long_pct_ge_2R_weighted (0–1).",
    )
    p.add_argument(
        "--long-max-pct-min-le-1R",
        type=float,
        default=0.98,
        help="Maksymalny dopuszczalny long_pct_min_le_-1R_weighted (0–1). "
             "Ustaw None, żeby nie filtrować po tym.",
    )
    p.add_argument(
        "--top-long",
        type=int,
        default=15,
        help="Maksymalna liczba setupów longa po filtrach (None = bez limitu).",
    )

    # SHORT progi
    p.add_argument(
        "--short-min-entry-score",
        type=float,
        default=0.5,
        help="Minimalny short_entry_score.",
    )
    p.add_argument(
        "--short-min-avg-max-R",
        type=float,
        default=2.0,
        help="Minimalne short_avg_max_R_weighted.",
    )
    p.add_argument(
        "--short-min-pct-ge-2R",
        type=float,
        default=0.4,
        help="Minimalny short_pct_ge_2R_weighted.",
    )
    p.add_argument(
        "--short-max-pct-min-le-1R",
        type=float,
        default=0.98,
        help="Maksymalny short_pct_min_le_-1R_weighted.",
    )
    p.add_argument(
        "--top-short",
        type=int,
        default=15,
        help="Maksymalna liczba setupów shorta po filtrach.",
    )

    # LIMITY NA KONTEKST – tu pilnujemy, żeby entry nie były prawie takie same
    p.add_argument(
        "--max-long-per-context",
        type=int,
        default=1,
        help="Maksymalna liczba LONG setupów na jeden kontekst rynku (default: 1). "
             "Ustaw 0 lub None, żeby wyłączyć.",
    )
    p.add_argument(
        "--max-short-per-context",
        type=int,
        default=1,
        help="Maksymalna liczba SHORT setupów na jeden kontekst rynku (default: 1). "
             "Ustaw 0 lub None, żeby wyłączyć.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    df = load_entry_setups(args.entry_setups)

    # zbuduj klucz kontekstu (wspólny dla longa i shorta)
    df["context_key"] = build_context_key(df)

    # LONG
    print("\n==> Filtrowanie LONG setupów...")
    long_filtered = filter_side(
        df=df,
        side="long",
        min_setup_trades=args.min_setup_trades,
        min_setup_clusters=args.min_setup_clusters,
        min_entry_score=args.long_min_entry_score,
        min_avg_max_R=args.long_min_avg_max_R,
        min_pct_ge_2R=args.long_min_pct_ge_2R,
        max_pct_min_le_minus1R=args.long_max_pct_min_le_1R,
        top_k=args.top_long if args.top_long > 0 else None,
        max_per_context=args.max_long_per_context if args.max_long_per_context and args.max_long_per_context > 0 else None,
    )
    if not long_filtered.empty:
        Path(args.out_long).parent.mkdir(parents=True, exist_ok=True)
        long_filtered.to_csv(args.out_long, sep=";", index=False)
        print(f"==> Zapisano LONG setupy do {args.out_long}")
        cols_show = [
            "instrument",
            "daily_trend",
            "h1_trend",
            "m5_rsi_state",
            "n_trades",
            "n_clusters",
            "long_entry_score",
            "long_avg_max_R_weighted",
            "long_pct_ge_2R_weighted",
            "entry_quality_score",
        ]
        cols_show = [c for c in cols_show if c in long_filtered.columns]
        print(long_filtered[cols_show].to_string(index=False))

    # SHORT
    print("\n==> Filtrowanie SHORT setupów...")
    short_filtered = filter_side(
        df=df,
        side="short",
        min_setup_trades=args.min_setup_trades,
        min_setup_clusters=args.min_setup_clusters,
        min_entry_score=args.short_min_entry_score,
        min_avg_max_R=args.short_min_avg_max_R,
        min_pct_ge_2R=args.short_min_pct_ge_2R,
        max_pct_min_le_minus1R=args.short_max_pct_min_le_1R,
        top_k=args.top_short if args.top_short > 0 else None,
        max_per_context=args.max_short_per_context if args.max_short_per_context and args.max_short_per_context > 0 else None,
    )
    if not short_filtered.empty:
        Path(args.out_short).parent.mkdir(parents=True, exist_ok=True)
        short_filtered.to_csv(args.out_short, sep=";", index=False)
        print(f"==> Zapisano SHORT setupy do {args.out_short}")
        cols_show = [
            "instrument",
            "daily_trend",
            "h1_trend",
            "m5_rsi_state",
            "n_trades",
            "n_clusters",
            "short_entry_score",
            "short_avg_max_R_weighted",
            "short_pct_ge_2R_weighted",
            "entry_quality_score",
        ]
        cols_show = [c for c in cols_show if c in short_filtered.columns]
        print(short_filtered[cols_show].to_string(index=False))

    print("\n==> GOTOWE.")
    if long_filtered.empty:
        print("Uwaga: brak dobrych setupów LONG przy tych progach – poluzuj parametry long-*.")
    if short_filtered.empty:
        print("Uwaga: brak dobrych setupów SHORT przy tych progach – poluzuj parametry short-*.")


if __name__ == "__main__":
    main()
