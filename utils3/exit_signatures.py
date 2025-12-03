import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Exit signatures: analiza idealnych wyjść na bazie paths_v3."
    )
    p.add_argument(
        "--paths-v3",
        type=str,
        default="brent_paths_v3.csv",
        help="Ścieżka do pliku paths_v3 (per-bar ścieżki z feature'ami).",
    )
    p.add_argument(
        "--side",
        type=str,
        choices=["long", "short"],
        default="long",
        help="Którą stronę analizujemy (long/short).",
    )
    p.add_argument(
        "--max-bars",
        type=int,
        default=288,
        help="Maksymalna liczba barów w tradu, którą bierzemy pod uwagę.",
    )
    p.add_argument(
        "--min-ideal-R",
        type=float,
        default=1.0,
        help="Minimalny max_R, żeby trade był brany do analizy idealnego exitu.",
    )
    p.add_argument(
        "--min-exits-per-group",
        type=int,
        default=20,
        help="Minimalna liczba exitów w grupie, żeby ją zostawić w exit signatures.",
    )
    p.add_argument(
        "--out-ideal-exits",
        type=str,
        default="ideal_exits_long_v1.csv",
        help="Ścieżka wyjściowa dla tabeli idealnych exitów (per trade).",
    )
    p.add_argument(
        "--out-signatures",
        type=str,
        default="exit_signatures_long_v1.csv",
        help="Ścieżka wyjściowa dla zgrupowanych sygnatur exitów.",
    )
    return p.parse_args()


# --------- pomocnicze buckety / kategorie ----------

def bucket_R(R: float) -> str:
    if np.isnan(R):
        return "NA"
    if R < 1.0:
        return "<1R"
    if R < 2.0:
        return "1-2R"
    if R < 3.0:
        return "2-3R"
    if R < 5.0:
        return "3-5R"
    return "5+R"


def bucket_bar_phase(bar_offset: int, max_bars: int) -> str:
    if max_bars <= 0 or pd.isna(bar_offset):
        return "UNKNOWN"
    ratio = bar_offset / max_bars
    if ratio <= 1/3:
        return "EARLY"
    if ratio <= 2/3:
        return "MID"
    return "LATE"


def sr_context_for_side(side: str, dist_atr: float, has_sr: bool) -> str:
    if not has_sr or pd.isna(dist_atr):
        return "NO_SR"
    if dist_atr <= 0.5:
        return "NEAR_SR"
    if dist_atr <= 1.5:
        return "MID_SR"
    return "FAR_SR"


def dir_from_delta(delta: float, eps: float = 2.0) -> str:
    if np.isnan(delta):
        return "UNKNOWN"
    if delta > eps:
        return "UP"
    if delta < -eps:
        return "DOWN"
    return "FLAT"


# --------- główna logika: idealne exity ----------

def compute_ideal_exits(paths_v3: pd.DataFrame,
                        side: str,
                        max_bars: int,
                        min_ideal_R: float) -> pd.DataFrame:
    """
    Dla każdego trade_id:
      - ogranicza się do bar_offset <= max_bars
      - znajduje bar z max R (R_long/R_short)
      - jeśli max_R >= min_ideal_R, zapisuje stan + zmiany (delta) na idealnym exicie

    Zwraca DF z jednym wierszem na trade + cechy idealnego exitu.
    """
    df = paths_v3.copy()

    # Wymagane kolumny
    required = {"trade_id", "bar_offset", "bar_ts", "R_long", "R_short"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w paths_v3: {missing}")

    df["bar_ts"] = pd.to_datetime(df["bar_ts"], errors="coerce", utc=True)
    df["bar_ts"] = df["bar_ts"].dt.tz_convert(None)
    df["bar_offset"] = pd.to_numeric(df["bar_offset"], errors="coerce")

    col_R = "R_long" if side == "long" else "R_short"
    col_MFE = "MFE_R_long" if side == "long" else "MFE_R_short"

    results = []

    grouped = df.groupby("trade_id", sort=False)
    total_trades = len(grouped)
    print(f"  Analiza idealnych exitów dla {total_trades} trade_id...")

    for idx, (tid, g) in enumerate(grouped, start=1):
        g = g.sort_values("bar_offset")
        g = g[g["bar_offset"] <= max_bars]

        if g.empty:
            continue

        R = g[col_R].to_numpy()
        offsets = g["bar_offset"].to_numpy()

        if len(R) == 0:
            continue

        # max R i jego pozycja
        max_i = int(np.nanargmax(R))
        max_R_value = float(R[max_i])

        if max_R_value < min_ideal_R:
            # trade nigdy nie osiągnął wystarczającego R
            continue

        ideal_row = g.iloc[max_i]
        # poprzedni bar (jeśli jest)
        if max_i > 0:
            prev_row = g.iloc[max_i - 1]
        else:
            prev_row = None

        entry_ts = g["entry_ts"].iloc[0] if "entry_ts" in g.columns else pd.NaT

        R_exit = max_R_value
        MFE_exit = float(ideal_row[col_MFE]) if col_MFE in g.columns else np.nan
        giveback = MFE_exit - R_exit if not np.isnan(MFE_exit) else np.nan
        bar_offset_exit = int(ideal_row["bar_offset"])
        bar_ts_exit = ideal_row["bar_ts"]

        # RSI
        rsi_curr = ideal_row["m5_rsi14"] if "m5_rsi14" in g.columns else np.nan
        rsi_prev = prev_row["m5_rsi14"] if (prev_row is not None and "m5_rsi14" in g.columns) else np.nan
        delta_rsi = rsi_curr - rsi_prev if (not np.isnan(rsi_curr) and not np.isnan(rsi_prev)) else np.nan
        rsi_dir = dir_from_delta(delta_rsi, eps=2.0)

        # R delta (między barami)
        if prev_row is not None:
            R_prev = prev_row[col_R]
            delta_R = R_exit - float(R_prev) if not np.isnan(R_prev) else np.nan
        else:
            delta_R = np.nan
        R_dir = dir_from_delta(delta_R, eps=0.2)  # mały eps, bo R w R-rach

        # SR dist i kontekst
        if side == "long":
            dist_col = "m5_nearest_resistance_dist_atr"
        else:
            dist_col = "m5_nearest_support_dist_atr"

        if dist_col in g.columns:
            dist_curr = ideal_row[dist_col]
            has_sr = not pd.isna(dist_curr)
            sr_ctx = sr_context_for_side(side, float(dist_curr) if has_sr else np.nan, has_sr)
        else:
            dist_curr = np.nan
            sr_ctx = "NO_INFO"

        # Pozycja w rangu / vol / stany RSI/H1/D1 jeśli są
        pos_range = ideal_row["m5_pos_in_recent_range"] if "m5_pos_in_recent_range" in g.columns else "UNKNOWN"
        vol_bucket = ideal_row["m5_vol_bucket"] if "m5_vol_bucket" in g.columns else "UNKNOWN"
        m5_rsi_state = ideal_row["m5_rsi_state"] if "m5_rsi_state" in g.columns else "UNKNOWN"

        daily_trend = ideal_row["daily_trend"] if "daily_trend" in g.columns else "UNKNOWN"
        daily_rsi_state = ideal_row["daily_rsi_state"] if "daily_rsi_state" in g.columns else "UNKNOWN"
        h1_trend = ideal_row["h1_trend"] if "h1_trend" in g.columns else "UNKNOWN"
        h1_rsi_state = ideal_row["h1_rsi_state"] if "h1_rsi_state" in g.columns else "UNKNOWN"

        session_bucket = ideal_row["session_bucket"] if "session_bucket" in g.columns else "UNKNOWN"

        # buckety
        R_bucket_val = bucket_R(R_exit)
        bar_phase = bucket_bar_phase(bar_offset_exit, max_bars)

        rec = {
            "trade_id": tid,
            "side": side,
            "entry_ts": entry_ts,
            "exit_ts": bar_ts_exit,
            "R_exit": R_exit,
            "MFE_exit": MFE_exit,
            "giveback_from_MFE": giveback,
            "bar_offset_exit": bar_offset_exit,
            "R_bucket": R_bucket_val,
            "bar_phase": bar_phase,
            "m5_rsi14": rsi_curr,
            "m5_rsi_state": m5_rsi_state,
            "delta_rsi": delta_rsi,
            "rsi_dir": rsi_dir,
            "delta_R": delta_R,
            "R_dir": R_dir,
            "sr_dist_atr": float(dist_curr) if not pd.isna(dist_curr) else np.nan,
            "sr_context": sr_ctx,
            "m5_pos_in_recent_range": pos_range,
            "m5_vol_bucket": vol_bucket,
            "daily_trend": daily_trend,
            "daily_rsi_state": daily_rsi_state,
            "h1_trend": h1_trend,
            "h1_rsi_state": h1_rsi_state,
            "session_bucket": session_bucket,
        }

        results.append(rec)

        if idx % 1000 == 0:
            print(f"    Przetworzono {idx} / {total_trades} trade_id...")

    ideal_df = pd.DataFrame(results)
    print(f"  Idealnych exitów zakwalifikowanych: {len(ideal_df)}")
    return ideal_df


# --------- grupowanie sygnatur exitów ----------

def compute_exit_signatures(ideal_exits: pd.DataFrame, min_exits_per_group: int) -> pd.DataFrame:
    """
    Grupuje idealne exity po sygnaturze (kombinacji stanów/zmian) i liczy statystyki.
    """
    if ideal_exits.empty:
        return pd.DataFrame()

    group_cols = [
        "R_bucket",
        "bar_phase",
        "m5_rsi_state",
        "rsi_dir",
        "sr_context",
        "m5_pos_in_recent_range",
        "m5_vol_bucket",
        "daily_trend",
        "daily_rsi_state",
        "h1_trend",
        "h1_rsi_state",
        "session_bucket",
    ]

    def agg(grp: pd.DataFrame) -> pd.Series:
        R = grp["R_exit"].to_numpy()
        n = len(grp)
        avg_R = float(np.mean(R)) if n > 0 else np.nan
        med_R = float(np.median(R)) if n > 0 else np.nan
        avg_giveback = float(np.mean(grp["giveback_from_MFE"])) if "giveback_from_MFE" in grp.columns else np.nan
        avg_bar_offset = float(np.mean(grp["bar_offset_exit"])) if "bar_offset_exit" in grp.columns else np.nan

        return pd.Series(
            {
                "n_exits": n,
                "avg_R_exit": avg_R,
                "median_R_exit": med_R,
                "avg_giveback_from_MFE": avg_giveback,
                "avg_bar_offset_exit": avg_bar_offset,
            }
        )

    grouped = ideal_exits.groupby(group_cols, dropna=False).apply(agg).reset_index()

    before = len(grouped)
    grouped = grouped[grouped["n_exits"] >= min_exits_per_group].reset_index(drop=True)
    after = len(grouped)
    print(f"  Grup exitów przed filtrem: {before}, po filtrze min_exits={min_exits_per_group}: {after}")

    # sort np. po avg_R_exit i n_exits
    grouped = grouped.sort_values(
        by=["avg_R_exit", "n_exits"], ascending=[False, False]
    ).reset_index(drop=True)

    return grouped


def main():
    args = parse_args()

    paths_path = Path(args.paths_v3)
    out_ideal_path = Path(args.out_ideal_exits)
    out_signatures_path = Path(args.out_signatures)

    side = args.side

    print(f"==> Wczytuję paths_v3 z {paths_path} ...")
    paths_v3 = pd.read_csv(paths_path, sep=";")
    print(f"  paths_v3: {len(paths_v3)} wierszy.")

    print(f"==> Liczę idealne exity dla side={side} (max_bars={args.max_bars}, min_ideal_R={args.min_ideal_R})...")
    ideal_exits = compute_ideal_exits(
        paths_v3=paths_v3,
        side=side,
        max_bars=args.max_bars,
        min_ideal_R=args.min_ideal_R,
    )

    print(f"==> Zapisuję idealne exity do {out_ideal_path} ...")
    ideal_exits.to_csv(out_ideal_path, sep=";", index=False)

    if ideal_exits.empty:
        print("Brak idealnych exitów do grupowania. Kończę.")
        return

    print("==> Liczę sygnatury exitów (grupowanie)...")
    signatures = compute_exit_signatures(
        ideal_exits=ideal_exits,
        min_exits_per_group=args.min_exits_per_group,
    )

    print(f"==> Zapisuję sygnatury do {out_signatures_path} ...")
    signatures.to_csv(out_signatures_path, sep=";", index=False)

    print("==> GOTOWE.")


if __name__ == "__main__":
    main()
