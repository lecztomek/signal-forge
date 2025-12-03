import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Generator ścieżek v3 (streaming): per-bar R + feature'y, zapis kawałkami."
    )
    p.add_argument(
        "--snapshots-enriched",
        type=str,
        default="brent_snapshots_enriched_v1.csv",
        help="Wejściowy plik snapshotów wzbogaconych (z groupera).",
    )
    p.add_argument(
        "--out-paths",
        type=str,
        default="brent_paths_v3.csv",
        help="Ścieżka wyjściowa dla ścieżek v3.",
    )
    p.add_argument(
        "--max-bars-ahead",
        type=int,
        default=288,
        help="Maksymalna liczba barów w przyszłość (default: 288).",
    )
    p.add_argument(
        "--trades-per-chunk",
        type=int,
        default=500,
        help="Ile trade'ów przetwarzać zanim zapiszemy kolejny kawałek do CSV (default: 500).",
    )
    return p.parse_args()


# Jakie kolumny kopiujemy z future-barów do ścieżki
FEATURE_COLS_PER_BAR = [
    # D1
    "daily_rsi14",
    "daily_rsi14_pct_rank",
    "daily_rsi_state",
    "daily_trend",
    "daily_atr14",
    "daily_vol_bucket",

    # H1
    "h1_rsi14",
    "h1_rsi14_pct_rank",
    "h1_rsi_state",
    "h1_trend",
    "h1_slope_state",
    "h1_atr14",
    "h1_position_vs_swings",

    # M5
    "m5_close",
    "m5_atr14",
    "m5_atr14_pct_rank",
    "m5_rsi14",
    "m5_rsi14_pct_rank",
    "m5_rsi_state",
    "m5_pos_in_recent_range",
    "m5_vol_bucket",

    # SR
    "m5_nearest_support_dist_atr",
    "m5_nearest_support_strength",
    "m5_nearest_support_timeframe",
    "near_support",
    "m5_nearest_resistance_dist_atr",
    "m5_nearest_resistance_strength",
    "m5_nearest_resistance_timeframe",
    "near_resistance",

    # Session / kontekst
    "session_bucket",
]


def generate_paths_v3_chunks(
    snaps: pd.DataFrame,
    max_bars_ahead: int,
    trades_per_chunk: int,
):
    """
    Generator: zwraca kolejne kawałki (DataFrame) ścieżek v3.
    Każdy chunk to ścieżki dla ~trades_per_chunk wejść (trade_id-ów).
    """
    df = snaps.copy()

    # Minimalny zestaw kolumn
    required = {"trade_id", "instrument", "m5_ts", "m5_idx", "m5_close", "m5_atr14"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn wymaganych do path_generatora: {missing}")

    # Daty → datetime
    df["m5_ts"] = pd.to_datetime(df["m5_ts"], errors="coerce", utc=True)
    df["m5_ts"] = df["m5_ts"].dt.tz_convert(None)

    # m5_idx → numeric
    df["m5_idx"] = pd.to_numeric(df["m5_idx"], errors="coerce")

    # sort po instrumencie + m5_idx, to nam ustawia kolejność barów
    df = df.sort_values(["instrument", "m5_idx"]).reset_index(drop=True)

    # Feature’y, które REALNIE istnieją
    features_exist = [c for c in FEATURE_COLS_PER_BAR if c in df.columns]
    missing_feat = [c for c in FEATURE_COLS_PER_BAR if c not in df.columns]
    if missing_feat:
        print(f"UWAGA: brak części feature'ów w snapshots_enriched: {missing_feat}")
    print(f"  Użyję per-bar feature'ów: {features_exist}")

    # Dodaj pozycję w ramach instrumentu (do future-slicingu po iloc)
    df["row_pos"] = np.arange(len(df))

    total_trades = len(df)
    print(f"  Łącznie snapshotów (kandydatów na wejście): {total_trades}")

    processed_trades = 0
    chunk_records = []

    # Group by instrument, żeby przyszłość była w tej samej serii
    for inst, inst_df in df.groupby("instrument", sort=False):
        inst_df = inst_df.sort_values("m5_idx").reset_index(drop=True)
        inst_df["row_pos"] = np.arange(len(inst_df))

        n_inst_trades = len(inst_df)
        print(f"  Instrument {inst}: {n_inst_trades} snapshotów")

        # użyj itertuples (szybsze niż iterrows)
        for row in inst_df.itertuples(index=False):
            trade_id = int(row.trade_id)
            entry_ts = row.m5_ts
            entry_idx = row.m5_idx
            entry_price = row.m5_close
            entry_risk = row.m5_atr14

            if pd.isna(entry_risk) or entry_risk <= 0 or pd.isna(entry_price):
                processed_trades += 1
                continue

            pos = int(row.row_pos)
            # przyszłe bary dla tego instrumentu
            future = inst_df.iloc[pos + 1 : pos + 1 + max_bars_ahead]
            if future.empty:
                processed_trades += 1
                continue

            prices = future["m5_close"].to_numpy()
            bar_ts = future["m5_ts"].to_numpy()
            bar_idx = future["m5_idx"].to_numpy()

            # R w oparciu o ATR z wejścia
            R_long = (prices - entry_price) / entry_risk
            R_short = (entry_price - prices) / entry_risk

            # MFE/MAE
            MFE_long = np.maximum.accumulate(R_long)
            MAE_long = np.minimum.accumulate(R_long)
            MFE_short = np.maximum.accumulate(R_short)
            MAE_short = np.minimum.accumulate(R_short)

            # Zapisujemy wszystkie bary tego tradu
            for i in range(len(future)):
                frow = future.iloc[i]
                rec = {
                    "trade_id": trade_id,
                    "instrument": inst,
                    "entry_ts": entry_ts,
                    "entry_m5_idx": entry_idx,
                    "entry_price": float(entry_price),
                    "entry_risk_per_unit": float(entry_risk),
                    "bar_offset": int(i + 1),
                    "bar_ts": bar_ts[i],
                    "bar_m5_idx": int(bar_idx[i]),
                    "price": float(prices[i]),
                    "R_long": float(R_long[i]),
                    "R_short": float(R_short[i]),
                    "MFE_R_long": float(MFE_long[i]),
                    "MAE_R_long": float(MAE_long[i]),
                    "MFE_R_short": float(MFE_short[i]),
                    "MAE_R_short": float(MAE_short[i]),
                }

                for col in features_exist:
                    rec[col] = frow[col]

                chunk_records.append(rec)

            processed_trades += 1

            # Prosty progress log
            if processed_trades % 1000 == 0:
                print(f"    Przetworzono {processed_trades} / {total_trades} snapshotów...")

            # Jeżeli przekroczyliśmy limit trade'ów w chunku → zwracamy kawałek
            if processed_trades % trades_per_chunk == 0:
                if chunk_records:
                    chunk_df = pd.DataFrame.from_records(chunk_records)
                    chunk_records = []
                    yield chunk_df

    # wszystko przeszliśmy, zwróć ostatni niedomknięty chunk
    if chunk_records:
        chunk_df = pd.DataFrame.from_records(chunk_records)
        yield chunk_df


def main():
    args = parse_args()

    snaps_path = Path(args.snapshots_enriched)
    out_paths_path = Path(args.out_paths)

    print(f"==> Wczytuję snapshots_enriched z {snaps_path} ...")
    snaps = pd.read_csv(snaps_path, sep=";")
    print(f"  Wierszy: {len(snaps)}")

    # Usuwamy potencjalne śmieci z trade_id
    if "trade_id" not in snaps.columns:
        snaps.insert(0, "trade_id", np.arange(len(snaps), dtype=int))

    print(
        f"==> Generuję ścieżki v3 (max_bars_ahead={args.max_bars_ahead}, "
        f"trades_per_chunk={args.trades_per_chunk})..."
    )

    first_chunk = True
    total_rows = 0

    # Upewnij się, że plik startuje od zera
    if out_paths_path.exists():
        print(f"  UWAGA: plik {out_paths_path} już istnieje, nadpisuję...")
        out_paths_path.unlink()

    for chunk_df in generate_paths_v3_chunks(
        snaps,
        max_bars_ahead=args.max_bars_ahead,
        trades_per_chunk=args.trades_per_chunk,
    ):
        # zapisujemy kawałek
        mode = "w" if first_chunk else "a"
        header = first_chunk

        chunk_df.to_csv(out_paths_path, sep=";", index=False, mode=mode, header=header)
        total_rows += len(chunk_df)

        print(f"  -> zapisano chunk: {len(chunk_df)} wierszy (łącznie: {total_rows})")

        first_chunk = False

    print(f"==> GOTOWE. Łącznie zapisano {total_rows} wierszy do {out_paths_path}")


if __name__ == "__main__":
    main()
