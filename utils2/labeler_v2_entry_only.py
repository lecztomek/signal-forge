import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================
# FUNKCJE POMOCNICZE
# ==========================

def load_candles_5m(path: Path) -> pd.DataFrame:
    """
    Wczytuje brent_5m.csv w formacie:
    date;time;open;high;low;close;volume

    Zwraca DataFrame z kolumnami:
    ['ts', 'open', 'high', 'low', 'close', 'volume', 'bar_index']
    posortowany po ts, z indeksami RangeIndex (0..n-1)
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date": str,
            "time": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        },
    )

    # Sklejamy date + time w jeden timestamp
    df["ts"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # Wywalamy śmieciowe wiersze bez daty
    df = df.dropna(subset=["ts"]).copy()

    # Sort + reset indeksu (ważne dla iloc)
    df = df.sort_values("ts").reset_index(drop=True)

    # Dodajemy "bar_index" = pozycja w DF (0..n-1)
    df["bar_index"] = df.index

    return df[["ts", "open", "high", "low", "close", "volume", "bar_index"]]


def load_snapshots(path: Path) -> pd.DataFrame:
    """
    Wczytuje plik snapshotów (separator ';').
    Zakładamy, że jest kolumna 'm5_ts' oraz 'm5_atr14'.
    """
    df = pd.read_csv(path, sep=";")

    # Parsowanie timestampu M5
    df["m5_ts"] = pd.to_datetime(df["m5_ts"], errors="coerce", utc=True)
    df["m5_ts"] = df["m5_ts"].dt.tz_convert(None)  # naive datetime

    # ATR musi być jako float
    if "m5_atr14" in df.columns:
        df["m5_atr14"] = pd.to_numeric(df["m5_atr14"], errors="coerce")
    else:
        raise ValueError("W snapshotach brakuje kolumny 'm5_atr14'")

    # Dodajemy trade_id (stałe ID snapshotu)
    df = df.reset_index(drop=True)
    df.insert(0, "trade_id", df.index.astype(int))

    return df


def attach_entry_bar_index(snapshots: pd.DataFrame, candles_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Do snapshotów doda:
    - entry_ts: timestamp świecy M5 (dopasowany do brent_5m)
    - entry_close: close z tej świecy
    - entry_bar_index: indeks świecy (0..n-1)

    Łączymy po najbliższej wcześniejszej świecy (merge_asof, backward).
    """
    snaps = snapshots.copy()

    # Zachowujemy oryginalną kolejność za pomocą helpera
    snaps["_sort_order"] = np.arange(len(snaps))

    snaps_sorted = snaps.sort_values("m5_ts").reset_index(drop=True)

    candles = candles_5m[["ts", "close", "bar_index"]].copy()
    candles = candles.sort_values("ts")

    merged = pd.merge_asof(
        snaps_sorted,
        candles,
        left_on="m5_ts",
        right_on="ts",
        direction="backward",
    )

    missing = merged["close"].isna().sum()
    if missing > 0:
        print(f"UWAGA: {missing} snapshotów nie ma dopasowanej świecy 5m (close NaN).")

    merged = merged.rename(
        columns={
            "ts": "entry_ts",
            "close": "entry_close",
            "bar_index": "entry_bar_index",
        }
    )

    # Wracamy do oryginalnej kolejności snapshotów
    merged = merged.sort_values("_sort_order").drop(columns=["_sort_order"]).reset_index(drop=True)

    return merged


def compute_trade_stats_for_row(
    row: pd.Series,
    candles_5m: pd.DataFrame,
    max_bars_ahead: int,
    checkpoint_bars,
) -> pd.Series:
    """
    Liczy:
    - max_R_long, min_R_long, bar_of_max_R_long, bar_of_min_R_long
    - max_R_short, min_R_short, bar_of_max_R_short, bar_of_min_R_short
    - checkpointy R_long_Xbars, R_short_Xbars dla X w checkpoint_bars

    R definiujemy jako:
      long:  (price - entry_price) / m5_atr14
      short: (entry_price - price) / m5_atr14
    """

    if checkpoint_bars is None:
        checkpoint_bars = []

    entry_bar_index = row.get("entry_bar_index", np.nan)
    entry_price = row.get("entry_close", np.nan)
    risk_per_unit = row.get("m5_atr14", np.nan)

    # Jeżeli coś jest NaN albo ATR <= 0 -> zwracamy NaN-y
    if (
        pd.isna(entry_bar_index)
        or pd.isna(entry_price)
        or pd.isna(risk_per_unit)
        or risk_per_unit <= 0
    ):
        out = {}
        for prefix in ["long", "short"]:
            out[f"max_R_{prefix}"] = np.nan
            out[f"min_R_{prefix}"] = np.nan
            out[f"bar_of_max_R_{prefix}"] = np.nan
            out[f"bar_of_min_R_{prefix}"] = np.nan
            for cp in checkpoint_bars:
                out[f"R_{prefix}_{cp}bars"] = np.nan
        return pd.Series(out)

    entry_bar_index = int(entry_bar_index)

    # Przyszłe świece
    start = entry_bar_index + 1
    end = entry_bar_index + 1 + max_bars_ahead
    future = candles_5m.iloc[start:end]

    if future.empty:
        out = {}
        for prefix in ["long", "short"]:
            out[f"max_R_{prefix}"] = np.nan
            out[f"min_R_{prefix}"] = np.nan
            out[f"bar_of_max_R_{prefix}"] = np.nan
            out[f"bar_of_min_R_{prefix}"] = np.nan
            for cp in checkpoint_bars:
                out[f"R_{prefix}_{cp}bars"] = np.nan
        return pd.Series(out)

    prices = future["close"].to_numpy()
    R_long = (prices - entry_price) / risk_per_unit
    R_short = (entry_price - prices) / risk_per_unit

    def extrema_stats(R_array, side_prefix: str):
        if R_array.size == 0:
            return {
                f"max_R_{side_prefix}": np.nan,
                f"min_R_{side_prefix}": np.nan,
                f"bar_of_max_R_{side_prefix}": np.nan,
                f"bar_of_min_R_{side_prefix}": np.nan,
            }

        max_R = float(np.max(R_array))
        min_R = float(np.min(R_array))

        bar_of_max = int(np.argmax(R_array) + 1)  # offset liczymy od 1
        bar_of_min = int(np.argmin(R_array) + 1)

        return {
            f"max_R_{side_prefix}": max_R,
            f"min_R_{side_prefix}": min_R,
            f"bar_of_max_R_{side_prefix}": bar_of_max,
            f"bar_of_min_R_{side_prefix}": bar_of_min,
        }

    out = {}
    out.update(extrema_stats(R_long, "long"))
    out.update(extrema_stats(R_short, "short"))

    def checkpoint_stats(R_array, side_prefix: str):
        d = {}
        for cp in checkpoint_bars:
            if cp <= R_array.size:
                d[f"R_{side_prefix}_{cp}bars"] = float(R_array[cp - 1])
            else:
                d[f"R_{side_prefix}_{cp}bars"] = np.nan
        return d

    out.update(checkpoint_stats(R_long, "long"))
    out.update(checkpoint_stats(R_short, "short"))

    return pd.Series(out)


def generate_paths_dataset(
    snapshots_with_entry: pd.DataFrame,
    candles_5m: pd.DataFrame,
    max_bars_ahead: int,
) -> pd.DataFrame:
    """
    Generuje per-bar path dataset:

    Kolumny (przykład):
      - trade_id
      - entry_ts
      - entry_price
      - entry_bar_index
      - risk_per_unit
      - bar_offset (1..max_bars_ahead)
      - bar_index  (indeks świecy w głównym DF)
      - bar_ts
      - price
      - R_long, R_short
      - MFE_R_long, MAE_R_long
      - MFE_R_short, MAE_R_short
    """
    records = []

    for _, row in snapshots_with_entry.iterrows():
        trade_id = row.get("trade_id")
        entry_bar_index = row.get("entry_bar_index", np.nan)
        entry_price = row.get("entry_close", np.nan)
        risk_per_unit = row.get("m5_atr14", np.nan)
        entry_ts = row.get("entry_ts", pd.NaT)

        if (
            pd.isna(entry_bar_index)
            or pd.isna(entry_price)
            or pd.isna(risk_per_unit)
            or risk_per_unit <= 0
        ):
            # Nie ma co liczyć dla tego trade'u
            continue

        entry_bar_index = int(entry_bar_index)

        start = entry_bar_index + 1
        end = entry_bar_index + 1 + max_bars_ahead
        future = candles_5m.iloc[start:end]

        if future.empty:
            continue

        prices = future["close"].to_numpy()
        bar_indices = future["bar_index"].to_numpy()
        bar_ts = future["ts"].to_numpy()

        R_long = (prices - entry_price) / risk_per_unit
        R_short = (entry_price - prices) / risk_per_unit

        # Skumulowane MFE/MAE po drodze
        MFE_long = np.maximum.accumulate(R_long)
        MAE_long = np.minimum.accumulate(R_long)
        MFE_short = np.maximum.accumulate(R_short)
        MAE_short = np.minimum.accumulate(R_short)

        for i in range(len(future)):
            records.append(
                {
                    "trade_id": int(trade_id),
                    "entry_ts": entry_ts,
                    "entry_price": float(entry_price),
                    "entry_bar_index": int(entry_bar_index),
                    "risk_per_unit": float(risk_per_unit),
                    "bar_offset": int(i + 1),
                    "bar_index": int(bar_indices[i]),
                    "bar_ts": bar_ts[i],
                    "price": float(prices[i]),
                    "R_long": float(R_long[i]),
                    "R_short": float(R_short[i]),
                    "MFE_R_long": float(MFE_long[i]),
                    "MAE_R_long": float(MAE_long[i]),
                    "MFE_R_short": float(MFE_short[i]),
                    "MAE_R_short": float(MAE_short[i]),
                }
            )

    paths_df = pd.DataFrame.from_records(records)
    return paths_df


# ==========================
# MAIN / ARGPARSE
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Labeler v2: entry-only stats + per-bar paths dataset."
    )

    parser.add_argument(
        "--candles",
        type=str,
        default="brent_5m.csv",
        help="Ścieżka do pliku ze świecami 5m (default: brent_5m.csv)",
    )

    parser.add_argument(
        "--snapshots",
        type=str,
        default="brent_snapshots.csv",
        help="Ścieżka do pliku snapshotów (default: brent_snapshots.csv)",
    )

    parser.add_argument(
        "--out-snapshots",
        type=str,
        default="brent_snapshots_labeled_v2.csv",
        help="Ścieżka wyjściowa dla snapshotów z dopisanymi statystykami (default: brent_snapshots_labeled_v2.csv)",
    )

    parser.add_argument(
        "--out-paths",
        type=str,
        default="brent_paths_v2.csv",
        help="Ścieżka wyjściowa dla per-bar path datasetu (default: brent_paths_v2.csv)",
    )

    parser.add_argument(
        "--max-bars-ahead",
        type=int,
        default=288,
        help="Ile świec 5m w przyszłość analizować (default: 288 ~ 24h).",
    )

    parser.add_argument(
        "--checkpoint-bars",
        type=str,
        default="10,50,100",
        help="Lista checkpointów w świecach, np. '10,50,100'. Puste '' = brak checkpointów.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    candles_path = Path(args.candles)
    snapshots_path = Path(args.snapshots)
    out_snapshots_path = Path(args.out_snapshots)
    out_paths_path = Path(args.out_paths)

    max_bars_ahead = args.max_bars_ahead

    # Parsowanie checkpointów
    if args.checkpoint_bars.strip() == "":
        checkpoint_bars = []
    else:
        checkpoint_bars = [
            int(x) for x in args.checkpoint_bars.split(",") if x.strip() != ""
        ]

    print("==> KONFIG:")
    print(f"  Świece 5m:              {candles_path}")
    print(f"  Snapshoty:              {snapshots_path}")
    print(f"  Wyjście snapshoty:      {out_snapshots_path}")
    print(f"  Wyjście ścieżki per-bar:{out_paths_path}")
    print(f"  max_bars_ahead:         {max_bars_ahead}")
    print(f"  checkpoint_bars:        {checkpoint_bars}")
    print("")

    print("==> Wczytuję świece 5m...")
    candles_5m = load_candles_5m(candles_path)
    print(f"  Załadowano {len(candles_5m)} świec 5m.")

    print("==> Wczytuję snapshoty...")
    snapshots = load_snapshots(snapshots_path)
    print(f"  Załadowano {len(snapshots)} snapshotów.")

    print("==> Dopinanie entry_bar_index / entry_close...")
    snapshots_with_entry = attach_entry_bar_index(snapshots, candles_5m)

    print("==> Liczenie statystyk R dla każdego snapshotu (entry-only)...")
    stats = snapshots_with_entry.apply(
        compute_trade_stats_for_row,
        axis=1,
        candles_5m=candles_5m,
        max_bars_ahead=max_bars_ahead,
        checkpoint_bars=checkpoint_bars,
    )

    print("==> Sklejanie snapshotów z nowymi kolumnami...")
    labeled_snapshots = pd.concat([snapshots_with_entry, stats], axis=1)

    print(f"==> Zapis snapshotów do {out_snapshots_path} ...")
    labeled_snapshots.to_csv(out_snapshots_path, sep=";", index=False)
    print("  OK.")

    print("==> Generowanie per-bar path datasetu...")
    paths_df = generate_paths_dataset(
        snapshots_with_entry=snapshots_with_entry,
        candles_5m=candles_5m,
        max_bars_ahead=max_bars_ahead,
    )
    print(f"  Powstało {len(paths_df)} wierszy ścieżek.")

    print(f"==> Zapis ścieżek do {out_paths_path} ...")
    paths_df.to_csv(out_paths_path, sep=";", index=False)
    print("  OK.")

    print("==> GOTOWE.")


if __name__ == "__main__":
    main()
