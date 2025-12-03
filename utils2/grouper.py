import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- Feature builders ----------

def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    # daily_trend
    if {'daily_close', 'daily_ema20', 'daily_ema50'}.issubset(df.columns):
        close = df['daily_close']
        ema20 = df['daily_ema20']
        ema50 = df['daily_ema50']
        cond_up = (close > ema20) & (ema20 > ema50)
        cond_down = (close < ema20) & (ema20 < ema50)
        df['daily_trend'] = np.select(
            [cond_up, cond_down],
            ['UP', 'DOWN'],
            default='RANGE'
        )
    else:
        df['daily_trend'] = 'UNKNOWN'

    # daily_vol_bucket
    if 'daily_atr14_pct_rank' in df.columns:
        atr_pct = df['daily_atr14_pct_rank']
        cond_low = atr_pct < 33
        cond_high = atr_pct > 66
        df['daily_vol_bucket'] = np.select(
            [cond_low, cond_high],
            ['VOL_LOW', 'VOL_HIGH'],
            default='VOL_MID'
        )
    else:
        df['daily_vol_bucket'] = 'UNKNOWN'

    # daily_rsi_state
    if 'daily_rsi14_pct_rank' in df.columns:
        r = df['daily_rsi14_pct_rank']
        cond_os = r <= 20
        cond_ob = r >= 80
        df['daily_rsi_state'] = np.select(
            [cond_os, cond_ob],
            ['OS', 'OB'],
            default='NEUTRAL'
        )
    else:
        df['daily_rsi_state'] = 'UNKNOWN'

    return df


def add_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    # h1_trend
    if {'h1_close', 'h1_ema'}.issubset(df.columns):
        close = df['h1_close']
        ema = df['h1_ema']
        cond_up = close > ema
        cond_down = close < ema
        df['h1_trend'] = np.select(
            [cond_up, cond_down],
            ['UP', 'DOWN'],
            default='RANGE'
        )
    else:
        df['h1_trend'] = 'UNKNOWN'

    # h1_slope_state na podstawie kwantyli h1_ema_slope_pct / h1_ema_slope
    slope_col = None
    if 'h1_ema_slope_pct' in df.columns:
        slope_col = 'h1_ema_slope_pct'
    elif 'h1_ema_slope' in df.columns:
        slope_col = 'h1_ema_slope'

    if slope_col is not None:
        s = df[slope_col]
        s_valid = s.dropna()
        if len(s_valid) >= 10:
            q_low = s_valid.quantile(0.33)
            q_high = s_valid.quantile(0.66)
            cond_down = s <= q_low
            cond_up = s >= q_high
            df['h1_slope_state'] = np.select(
                [cond_down, cond_up],
                ['SLOPE_DOWN', 'SLOPE_UP'],
                default='SLOPE_FLAT'
            )
        else:
            df['h1_slope_state'] = 'SLOPE_FLAT'
    else:
        df['h1_slope_state'] = 'UNKNOWN'

    # h1_rsi_state
    if 'h1_rsi14_pct_rank' in df.columns:
        r = df['h1_rsi14_pct_rank']
        cond_os = r <= 20
        cond_ob = r >= 80
        df['h1_rsi_state'] = np.select(
            [cond_os, cond_ob],
            ['OS', 'OB'],
            default='NEUTRAL'
        )
    else:
        df['h1_rsi_state'] = 'UNKNOWN'

    # h1_position_vs_swings
    required = {'h1_last_high', 'h1_last_low', 'h1_atr14', 'm5_close'}
    if required.issubset(df.columns):
        last_high = df['h1_last_high']
        last_low = df['h1_last_low']
        atr = df['h1_atr14']
        close_m5 = df['m5_close']
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_low = (close_m5 - last_low) / atr
            dist_high = (last_high - close_m5) / atr
        cond_near_low = dist_low <= 0.5
        cond_near_high = dist_high <= 0.5
        df['h1_position_vs_swings'] = np.select(
            [cond_near_low, cond_near_high],
            ['NEAR_LAST_LOW', 'NEAR_LAST_HIGH'],
            default='MIDDLE'
        )
    else:
        df['h1_position_vs_swings'] = 'UNKNOWN'

    return df


def add_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    # m5_rsi_state
    if 'm5_rsi14_pct_rank' in df.columns:
        r = df['m5_rsi14_pct_rank']
        cond_os = r <= 20
        cond_ob = r >= 80
        df['m5_rsi_state'] = np.select(
            [cond_os, cond_ob],
            ['OS', 'OB'],
            default='NEUTRAL'
        )
    else:
        df['m5_rsi_state'] = 'UNKNOWN'

    # m5_pos_in_recent_range
    required = {'m5_recent_high', 'm5_recent_low', 'm5_close'}
    if required.issubset(df.columns):
        rh = df['m5_recent_high']
        rl = df['m5_recent_low']
        close_m5 = df['m5_close']
        width = rh - rl
        # zabezpieczenie przed zerowym zakresem
        width_safe = width.replace(0, np.nan)
        pos = (close_m5 - rl) / width_safe
        # jeśli width=0 lub NaN -> pos=NaN, dostanie default=MIDDLE
        cond_low = pos <= 0.25
        cond_high = pos >= 0.75
        df['m5_pos_in_recent_range'] = np.select(
            [cond_low, cond_high],
            ['NEAR_LOW', 'NEAR_HIGH'],
            default='MIDDLE'
        )
    else:
        df['m5_pos_in_recent_range'] = 'UNKNOWN'

    # m5_vol_bucket
    if 'm5_atr14_pct_rank' in df.columns:
        atr_pct = df['m5_atr14_pct_rank']
        cond_low = atr_pct < 33
        cond_high = atr_pct > 66
        df['m5_vol_bucket'] = np.select(
            [cond_low, cond_high],
            ['VOL_LOW', 'VOL_HIGH'],
            default='VOL_MID'
        )
    else:
        df['m5_vol_bucket'] = 'UNKNOWN'

    return df


def add_sr_features(df: pd.DataFrame) -> pd.DataFrame:
    # near_support
    if 'm5_nearest_support_dist_atr' in df.columns:
        d = df['m5_nearest_support_dist_atr']
        cond_none = d.isna()
        cond_near = d <= 0.5
        df['near_support'] = np.select(
            [cond_none, cond_near],
            ['SUPPORT_NONE', 'SUPPORT_NEAR'],
            default='SUPPORT_FAR'
        )
    else:
        df['near_support'] = 'UNKNOWN'

    # near_resistance
    if 'm5_nearest_resistance_dist_atr' in df.columns:
        d = df['m5_nearest_resistance_dist_atr']
        cond_none = d.isna()
        cond_near = d <= 0.5
        df['near_resistance'] = np.select(
            [cond_none, cond_near],
            ['RESISTANCE_NONE', 'RESISTANCE_NEAR'],
            default='RESISTANCE_FAR'
        )
    else:
        df['near_resistance'] = 'UNKNOWN'

    return df


def add_session_feature(df: pd.DataFrame) -> pd.DataFrame:
    # m5_ts → datetime
    if 'm5_ts' in df.columns:
        if not np.issubdtype(df['m5_ts'].dtype, np.datetime64):
            ts = pd.to_datetime(df['m5_ts'], errors='coerce', utc=True)
            ts = ts.dt.tz_convert(None)
            df['m5_ts'] = ts
    else:
        df['m5_ts'] = pd.NaT

    hours = df['m5_ts'].dt.hour
    session = np.full(len(df), 'UNKNOWN', dtype=object)

    # Asia: 0–6, Europe: 7–15, US: 16–22
    session[(hours >= 0) & (hours <= 6)] = 'ASIA'
    session[(hours >= 7) & (hours <= 15)] = 'EU'
    session[(hours >= 16) & (hours <= 22)] = 'US'
    session[hours.isna()] = 'UNKNOWN'

    df['session_bucket'] = session
    return df


# ---------- Clustering ----------

def add_clusters(df: pd.DataFrame, cluster_gap_bars: int = 10, bar_minutes: int = 5) -> pd.DataFrame:
    """
    Dodaje:
      - cluster_id (per instrument, w oparciu o m5_ts)
      - cluster_size
      - cluster_weight = 1 / cluster_size

    Klaster łamie się, jeśli przerwa między snapshotami > cluster_gap_bars * bar_minutes.
    """
    df = df.copy()

    if 'instrument' not in df.columns:
        df['instrument'] = 'UNKNOWN'

    # m5_ts → datetime
    if 'm5_ts' not in df.columns or not np.issubdtype(df['m5_ts'].dtype, np.datetime64):
        df['m5_ts'] = pd.to_datetime(df.get('m5_ts'), errors='coerce', utc=True)
        df['m5_ts'] = df['m5_ts'].dt.tz_convert(None)

    cluster_gap_minutes = cluster_gap_bars * bar_minutes

    # sort + zapamiętanie oryginalnego indeksu
    df_sorted = df.sort_values(['instrument', 'm5_ts']).reset_index(names='orig_index')

    cluster_ids = []
    current_cluster = 0
    prev_inst = None
    prev_ts = None

    for _, row in df_sorted.iterrows():
        inst = row['instrument']
        ts = row['m5_ts']

        if pd.isna(ts):
            current_cluster += 1
        else:
            if (prev_inst is None) or (inst != prev_inst) or (prev_ts is None):
                current_cluster += 1
            else:
                dt = ts - prev_ts
                dt_min = dt.total_seconds() / 60.0
                if dt_min > cluster_gap_minutes:
                    current_cluster += 1
                # else: ten sam klaster

        cluster_ids.append(current_cluster)
        prev_inst = inst
        prev_ts = ts

    df_sorted['cluster_id'] = cluster_ids

    # przywróć oryginalną kolejność
    df_sorted = df_sorted.set_index('orig_index').sort_index()

    # cluster_size i wagi
    df_sorted['cluster_size'] = df_sorted.groupby(['instrument', 'cluster_id'])['cluster_id'].transform('size')
    df_sorted['cluster_weight'] = 1.0 / df_sorted['cluster_size'].replace(0, np.nan)

    return df_sorted


# ---------- Group stats ----------

def compute_group_stats(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    """
    Liczy statystyki dla longa/shorta, zarówno plain jak i cluster-weighted.
    Zakłada, że w df są kolumny:
      - max_R_long, min_R_long, bar_of_max_R_long
      - max_R_short, min_R_short, bar_of_max_R_short
      - cluster_id, cluster_weight
    """
    # pomocnicze flagi na poziomie DF
    for side in ['long', 'short']:
        max_col = f'max_R_{side}'
        min_col = f'min_R_{side}'
        if max_col not in df.columns:
            df[max_col] = np.nan
        if min_col not in df.columns:
            df[min_col] = np.nan

        df[f'has_{side}_ge_1R'] = df[max_col] >= 1.0
        df[f'has_{side}_ge_2R'] = df[max_col] >= 2.0
        df[f'has_{side}_ge_3R'] = df[max_col] >= 3.0
        df[f'has_{side}_ge_5R'] = df[max_col] >= 5.0
        df[f'min_{side}_le_minus1R'] = df[min_col] <= -1.0

    def agg_group(grp: pd.DataFrame) -> pd.Series:
        res = {}
        res['n_trades'] = len(grp)
        res['n_clusters'] = grp['cluster_id'].nunique()

        for side in ['long', 'short']:
            prefix = side

            max_col = f'max_R_{side}'
            min_col = f'min_R_{side}'
            bar_max_col = f'bar_of_max_R_{side}'
            has1 = f'has_{side}_ge_1R'
            has2 = f'has_{side}_ge_2R'
            has3 = f'has_{side}_ge_3R'
            has5 = f'has_{side}_ge_5R'
            minle1 = f'min_{side}_le_minus1R'

            mask_valid = grp[max_col].notna()
            w = grp.loc[mask_valid, 'cluster_weight'].fillna(0.0)
            w_sum = w.sum()

            max_vals = grp.loc[mask_valid, max_col]
            min_vals = grp.loc[mask_valid, min_col]
            if bar_max_col in grp.columns:
                bar_max_vals = grp.loc[mask_valid, bar_max_col]
            else:
                bar_max_vals = pd.Series(index=max_vals.index, dtype=float)

            # --- plain stats ---
            res[f'{prefix}_avg_max_R_plain'] = max_vals.mean() if len(max_vals) > 0 else np.nan
            res[f'{prefix}_median_max_R_plain'] = max_vals.median() if len(max_vals) > 0 else np.nan
            res[f'{prefix}_avg_min_R_plain'] = min_vals.mean() if len(min_vals) > 0 else np.nan

            res[f'{prefix}_pct_ge_1R_plain'] = grp.loc[mask_valid, has1].mean() if mask_valid.any() else np.nan
            res[f'{prefix}_pct_ge_2R_plain'] = grp.loc[mask_valid, has2].mean() if mask_valid.any() else np.nan
            res[f'{prefix}_pct_ge_3R_plain'] = grp.loc[mask_valid, has3].mean() if mask_valid.any() else np.nan
            res[f'{prefix}_pct_ge_5R_plain'] = grp.loc[mask_valid, has5].mean() if mask_valid.any() else np.nan
            res[f'{prefix}_pct_min_le_-1R_plain'] = grp.loc[mask_valid, minle1].mean() if mask_valid.any() else np.nan

            res[f'{prefix}_median_bar_of_max_plain'] = bar_max_vals.median() if len(bar_max_vals) > 0 else np.nan

            # --- weighted stats (cluster-weighted) ---
            if w_sum > 0:
                res[f'{prefix}_avg_max_R_weighted'] = np.average(max_vals, weights=w)
                res[f'{prefix}_avg_min_R_weighted'] = np.average(min_vals, weights=w)

                res[f'{prefix}_pct_ge_1R_weighted'] = np.average(
                    grp.loc[mask_valid, has1].astype(float),
                    weights=w
                )
                res[f'{prefix}_pct_ge_2R_weighted'] = np.average(
                    grp.loc[mask_valid, has2].astype(float),
                    weights=w
                )
                res[f'{prefix}_pct_ge_3R_weighted'] = np.average(
                    grp.loc[mask_valid, has3].astype(float),
                    weights=w
                )
                res[f'{prefix}_pct_ge_5R_weighted'] = np.average(
                    grp.loc[mask_valid, has5].astype(float),
                    weights=w
                )
                res[f'{prefix}_pct_min_le_-1R_weighted'] = np.average(
                    grp.loc[mask_valid, minle1].astype(float),
                    weights=w
                )
            else:
                res[f'{prefix}_avg_max_R_weighted'] = np.nan
                res[f'{prefix}_avg_min_R_weighted'] = np.nan
                res[f'{prefix}_pct_ge_1R_weighted'] = np.nan
                res[f'{prefix}_pct_ge_2R_weighted'] = np.nan
                res[f'{prefix}_pct_ge_3R_weighted'] = np.nan
                res[f'{prefix}_pct_ge_5R_weighted'] = np.nan
                res[f'{prefix}_pct_min_le_-1R_weighted'] = np.nan

        # --- proste score dla longa/shorta ---
        for side in ['long', 'short']:
            prefix = side
            p3 = res.get(f'{prefix}_pct_ge_3R_weighted', np.nan)
            p5 = res.get(f'{prefix}_pct_ge_5R_weighted', np.nan)
            pdraw = res.get(f'{prefix}_pct_min_le_-1R_weighted', np.nan)
            if np.isnan(p3) or np.isnan(p5) or np.isnan(pdraw):
                score = np.nan
            else:
                score = 1.0 * p3 + 0.5 * p5 - 0.7 * pdraw
            res[f'{prefix}_entry_score'] = score

        return pd.Series(res)

    grouped = df.groupby(group_cols, dropna=False).apply(agg_group).reset_index()
    return grouped


# ---------- Main ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Entry Grouper: szukanie setupów wejścia + cluster-weighted statystyki."
    )
    p.add_argument(
        "--snapshots",
        type=str,
        default="brent_snapshots_labeled_v2.csv",
        help="Ścieżka do pliku snapshotów z labelera (default: brent_snapshots_labeled_v2.csv)",
    )
    p.add_argument(
        "--out-setups",
        type=str,
        default="brent_entry_setups_v1.csv",
        help="Ścieżka wyjściowa tabeli setupów (default: brent_entry_setups_v1.csv)",
    )
    p.add_argument(
        "--out-snapshots",
        type=str,
        default="brent_snapshots_enriched_v1.csv",
        help="Ścieżka wyjściowa snapshotów z dodatkowymi kolumnami (default: brent_snapshots_enriched_v1.csv)",
    )
    p.add_argument(
        "--cluster-gap-bars",
        type=int,
        default=10,
        help="Max liczba świec 5m przerwy w obrębie jednego klastra (default: 10).",
    )
    p.add_argument(
        "--min-trades-per-group",
        type=int,
        default=30,
        help="Minimalna liczba snapshotów w grupie, żeby ją zostawić w wynikach (default: 30).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    snapshots_path = Path(args.snapshots)
    out_setups_path = Path(args.out_setups)
    out_snaps_enriched_path = Path(args.out_snapshots)

    print("==> Wczytuję snapshoty...")
    df = pd.read_csv(snapshots_path, sep=";")
    print(f"  Załadowano {len(df)} wierszy.")

    # trade_id fallback (jakby nie było z labelera)
    if 'trade_id' not in df.columns:
        df.insert(0, 'trade_id', np.arange(len(df), dtype=int))

    # Featury
    print("==> Liczę cechy D1...")
    df = add_daily_features(df)

    print("==> Liczę cechy H1...")
    df = add_h1_features(df)

    print("==> Liczę cechy M5...")
    df = add_m5_features(df)

    print("==> Liczę cechy SR...")
    df = add_sr_features(df)

    print("==> Wyznaczam session bucket...")
    df = add_session_feature(df)

    print("==> Buduję klastry snapshotów...")
    df = add_clusters(df, cluster_gap_bars=args.cluster_gap_bars)

    # Zapis snapshotów z dodatkowymi kolumnami
    print(f"==> Zapis snapshotów z dodatkowymi kolumnami do {out_snaps_enriched_path} ...")
    df.to_csv(out_snaps_enriched_path, sep=";", index=False)

    # Kolumny grupujące (setup definicja)
    group_cols = [
        'instrument',
        'daily_trend',
        'daily_vol_bucket',
        'daily_rsi_state',
        'h1_trend',
        'h1_slope_state',
        'h1_rsi_state',
        'h1_position_vs_swings',
        'm5_rsi_state',
        'm5_pos_in_recent_range',
        'm5_vol_bucket',
        'near_support',
        'near_resistance',
        'session_bucket',
    ]

    print("==> Liczę statystyki grup wejściowych...")
    setups = compute_group_stats(df, group_cols=group_cols)

    # filtr po minimalnej liczbie trade'ów
    before = len(setups)
    setups = setups[setups['n_trades'] >= args.min_trades_per_group].reset_index(drop=True)
    after = len(setups)
    print(f"  Grupy przed filtrem: {before}, po filtrze min_trades={args.min_trades_per_group}: {after}")

    print(f"==> Zapisuję tabelę setupów do {out_setups_path} ...")
    setups.to_csv(out_setups_path, sep=";", index=False)
    print("==> GOTOWE.")


if __name__ == "__main__":
    main()
