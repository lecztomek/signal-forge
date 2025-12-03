import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =======================
# Konfiguracja / stałe
# =======================

GROUP_COLS = [
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


# =======================
# Helpers: parsowanie
# =======================

def parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_int_list(s: str):
    return [int(float(x)) for x in s.split(",") if x.strip() != ""]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip() != ""]


# =======================
# Buckety / kierunki
# =======================

def bucket_bar_phase(bar_offset: int, max_bars: int) -> str:
    if max_bars <= 0 or pd.isna(bar_offset):
        return "UNKNOWN"
    ratio = bar_offset / max_bars
    if ratio <= 1/3:
        return "EARLY"
    if ratio <= 2/3:
        return "MID"
    return "LATE"


def dir_from_delta(delta: float, eps: float = 2.0) -> str:
    if np.isnan(delta):
        return "UNKNOWN"
    if delta > eps:
        return "UP"
    if delta < -eps:
        return "DOWN"
    return "FLAT"


def sr_context_for_side(side: str, dist_atr: float, has_sr: bool) -> str:
    if not has_sr or pd.isna(dist_atr):
        return "NO_SR"
    if dist_atr <= 0.5:
        return "NEAR_SR"
    if dist_atr <= 1.5:
        return "MID_SR"
    return "FAR_SR"


# =======================
# Entry: wybór setupów
# =======================

def select_setups(entry_setups: pd.DataFrame, side: str, min_entry_score: float, min_clusters: int):
    """
    Wybiera setupy o sensownym entry_score i min_clusters.
    """
    score_col = f"{side}_entry_score"
    setups = entry_setups.copy()

    if score_col not in setups.columns:
        raise ValueError(f"Brak kolumny {score_col} w entry_setups CSV.")

    setups = setups[
        (setups["n_clusters"] >= min_clusters)
        & (setups[score_col].notna())
        & (setups[score_col] >= min_entry_score)
    ].reset_index(drop=True)

    return setups


def get_trades_for_setups(
    snapshots_enriched: pd.DataFrame,
    setups: pd.DataFrame,
):
    """
    Zwraca DF z kolumnami ['trade_id', 'entry_ts'] dla trade'ów należących do zadanych setupów.
    """
    snaps = snapshots_enriched.copy()
    missing = [c for c in GROUP_COLS if c not in snaps.columns]
    if missing:
        raise ValueError(f"Brak kolumn {missing} w snapshots_enriched.")

    if "trade_id" not in snaps.columns:
        raise ValueError("Brak kolumny 'trade_id' w snapshots_enriched.")

    if "entry_ts" not in snaps.columns:
        if "m5_ts" in snaps.columns:
            snaps["entry_ts"] = snaps["m5_ts"]
        else:
            raise ValueError("Brak 'entry_ts' i 'm5_ts' w snapshots_enriched.")

    all_trade_rows = []

    for _, srow in setups.iterrows():
        cond = pd.Series(True, index=snaps.index)
        for c in GROUP_COLS:
            cond &= snaps[c] == srow[c]
        subset = snaps.loc[cond, ["trade_id", "entry_ts"]].copy()
        all_trade_rows.append(subset)

    if not all_trade_rows:
        return pd.DataFrame(columns=["trade_id", "entry_ts"])

    trades = (
        pd.concat(all_trade_rows, axis=0)
        .drop_duplicates(subset=["trade_id"])
        .reset_index(drop=True)
    )
    return trades


# =======================
# Signature gating
# =======================

def build_signature_gates(
    signatures: pd.DataFrame,
    min_n_exits: int,
    min_avg_R_exit: float,
    allowed_R_buckets: list[str] | None,
):
    """
    Na podstawie exit_signatures buduje zbiory do "gatingu" exitu:

      - bar_phase
      - m5_rsi_state
      - rsi_dir
      - sr_context
      - m5_pos_in_recent_range
      - m5_vol_bucket

    Filtruje tylko "mocne" sygnatury: n_exits >= min_n_exits,
    avg_R_exit >= min_avg_R_exit, R_bucket ∈ allowed_R_buckets (jeśli podano).
    """
    sig = signatures.copy()

    if "n_exits" not in sig.columns or "avg_R_exit" not in sig.columns:
        raise ValueError("Brakuje 'n_exits' lub 'avg_R_exit' w exit_signatures CSV.")

    mask = (sig["n_exits"] >= min_n_exits) & (sig["avg_R_exit"] >= min_avg_R_exit)

    if allowed_R_buckets:
        if "R_bucket" not in sig.columns:
            raise ValueError("Brak kolumny 'R_bucket' w exit_signatures, a podano --sig-R-buckets.")
        mask &= sig["R_bucket"].isin(allowed_R_buckets)

    sig_filt = sig[mask].copy()
    print(f"==> Signatures: {len(sig)} wierszy, po filtrze: {len(sig_filt)}")

    # jeśli wszystko wycięliśmy, poluzuj kryteria: użyj wszystkich
    if sig_filt.empty:
        print("  Uwaga: po filtrze nie ma żadnych sygnatur. "
              "Używam wszystkich sygnatur jako bazę gatingu.")
        sig_filt = sig

    def nonempty_set(col):
        if col not in sig_filt.columns:
            return None
        vals = sorted(x for x in sig_filt[col].dropna().unique() if str(x) != "")
        if not vals:
            return None
        return set(vals)

    gates = {
        "bar_phase": nonempty_set("bar_phase"),
        "m5_rsi_state": nonempty_set("m5_rsi_state"),
        "rsi_dir": nonempty_set("rsi_dir"),
        "sr_context": nonempty_set("sr_context"),
        "m5_pos_in_recent_range": nonempty_set("m5_pos_in_recent_range"),
        "m5_vol_bucket": nonempty_set("m5_vol_bucket"),
        # potencjalnie można dodać tu daily/h1/session, ale na razie nie gate'ujemy po tym
    }

    print("==> Wyznaczone bramki (signature gates):")
    for k, v in gates.items():
        if v is None:
            print(f"  {k}: brak gatingu (None)")
        else:
            print(f"  {k}: {sorted(list(v))}")

    return gates


def passes_gate(value: str, allowed: set[str] | None) -> bool:
    """
    Czy wartość mieści się w zbiorze dopuszczalnych?
    Jeśli allowed is None => brak gatingu => zawsze True.
    """
    if allowed is None:
        return True
    if value in allowed:
        return True
    # jeśli wartość nieznana, ale gating istnieje, to traktuj jako False
    return False


# =======================
# Symulator exitu v3
# =======================

def simulate_single_trade_exit_v3(
    trade_path: pd.DataFrame,
    side: str,
    min_target_R: float,
    trail_giveback_R: float,
    stop_R: float,
    max_bars: int,
    rsi_exit_threshold: float,
    rsi_min_R: float,
    sr_dist_threshold_atr: float,
    gates: dict,
) -> tuple[float, int, pd.Timestamp, str]:
    """
    Zaawansowany exit v3 dla pojedynczego trade'u na podstawie paths_v3 + signature gating.

    trade_path: wszystkie wiersze jednego trade_id (posortowane po bar_offset).
    side: 'long' lub 'short'

    Zasady (dla longa; short symetryczny na RSI):

      1) hard stop: R <= stop_R  -> exit (reason='stop') [Działa zawsze]
      2) exit kontekstowy RSI+SR: [Działa tylko, jeśli spełnione bramki z signatures]
         - R >= rsi_min_R
         - signature gating:
             bar_phase ∈ gates["bar_phase"]           (EARLY/MID/LATE)
             m5_rsi_state ∈ gates["m5_rsi_state"]
             rsi_dir ∈ gates["rsi_dir"]               (UP/DOWN/FLAT)
             sr_context ∈ gates["sr_context"]         (NEAR_SR/MID_SR/...)
             m5_pos ∈ gates["m5_pos_in_recent_range"]
             m5_vol_bucket ∈ gates["m5_vol_bucket"]
         - RSI:
             long:  m5_rsi14 <= rsi_exit_threshold
             short: m5_rsi14 >= rsi_exit_threshold
         - SR:
             jeśli sr_dist_threshold_atr < 0 -> ignorujemy SR (tylko RSI)
             jeśli >= 0 -> wymagamy dystansu do SR <= próg
      3) trailing od MFE: [Działa zawsze]
         - R >= min_target_R
         - MFE - R >= trail_giveback_R  -> exit (reason='trail')
      4) timeout:
         - jeśli nic nie zadziała do max_bars -> exit na ostatnim barze (reason='timeout')

    Zwraca:
      (R_exit, exit_bar_offset, exit_ts, exit_reason)
    """
    col_R = "R_long" if side == "long" else "R_short"
    col_MFE = "MFE_R_long" if side == "long" else "MFE_R_short"

    path = trade_path.sort_values("bar_offset")
    R = path[col_R].to_numpy()
    MFE = path[col_MFE].to_numpy()
    offsets = path["bar_offset"].to_numpy()
    bar_ts = path["bar_ts"].to_numpy()

    n = len(path)
    if n == 0:
        return np.nan, np.nan, pd.NaT, "no_data"

    # RSI per bar
    if "m5_rsi14" in path.columns:
        rsi_arr = path["m5_rsi14"].to_numpy()
    else:
        rsi_arr = np.full(n, np.nan, dtype=float)

    # M5 states
    m5_rsi_state_arr = path["m5_rsi_state"].astype(str).to_numpy() if "m5_rsi_state" in path.columns else np.array(["UNKNOWN"] * n)
    pos_range_arr = path["m5_pos_in_recent_range"].astype(str).to_numpy() if "m5_pos_in_recent_range" in path.columns else np.array(["UNKNOWN"] * n)
    vol_bucket_arr = path["m5_vol_bucket"].astype(str).to_numpy() if "m5_vol_bucket" in path.columns else np.array(["UNKNOWN"] * n)

    # SR distance per bar
    if side == "long":
        dist_col = "m5_nearest_resistance_dist_atr"
    else:
        dist_col = "m5_nearest_support_dist_atr"

    if dist_col in path.columns:
        sr_dist_arr = path[dist_col].to_numpy()
    else:
        sr_dist_arr = np.full(n, np.nan, dtype=float)

    # ograniczamy do max_bars
    max_idx = np.searchsorted(offsets, max_bars, side="right") - 1
    if max_idx < 0:
        max_idx = 0
    max_idx = min(max_idx, n - 1)

    exit_idx = None
    exit_reason = "timeout"

    # pętla po barach
    for i in range(0, max_idx + 1):
        r = R[i]
        mfe = MFE[i]
        rsi = rsi_arr[i]
        sr_dist = sr_dist_arr[i]
        bar_offset_i = offsets[i]

        # 1) hard stop
        if r <= stop_R:
            exit_idx = i
            exit_reason = "stop"
            break

        # trailing (działa niezależnie od gatingu)
        trailing_triggered = False
        if r >= min_target_R:
            giveback = mfe - r
            if giveback >= trail_giveback_R:
                trailing_triggered = True

        # 2) Kontekstowy exit RSI+SR + signature gating
        # Najpierw sprawdźmy, czy w ogóle kontekst (gates) pozwala na exit
        context_ok = True

        # bar_phase
        phase = bucket_bar_phase(bar_offset_i, max_bars)
        if not passes_gate(phase, gates.get("bar_phase")):
            context_ok = False

        m5_rsi_state = m5_rsi_state_arr[i]
        if not passes_gate(m5_rsi_state, gates.get("m5_rsi_state")):
            context_ok = False

        # rsi_dir na podstawie delta rsi
        if i > 0 and not np.isnan(rsi_arr[i]) and not np.isnan(rsi_arr[i - 1]):
            delta_rsi = rsi_arr[i] - rsi_arr[i - 1]
            rsi_dir = dir_from_delta(delta_rsi, eps=2.0)
        else:
            rsi_dir = "UNKNOWN"
        if not passes_gate(rsi_dir, gates.get("rsi_dir")):
            context_ok = False

        # sr_context z dystansu
        if not np.isnan(sr_dist):
            sr_ctx = sr_context_for_side(side, float(sr_dist), True)
        else:
            sr_ctx = "NO_SR"
        if not passes_gate(sr_ctx, gates.get("sr_context")):
            context_ok = False

        pos_range = pos_range_arr[i]
        if not passes_gate(pos_range, gates.get("m5_pos_in_recent_range")):
            context_ok = False

        vol_bucket = vol_bucket_arr[i]
        if not passes_gate(vol_bucket, gates.get("m5_vol_bucket")):
            context_ok = False

        # 2a) jeśli gating się nie zgadza -> nie robimy exit RSI+SR, ale trailing + stop nadal obowiązują
        rsi_sr_triggered = False
        if context_ok and r >= rsi_min_R:
            rsi_cond = False
            if not np.isnan(rsi):
                if side == "long":
                    rsi_cond = rsi <= rsi_exit_threshold
                else:
                    rsi_cond = rsi >= rsi_exit_threshold

            if sr_dist_threshold_atr < 0:
                sr_cond = True  # ignorujemy SR, liczy się tylko RSI
            else:
                sr_cond = (not np.isnan(sr_dist)) and (sr_dist <= sr_dist_threshold_atr)

            if rsi_cond and sr_cond:
                rsi_sr_triggered = True

        # kolejność priorytetów:
        # 1) stop
        # 2) rsi_sr (jeśli gating OK)
        # 3) trailing
        if rsi_sr_triggered:
            exit_idx = i
            exit_reason = "rsi_sr"
            break
        elif trailing_triggered:
            exit_idx = i
            exit_reason = "trail"
            break

    # 4) timeout / brak exitu wcześniej
    if exit_idx is None:
        exit_idx = max_idx
        exit_reason = "timeout"

    R_exit = float(R[exit_idx])
    exit_bar_offset = int(offsets[exit_idx])
    exit_ts = pd.to_datetime(bar_ts[exit_idx])

    return R_exit, exit_bar_offset, exit_ts, exit_reason


# =======================
# Backtest: 1 trade naraz
# =======================

def backtest_one_trade_at_a_time_v3(
    trades_meta: pd.DataFrame,
    paths_v3: pd.DataFrame,
    side: str,
    min_target_R_values,
    trail_giveback_R_values,
    stop_R_values,
    max_bars_values,
    rsi_exit_threshold_values,
    rsi_min_R_values,
    sr_dist_threshold_values,
    gates: dict,
) -> pd.DataFrame:
    """
    trades_meta: DF z ['trade_id', 'entry_ts']
    paths_v3: DF z paths_v3 (per-bar ścieżki + feature'y)
    side: 'long' lub 'short'
    gates: słownik z bramkami signature (bar_phase, m5_rsi_state, itp.)

    Zwraca DF z wierszem na każdą kombinację parametrów exitu + statystyki.
    """
    trades = trades_meta.copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], errors="coerce", utc=True)
    trades["entry_ts"] = trades["entry_ts"].dt.tz_convert(None)
    trades = trades.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    paths = paths_v3.copy()
    paths["bar_ts"] = pd.to_datetime(paths["bar_ts"], errors="coerce", utc=True)
    paths["bar_ts"] = paths["bar_ts"].dt.tz_convert(None)
    paths = paths.sort_values(["trade_id", "bar_offset"])

    results = []

    for min_target_R in min_target_R_values:
        for trail_giveback_R in trail_giveback_R_values:
            for stop_R in stop_R_values:
                for max_bars in max_bars_values:
                    for rsi_exit_threshold in rsi_exit_threshold_values:
                        for rsi_min_R in rsi_min_R_values:
                            for sr_dist_threshold_atr in sr_dist_threshold_values:

                                trade_Rs = []
                                position_open = False
                                current_time = None

                                n_stop = 0
                                n_trail = 0
                                n_rsi_sr = 0
                                n_timeout = 0

                                for _, trow in trades.iterrows():
                                    trade_id = trow["trade_id"]
                                    entry_ts = trow["entry_ts"]

                                    if position_open and entry_ts <= current_time:
                                        # już jesteśmy w pozycji -> kolejny sygnał ignorujemy
                                        continue

                                    trade_path = paths[paths["trade_id"] == trade_id]
                                    if trade_path.empty:
                                        continue

                                    R_exit, exit_bar_offset, exit_ts, reason = simulate_single_trade_exit_v3(
                                        trade_path=trade_path,
                                        side=side,
                                        min_target_R=min_target_R,
                                        trail_giveback_R=trail_giveback_R,
                                        stop_R=stop_R,
                                        max_bars=max_bars,
                                        rsi_exit_threshold=rsi_exit_threshold,
                                        rsi_min_R=rsi_min_R,
                                        sr_dist_threshold_atr=sr_dist_threshold_atr,
                                        gates=gates,
                                    )

                                    if np.isnan(R_exit):
                                        continue

                                    trade_Rs.append(R_exit)

                                    if reason == "stop":
                                        n_stop += 1
                                    elif reason == "trail":
                                        n_trail += 1
                                    elif reason == "rsi_sr":
                                        n_rsi_sr += 1
                                    elif reason == "timeout":
                                        n_timeout += 1

                                    position_open = True
                                    current_time = exit_ts

                                ntr = len(trade_Rs)
                                if ntr == 0:
                                    avg_R = np.nan
                                    med_R = np.nan
                                    winrate = np.nan
                                    pct_ge_2R = np.nan
                                    pct_ge_3R = np.nan
                                    sum_R = 0.0
                                    max_dd = np.nan
                                    pct_stop = pct_trail = pct_rsi_sr = pct_timeout = np.nan
                                else:
                                    R_arr = np.array(trade_Rs, dtype=float)
                                    avg_R = float(np.mean(R_arr))
                                    med_R = float(np.median(R_arr))
                                    winrate = float(np.mean(R_arr > 0.0))
                                    pct_ge_2R = float(np.mean(R_arr >= 2.0))
                                    pct_ge_3R = float(np.mean(R_arr >= 3.0))
                                    sum_R = float(np.sum(R_arr))

                                    equity = np.cumsum(R_arr)
                                    peak = np.maximum.accumulate(equity)
                                    dd = equity - peak
                                    max_dd = float(dd.min())

                                    pct_stop = n_stop / ntr
                                    pct_trail = n_trail / ntr
                                    pct_rsi_sr = n_rsi_sr / ntr
                                    pct_timeout = n_timeout / ntr

                                results.append(
                                    {
                                        "side": side,
                                        "min_target_R": min_target_R,
                                        "trail_giveback_R": trail_giveback_R,
                                        "stop_R": stop_R,
                                        "max_bars": max_bars,
                                        "rsi_exit_threshold": rsi_exit_threshold,
                                        "rsi_min_R": rsi_min_R,
                                        "sr_dist_threshold_atr": sr_dist_threshold_atr,
                                        "n_trades": ntr,
                                        "avg_R": avg_R,
                                        "median_R": med_R,
                                        "winrate": winrate,
                                        "pct_ge_2R": pct_ge_2R,
                                        "pct_ge_3R": pct_ge_3R,
                                        "sum_R": sum_R,
                                        "max_drawdown_R": max_dd,
                                        "pct_exit_stop": pct_stop,
                                        "pct_exit_trail": pct_trail,
                                        "pct_exit_rsi_sr": pct_rsi_sr,
                                        "pct_exit_timeout": pct_timeout,
                                    }
                                )

    return pd.DataFrame(results)


# =======================
# CLI
# =======================

def parse_args():
    p = argparse.ArgumentParser(
        description="Exit-lab v3: grid-search kontekstowych exitów (R + RSI + SR) "
                    "z automatycznym gatingiem na podstawie exit_signatures."
    )
    p.add_argument(
        "--paths-v3",
        type=str,
        default="brent_paths_v3.csv",
        help="Ścieżka do pliku paths_v3 (per-bar ścieżki z feature'ami).",
    )
    p.add_argument(
        "--snapshots-enriched",
        type=str,
        default="brent_snapshots_enriched_v1.csv",
        help="Ścieżka do snapshots_enriched z groupera.",
    )
    p.add_argument(
        "--entry-setups",
        type=str,
        default="brent_entry_setups_v1.csv",
        help="Ścieżka do tabeli setupów z groupera.",
    )
    p.add_argument(
        "--signatures",
        type=str,
        default="exit_signatures_long_v1.csv",
        help="Ścieżka do pliku exit_signatures dla danej strony (long/short).",
    )
    p.add_argument(
        "--side",
        type=str,
        choices=["long", "short"],
        default="long",
        help="Którą stronę testujemy (long/short).",
    )
    p.add_argument(
        "--min-entry-score",
        type=float,
        default=0.2,
        help="Minimalny entry_score (dla danej strony), żeby setup trafił do universum.",
    )
    p.add_argument(
        "--min-clusters",
        type=int,
        default=20,
        help="Minimalna liczba klastrów w setupie.",
    )
    p.add_argument(
        "--out-results",
        type=str,
        default="exit_lab_v3_results_long.csv",
        help="Ścieżka wyjściowa wyników grid-search.",
    )

    # parametry exitu – domyślne zakresy, można nadpisać
    p.add_argument(
        "--min-target-R",
        type=str,
        default="1.0,2.0,3.0",
        help="Lista wartości min_target_R, np. '1.0,2.0,3.0'",
    )
    p.add_argument(
        "--trail-giveback-R",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Lista wartości trail_giveback_R, np. '0.5,1.0,1.5'",
    )
    p.add_argument(
        "--stop-R",
        type=str,
        default="-1.0",
        help="Lista wartości stop_R (ujemne), np. '-1.0,-1.5'",
    )
    p.add_argument(
        "--max-bars",
        type=str,
        default="100,200,288",
        help="Lista maksymalnych barów w trade, np. '100,200,288'",
    )
    p.add_argument(
        "--rsi-exit-threshold",
        type=str,
        default="45,50,55",
        help="Lista progów RSI do exitu, np. '45,50,55'. "
             "Dla longa: exit gdy RSI <= próg, dla shorta: RSI >= próg.",
    )
    p.add_argument(
        "--rsi-min-R",
        type=str,
        default="1.0,2.0",
        help="Lista minimalnego R, od którego w ogóle rozważamy RSI-exit, np. '1.0,2.0'.",
    )
    p.add_argument(
        "--sr-dist-threshold-atr",
        type=str,
        default="-1,0.5",
        help="Lista progów dystansu do SR w ATR, np. '-1,0.5'. "
             "Wartość <0 = ignoruj SR (czysty RSI-exit), >=0 = wymagaj dystansu <= próg.",
    )

    # parametry filtracji signatures
    p.add_argument(
        "--sig-min-n-exits",
        type=int,
        default=20,
        help="Minimalna liczba exitów w sygnaturze, żeby była użyta do gatingu.",
    )
    p.add_argument(
        "--sig-min-avg-R-exit",
        type=float,
        default=2.0,
        help="Minimalny avg_R_exit w sygnaturze, żeby była użyta do gatingu.",
    )
    p.add_argument(
        "--sig-R-buckets",
        type=str,
        default="2-3R,3-5R,5+R",
        help="Jakie R_bucket brać pod uwagę w signatures, np. '2-3R,3-5R,5+R'. "
             "Pusta wartość = brak filtra po R_bucket.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    paths_path = Path(args.paths_v3)
    snaps_enriched_path = Path(args.snapshots_enriched)
    entry_setups_path = Path(args.entry_setups)
    signatures_path = Path(args.signatures)
    out_results_path = Path(args.out_results)

    side = args.side

    min_target_R_values = parse_float_list(args.min_target_R)
    trail_giveback_R_values = parse_float_list(args.trail_giveback_R)
    stop_R_values = parse_float_list(args.stop_R)
    max_bars_values = parse_int_list(args.max_bars)
    rsi_exit_threshold_values = parse_float_list(args.rsi_exit_threshold)
    rsi_min_R_values = parse_float_list(args.rsi_min_R)
    sr_dist_threshold_values = parse_float_list(args.sr_dist_threshold_atr)

    sig_R_buckets = parse_str_list(args.sig_R_buckets) if args.sig_R_buckets.strip() != "" else None

    print("==> Wczytuję paths_v3...")
    paths_v3 = pd.read_csv(paths_path, sep=";")
    print(f"  paths_v3: {len(paths_v3)} wierszy.")

    print("==> Wczytuję snapshots_enriched...")
    snaps = pd.read_csv(snaps_enriched_path, sep=";")
    print(f"  snapshots_enriched: {len(snaps)} wierszy.")

    print("==> Wczytuję entry_setups...")
    entry_setups = pd.read_csv(entry_setups_path, sep=";")
    print(f"  entry_setups: {len(entry_setups)} wierszy.")

    print("==> Wczytuję exit_signatures...")
    signatures = pd.read_csv(signatures_path, sep=";")
    print(f"  exit_signatures: {len(signatures)} wierszy.")

    # 1) budujemy bramki z signatures
    print("==> Buduję signature gates na podstawie exit_signatures...")
    gates = build_signature_gates(
        signatures=signatures,
        min_n_exits=args.sig_min_n_exits,
        min_avg_R_exit=args.sig_min_avg_R_exit,
        allowed_R_buckets=sig_R_buckets,
    )

    # 2) wybór setupów (entry)
    print(f"==> Wybieram setupy dla side={side}, "
          f"min_entry_score={args.min_entry_score}, min_clusters={args.min_clusters}...")
    setups_sel = select_setups(
        entry_setups=entry_setups,
        side=side,
        min_entry_score=args.min_entry_score,
        min_clusters=args.min_clusters,
    )
    print(f"  Wybrano {len(setups_sel)} setupów.")

    if len(setups_sel) == 0:
        print("Brak setupów spełniających kryteria. Kończę.")
        return

    # 3) trade_id dla tych setupów
    print("==> Szukam trade_id należących do wybranych setupów...")
    trades_meta = get_trades_for_setups(snaps, setups_sel)
    print(f"  trade_id w universum: {len(trades_meta)} unikalnych.")

    if len(trades_meta) == 0:
        print("Brak trade'ów dla wybranych setupów. Kończę.")
        return

    # 4) paths_v3 pod te trade_id
    paths_sel = paths_v3[paths_v3["trade_id"].isin(trades_meta["trade_id"].unique())].copy()
    print(f"  paths_v3 dla universum: {len(paths_sel)} wierszy.")

    # 5) grid-search
    print("==> Odpalam grid-search kontekstowych exitów v3 (1 trade naraz + signature gating)...")
    results_df = backtest_one_trade_at_a_time_v3(
        trades_meta=trades_meta,
        paths_v3=paths_sel,
        side=side,
        min_target_R_values=min_target_R_values,
        trail_giveback_R_values=trail_giveback_R_values,
        stop_R_values=stop_R_values,
        max_bars_values=max_bars_values,
        rsi_exit_threshold_values=rsi_exit_threshold_values,
        rsi_min_R_values=rsi_min_R_values,
        sr_dist_threshold_values=sr_dist_threshold_values,
        gates=gates,
    )

    # sortowanie: np. najpierw po avg_R, potem po n_trades
    results_df = results_df.sort_values(
        by=["avg_R", "n_trades"],
        ascending=[False, False]
    ).reset_index(drop=True)

    print(f"==> Zapisuję wyniki do {out_results_path} ...")
    results_df.to_csv(out_results_path, sep=";", index=False)
    print("==> GOTOWE.")


if __name__ == "__main__":
    main()
