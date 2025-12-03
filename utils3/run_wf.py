#!/usr/bin/env python
"""
run_wf.py

Master orchestrator:
- znajduje snapshoty typu: brent_snapshots_5m_100_400.csv
- buduje z nich kroki walk–forward (train / test),
- dla każdego kroku:
    * scala pliki TRAIN w snapshots_train.csv
    * scala pliki TEST  w snapshots_test.csv
    * odpala TRAIN pipeline:
        - labeler_v2_entry_only.py
        - grouper.py (pełny – liczy setupy)
        - path_generator_v3.py
        - exit_signatures.py (long/short)
        - filter_entry_setups.py
        - exit_lab.py (long/short)
        - pick_best_exits.py  → exit_lab_v3_long_ranked.csv / exit_lab_v3_short_ranked.csv
    * odpala TEST pipeline:
        - grouper.py --only-enrich na snapshots_test.csv
        - strategy_backtest.py (używa entry_setups_*_good + ranked exit rules, grid po entry/exit)
    * opcjonalnie sprząta ciężkie pliki pośrednie TRAIN (jeśli użyto --cleanup).

Dodatkowo:
- jeśli plik wyjściowy danego etapu już istnieje, krok jest pomijany (skip)
  – pozwala to restartować pipeline bez liczenia wszystkiego od zera.

Snapshoty źródłowe (brent_snapshots_5m_100_*.csv) NIGDY nie są kasowane.
"""

from __future__ import annotations

import argparse
import errno
import glob
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd  # do combine'ów


# ---------------------------------------------------------------------------
# Konfiguracja DOMYŚLNA (zmienisz pod siebie)
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENT = "brent"
DEFAULT_TIMEFRAME = "5m"

# Jak wyglądają pliki snapshotów, na których działamy
DEFAULT_SNAPSHOTS_GLOB = "brent_snapshots_5m_100_*.csv"

# Świece M5 do labelera:
DEFAULT_CANDLES_5M = "brent_5m.csv"

# Długość okna WF w ilości plików/okresów (nie w dniach!):
DEFAULT_TRAIN_LEN = 8
DEFAULT_TEST_LEN = 1

# Parametry labelera / path_generator / exitów (podstawione z Twoich komend):
DEFAULT_MAX_BARS_AHEAD = 100
DEFAULT_CHECKPOINT_BARS = "10,50,100"

DEFAULT_MIN_TRADES_PER_GROUP = 20
DEFAULT_CLUSTER_GAP_BARS = 5

DEFAULT_MIN_IDEAL_R = 1.0
DEFAULT_MIN_EXITS_PER_GROUP = 20

DEFAULT_FILTER_ARGS = {
    "min_setup_trades": 20,
    "min_setup_clusters": 2,
    "long_min_entry_score": 0.5,
    "long_min_avg_max_R": 2.0,
    "long_min_pct_ge_2R": 0.4,
    "long_max_pct_min_le_1R": 0.98,
    "top_long": 15,
    "short_min_entry_score": 0.5,
    "short_min_avg_max_R": 2.0,
    "short_min_pct_ge_2R": 0.4,
    "short_max_pct_min_le_1R": 0.98,
    "top_short": 15,
}

DEFAULT_EXIT_LAB_ARGS = {
    "min_entry_score": 0.2,
    "min_clusters": 2,
    "min_target_R": "1.0,2.0,3.0",
    "trail_giveback_R": "0.5,1.0,1.5",
    "stop_R": "-1.0",
    "max_bars": "100,200,288",
    "rsi_exit_threshold": "45,50,55",
    "rsi_min_R": "1.0,2.0",
    "sr_dist_threshold_atr": "-1,0.5",
    "sig_min_n_exits": 20,
    "sig_min_avg_R_exit": 2.0,
    "sig_R_buckets": "2-3R,3-5R,5+R",
}

DEFAULT_PICK_ARGS = {
    "min_n_trades": 5,
    "max_dd_abs": 40,
    "min_avg_R": 0.0,
    "min_winrate": 0.15,
    "min_sum_R": 0,
    "top_k": 10,
}

# Strategie backtest – techniczne parametry (nie gridowane w exit_lab):
DEFAULT_ATR_COL = "m5_atr14"
DEFAULT_RISK_MULT = 1.0
DEFAULT_RSI_COL = "m5_rsi14"


# ---------------------------------------------------------------------------
# Model okresu (period) – jak w periods.py
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Period:
    sort_key: int  # negative, żeby sortować od najstarszego
    period_id: int
    filename: str
    instrument: str
    timeframe: str
    days_back: int
    skip_days: int
    start_offset_days: int
    end_offset_days: int


SNAPSHOT_FILENAME_RE = re.compile(
    r"""
    ^(?P<instrument>.+?)         # instrument
    _snapshots_
    (?P<tf>[^_]+)                # tf, np. 5m
    _
    (?P<days_back>\d+)           # days_back
    _
    (?P<skip_days>\d+)           # skip_days
    \.csv$
    """,
    re.VERBOSE,
)


def discover_periods(
    directory: str,
    snapshots_glob: str,
) -> List[Period]:
    """
    Szuka plików snapshotów, parsuje z nazw: instrument, tf, days_back, skip_days,
    buduje listę Period i sortuje od najstarszych do najnowszych.
    """
    pattern = os.path.join(directory, snapshots_glob)
    files = sorted(glob.glob(pattern))
    periods: List[Period] = []
    period_id = 1

    for path in files:
        fname = os.path.basename(path)
        m = SNAPSHOT_FILENAME_RE.match(fname)
        if not m:
            print(f"[run_wf] Pomijam plik niepasujący do wzorca: {fname}")
            continue

        instrument = m.group("instrument")
        tf = m.group("tf")
        days_back = int(m.group("days_back"))
        skip_days = int(m.group("skip_days"))

        start_offset_days = skip_days + days_back
        end_offset_days = skip_days

        sort_key = -start_offset_days  # older first

        periods.append(
            Period(
                sort_key=sort_key,
                period_id=period_id,
                filename=os.path.abspath(path),
                instrument=instrument,
                timeframe=tf,
                days_back=days_back,
                skip_days=skip_days,
                start_offset_days=start_offset_days,
                end_offset_days=end_offset_days,
            )
        )
        period_id += 1

    periods.sort()
    for idx, p in enumerate(periods, start=1):
        p.period_id = idx

    return periods


# ---------------------------------------------------------------------------
# Walk-forward kroki
# ---------------------------------------------------------------------------

@dataclass
class WFStep:
    step_id: int
    train_periods: List[Period]
    test_periods: List[Period]


def build_wf_steps(
    periods: List[Period],
    train_len: int,
    test_len: int,
) -> List[WFStep]:
    """
    Buduje listę kroków walk-forward: [train_len] + [test_len] okien przesuwanych
    po liście periods.
    """
    n = len(periods)
    window_len = train_len + test_len
    steps: List[WFStep] = []

    step_id = 1
    for start_idx in range(0, n - window_len + 1):
        train_slice = periods[start_idx : start_idx + train_len]
        test_slice = periods[start_idx + train_len : start_idx + window_len]
        steps.append(
            WFStep(
                step_id=step_id,
                train_periods=train_slice,
                test_periods=test_slice,
            )
        )
        step_id += 1

    return steps


def print_wf_plan(steps: List[WFStep]) -> None:
    if not steps:
        print("[run_wf] Nie ma możliwych kroków WF dla podanych parametrów.")
        return

    print(
        f"[run_wf] Plan walk-forward: {len(steps)} kroków "
        f"(train_len={len(steps[0].train_periods)}, test_len={len(steps[0].test_periods)})"
    )
    for s in steps:
        train_ids = ",".join(str(p.period_id) for p in s.train_periods)
        test_ids = ",".join(str(p.period_id) for p in s.test_periods)
        print(f"  Krok {s.step_id:2d}: train=[{train_ids}]  test=[{test_ids}]")


# ---------------------------------------------------------------------------
# Helpery: odpalanie komend + kasowanie plików
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    """
    Odpala komendę jak w konsoli. Rzuca wyjątkiem jeśli status != 0.
    """
    if cwd:
        print(f"[run_wf] RUN (cwd={cwd}): {' '.join(cmd)}")
    else:
        print(f"[run_wf] RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def outputs_exist(outputs) -> bool:
    """
    Sprawdza, czy wszystkie pliki wyjściowe istnieją.
    outputs: string lub lista stringów.
    """
    if isinstance(outputs, str):
        outputs = [outputs]
    exists_flags = []
    for p in outputs:
        if p is None:
            continue
        exists = os.path.exists(p)
        exists_flags.append(exists)
    return bool(exists_flags) and all(exists_flags)


def run_cmd_if_missing(cmd: List[str], outputs, cwd: Optional[str] = None) -> None:
    """
    Odpala komendę tylko jeśli któryś z plików wyjściowych NIE istnieje.
    """
    if outputs_exist(outputs):
        if isinstance(outputs, str):
            outs_str = outputs
        else:
            outs_str = ", ".join(outputs)
        print(f"[run_wf] SKIP – wyjście już istnieje: {outs_str}")
        return
    run_cmd(cmd, cwd=cwd)


def safe_remove(path: str) -> None:
    """
    Usuń plik jeśli istnieje. Ignoruje brak pliku.
    """
    try:
        os.remove(path)
        print(f"[run_wf] Usunięto plik: {path}")
    except FileNotFoundError:
        pass
    except OSError as e:
        if e.errno != errno.ENOENT:
            print(f"[run_wf] UWAGA: nie udało się usunąć {path}: {e}")


# ---------------------------------------------------------------------------
# Wybór "różnorodnych" reguł exitu z ranked CSV
# ---------------------------------------------------------------------------

def select_diverse_rule_indices(
    ranked_path: str,
    max_rules: int = 3,
) -> List[int]:
    """
    Wczytuje exit_lab_v3_*_ranked.csv i wybiera maksymalnie max_rules
    reguł, które RZECZYWIŚCIE różnią się parametrami exitu.

    Różnicujemy po kombinacji:
      (min_target_R, trail_giveback_R, stop_R, rsi_exit_threshold, rsi_min_R, sr_dist_threshold_atr)

    max_bars celowo IGNORUJEMY, żeby nie wybierać 3 wersji różniących się
    tylko max_bars.
    """
    if not os.path.exists(ranked_path):
        print(f"[run_wf] WARN: ranked file {ranked_path} nie istnieje – fallback do [0].")
        return [0]

    try:
        df = pd.read_csv(ranked_path, sep=";")
    except Exception as e:
        print(f"[run_wf] WARN: nie mogę wczytać {ranked_path}: {e} – fallback do [0].")
        return [0]

    if df.empty:
        print(f"[run_wf] WARN: ranked file {ranked_path} jest pusty – fallback do [0].")
        return [0]

    key_cols_all = [
        "min_target_R",
        "trail_giveback_R",
        "stop_R",
        "rsi_exit_threshold",
        "rsi_min_R",
        "sr_dist_threshold_atr",
    ]
    key_cols = [c for c in key_cols_all if c in df.columns]

    # Jeśli żadnej z kolumn nie ma – po prostu bierz top max_rules
    if not key_cols:
        idxs = list(df.index[:max_rules])
        print(f"[run_wf] INFO: w {ranked_path} brak kolumn parametrów – wybieram pierwsze {len(idxs)} indeksy.")
        return idxs

    chosen: List[int] = []
    seen_keys = set()

    # df jest posortowany po score_composite w pick_best_exits, więc iterujemy od najlepszych
    for i, row in df.reset_index().iterrows():
        key = tuple(row[c] for c in key_cols)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        chosen.append(int(row["index"]))
        if len(chosen) >= max_rules:
            break

    if not chosen:
        chosen = [0]

    chosen_sorted = sorted(chosen)
    print(f"[run_wf] Wybrane różnorodne reguły z {ranked_path}: {chosen_sorted}")
    return chosen_sorted


# ---------------------------------------------------------------------------
# Główny pipeline dla jednego kroku WF
# ---------------------------------------------------------------------------

def run_wf_step(
    step: WFStep,
    workdir: str,
    candles_5m: str,
    max_bars_ahead: int,
    checkpoint_bars: str,
    cleanup: bool,
) -> None:
    """
    Dla jednego kroku WF:
    - scala TRAIN/TEST snapshots
    - odpala pełny TRAIN pipeline (entry+exit research)
    - odpala TEST pipeline (grouper --only-enrich + strategy_backtest grid po entry/exit)
    - jeśli cleanup=True -> sprząta pliki pośrednie TRAIN,
      jeśli cleanup=False -> nic nie kasuje.
    """
    step_dir = os.path.join(workdir, f"wf_step_{step.step_id:02d}")
    os.makedirs(step_dir, exist_ok=True)

    # 1) Pliki wejściowe / wyjściowe dla tego kroku
    train_files = [p.filename for p in step.train_periods]
    test_files = [p.filename for p in step.test_periods]

    snapshots_train = os.path.join(step_dir, "snapshots_train.csv")
    snapshots_test = os.path.join(step_dir, "snapshots_test.csv")

    # TRAIN – pośrednie
    labeled_snapshots_train = os.path.join(step_dir, "snapshots_train_labeled.csv")
    labeled_paths_train = os.path.join(step_dir, "paths_labeler_train.csv")

    enriched_snapshots_train = os.path.join(step_dir, "snapshots_enriched_train.csv")
    entry_setups_train = os.path.join(step_dir, "entry_setups_train.csv")

    paths_v3_train = os.path.join(step_dir, "paths_v3_train.csv")

    ideal_exits_long = os.path.join(step_dir, "ideal_exits_long.csv")
    signatures_long = os.path.join(step_dir, "exit_signatures_long.csv")
    ideal_exits_short = os.path.join(step_dir, "ideal_exits_short.csv")
    signatures_short = os.path.join(step_dir, "exit_signatures_short.csv")

    entry_setups_long_good = os.path.join(step_dir, "entry_setups_long_good.csv")
    entry_setups_short_good = os.path.join(step_dir, "entry_setups_short_good.csv")

    exit_lab_long = os.path.join(step_dir, "exit_lab_v3_long.csv")
    exit_lab_short = os.path.join(step_dir, "exit_lab_v3_short.csv")

    ranked_long = os.path.join(step_dir, "exit_lab_v3_long_ranked.csv")
    ranked_short = os.path.join(step_dir, "exit_lab_v3_short_ranked.csv")

    # TEST – wzbogacone snapshoty
    snapshots_test_enriched = os.path.join(step_dir, "snapshots_test_enriched.csv")

    # 2) Merge plików TRAIN / TEST (merge_files.py)
    merge_inputs_train = []
    for f in train_files:
        merge_inputs_train.extend(["--inputs", f])

    merge_inputs_test = []
    for f in test_files:
        merge_inputs_test.extend(["--inputs", f])

    run_cmd_if_missing(
        ["python", "merge_files.py", *merge_inputs_train, "--output", snapshots_train],
        outputs=snapshots_train,
    )
    run_cmd_if_missing(
        ["python", "merge_files.py", *merge_inputs_test, "--output", snapshots_test],
        outputs=snapshots_test,
    )

    # 3) LABELER – tylko na TRAIN
    run_cmd_if_missing(
        [
            "python",
            "labeler_v2_entry_only.py",
            "--candles",
            candles_5m,
            "--snapshots",
            snapshots_train,
            "--out-snapshots",
            labeled_snapshots_train,
            "--out-paths",
            labeled_paths_train,
            "--max-bars-ahead",
            str(max_bars_ahead),
            "--checkpoint-bars",
            checkpoint_bars,
        ],
        outputs=[labeled_snapshots_train, labeled_paths_train],
    )

    # 4) GROUPER – TRAIN (pełny: features + clusters + setupy)
    run_cmd_if_missing(
        [
            "python",
            "grouper.py",
            "--snapshots",
            labeled_snapshots_train,
            "--out-setups",
            entry_setups_train,
            "--out-snapshots",
            enriched_snapshots_train,
            "--min-trades-per-group",
            str(DEFAULT_MIN_TRADES_PER_GROUP),
            "--cluster-gap-bars",
            str(DEFAULT_CLUSTER_GAP_BARS),
        ],
        outputs=[entry_setups_train, enriched_snapshots_train],
    )

    # 5) PATH GENERATOR – TRAIN
    run_cmd_if_missing(
        [
            "python",
            "path_generator_v3.py",
            "--snapshots-enriched",
            enriched_snapshots_train,
            "--out-paths",
            paths_v3_train,
            "--max-bars-ahead",
            str(max_bars_ahead),
        ],
        outputs=paths_v3_train,
    )

    # 6) EXIT SIGNATURES – long
    run_cmd_if_missing(
        [
            "python",
            "exit_signatures.py",
            "--paths-v3",
            paths_v3_train,
            "--side",
            "long",
            "--max-bars",
            str(max_bars_ahead),
            "--min-ideal-R",
            str(DEFAULT_MIN_IDEAL_R),
            "--min-exits-per-group",
            str(DEFAULT_MIN_EXITS_PER_GROUP),
            "--out-ideal-exits",
            ideal_exits_long,
            "--out-signatures",
            signatures_long,
        ],
        outputs=[ideal_exits_long, signatures_long],
    )

    # 7) EXIT SIGNATURES – short
    run_cmd_if_missing(
        [
            "python",
            "exit_signatures.py",
            "--paths-v3",
            paths_v3_train,
            "--side",
            "short",
            "--max-bars",
            str(max_bars_ahead),
            "--min-ideal-R",
            str(DEFAULT_MIN_IDEAL_R),
            "--min-exits-per-group",
            str(DEFAULT_MIN_EXITS_PER_GROUP),
            "--out-ideal-exits",
            ideal_exits_short,
            "--out-signatures",
            signatures_short,
        ],
        outputs=[ideal_exits_short, signatures_short],
    )

    # 8) FILTER ENTRY SETUPS (TRAIN)
    fa = DEFAULT_FILTER_ARGS
    run_cmd_if_missing(
        [
            "python",
            "filter_entry_setups.py",
            "--entry-setups",
            entry_setups_train,
            "--min-setup-trades",
            str(fa["min_setup_trades"]),
            "--min-setup-clusters",
            str(fa["min_setup_clusters"]),
            "--long-min-entry-score",
            str(fa["long_min_entry_score"]),
            "--long-min-avg-max-R",
            str(fa["long_min_avg_max_R"]),
            "--long-min-pct-ge-2R",
            str(fa["long_min_pct_ge_2R"]),
            "--long-max-pct-min-le-1R",
            str(fa["long_max_pct_min_le_1R"]),
            "--top-long",
            str(fa["top_long"]),
            "--short-min-entry-score",
            str(fa["short_min_entry_score"]),
            "--short-min-avg-max-R",
            str(fa["short_min_avg_max_R"]),
            "--short-min-pct-ge-2R",
            str(fa["short_min_pct_ge_2R"]),
            "--short-max-pct-min-le-1R",
            str(fa["short_max_pct_min_le_1R"]),
            "--top-short",
            str(fa["top_short"]),
            "--out-long",
            entry_setups_long_good,
            "--out-short",
            entry_setups_short_good,
        ],
        outputs=[entry_setups_long_good, entry_setups_short_good],
    )

    # 9) EXIT LAB – SHORT
    ela = DEFAULT_EXIT_LAB_ARGS
    run_cmd_if_missing(
        [
            "python",
            "exit_lab.py",
            "--paths-v3",
            paths_v3_train,
            "--snapshots-enriched",
            enriched_snapshots_train,
            "--entry-setups",
            entry_setups_short_good,
            "--signatures",
            signatures_short,
            "--side",
            "short",
            "--min-entry-score",
            str(ela["min_entry_score"]),
            "--min-clusters",
            str(ela["min_clusters"]),
            "--min-target-R",
            ela["min_target_R"],
            "--trail-giveback-R",
            ela["trail_giveback_R"],
            "--stop-R",
            ela["stop_R"],
            "--max-bars",
            ela["max_bars"],
            "--rsi-exit-threshold",
            ela["rsi_exit_threshold"],
            "--rsi-min-R",
            ela["rsi_min_R"],
            f"--sr-dist-threshold-atr={ela['sr_dist_threshold_atr']}",
            "--sig-min-n-exits",
            str(ela["sig_min_n_exits"]),
            "--sig-min-avg-R-exit",
            str(ela["sig_min_avg_R_exit"]),
            "--sig-R-buckets",
            ela["sig_R_buckets"],
            "--out-results",
            exit_lab_short,
        ],
        outputs=exit_lab_short,
    )

    # 10) EXIT LAB – LONG
    run_cmd_if_missing(
        [
            "python",
            "exit_lab.py",
            "--paths-v3",
            paths_v3_train,
            "--snapshots-enriched",
            enriched_snapshots_train,
            "--entry-setups",
            entry_setups_long_good,
            "--signatures",
            signatures_long,
            "--side",
            "long",
            "--min-entry-score",
            str(ela["min_entry_score"]),
            "--min-clusters",
            str(ela["min_clusters"]),
            "--min-target-R",
            ela["min_target_R"],
            "--trail-giveback-R",
            ela["trail_giveback_R"],
            "--stop-R",
            ela["stop_R"],
            "--max-bars",
            ela["max_bars"],
            "--rsi-exit-threshold",
            ela["rsi_exit_threshold"],
            "--rsi-min-R",
            ela["rsi_min_R"],
            f"--sr-dist-threshold-atr={ela['sr_dist_threshold_atr']}",
            "--sig-min-n-exits",
            str(ela["sig_min_n_exits"]),
            "--sig-min-avg-R-exit",
            str(ela["sig_min_avg_R_exit"]),
            "--sig-R-buckets",
            ela["sig_R_buckets"],
            "--out-results",
            exit_lab_long,
        ],
        outputs=exit_lab_long,
    )

    # 11) PICK BEST EXITS – generuje exit_lab_v3_long_ranked.csv / short_ranked.csv w step_dir
    pa = DEFAULT_PICK_ARGS
    run_cmd_if_missing(
        [
            "python",
            "pick_best_exits.py",
            "--long-results",
            exit_lab_long,
            "--short-results",
            exit_lab_short,
            "--out-long",
            ranked_long,
            "--out-short",
            ranked_short,
            "--min-n-trades",
            str(pa["min_n_trades"]),
            "--max-dd-abs",
            str(pa["max_dd_abs"]),
            "--min-avg-R",
            str(pa["min_avg_R"]),
            "--min-winrate",
            str(pa["min_winrate"]),
            "--min-sum-R",
            str(pa["min_sum_R"]),
            "--top-k",
            str(pa["top_k"]),
        ],
        outputs=[ranked_long, ranked_short],
    )

    print(f"[run_wf] Krok {step.step_id:02d} – TRAIN pipeline zakończony.")
    print(f"[run_wf]  - snapshots_train: {snapshots_train}")
    print(f"[run_wf]  - snapshots_test:  {snapshots_test}")
    print(f"[run_wf]  - exit_lab long:   {exit_lab_long}")
    print(f"[run_wf]  - exit_lab short:  {exit_lab_short}")
    print(f"[run_wf]  - entry_setups_long_good:  {entry_setups_long_good}")
    print(f"[run_wf]  - entry_setups_short_good: {entry_setups_short_good}")
    print(f"[run_wf]  - ranked rules long:  {ranked_long}")
    print(f"[run_wf]  - ranked rules short: {ranked_short}")

    # 12) TEST: wzbogacenie snapshotów testowych (tylko featury + klastry, bez setupów)
    run_cmd_if_missing(
        [
            "python",
            "grouper.py",
            "--snapshots",
            snapshots_test,
            "--out-snapshots",
            snapshots_test_enriched,
            "--cluster-gap-bars",
            str(DEFAULT_CLUSTER_GAP_BARS),
            "--only-enrich",
        ],
        outputs=snapshots_test_enriched,
    )

    # 13) TEST: grid entry + exit
    exit_signatures_long = signatures_long
    exit_signatures_short = signatures_short

    long_rule_indices = select_diverse_rule_indices(ranked_long, max_rules=3)
    short_rule_indices = select_diverse_rule_indices(ranked_short, max_rules=3)

    # zbuduj warianty ENTRY: all_good + top3/top5/top10
    entry_variants = []
    try:
        df_long = pd.read_csv(entry_setups_long_good, sep=";")
        df_short = pd.read_csv(entry_setups_short_good, sep=";")
    except Exception as e:
        print(f"[run_wf] WARN: nie mogę wczytać entry_setups_*_good: {e} – używam tylko pełnych plików.")
        df_long = None
        df_short = None

    variant_idx = 0
    # wariant 0: pełna lista "good"
    entry_variants.append(
        (variant_idx, entry_setups_long_good, entry_setups_short_good)
    )
    variant_idx += 1

    if df_long is not None and df_short is not None and not df_long.empty and not df_short.empty:
        Ns = [3, 5, 10]
        for N in Ns:
            nL = min(N, len(df_long))
            nS = min(N, len(df_short))
            if nL <= 0 or nS <= 0:
                continue
            # jeśli to samo co pełna lista – pomijamy
            if nL == len(df_long) and nS == len(df_short):
                continue

            out_long_variant = os.path.join(step_dir, f"entry_setups_long_top{nL}.csv")
            out_short_variant = os.path.join(step_dir, f"entry_setups_short_top{nS}.csv")

            # zapis top-nL/nS
            df_long.head(nL).to_csv(out_long_variant, sep=";", index=False)
            df_short.head(nS).to_csv(out_short_variant, sep=";", index=False)

            entry_variants.append(
                (variant_idx, out_long_variant, out_short_variant)
            )
            print(f"[run_wf] ENTRY wariant {variant_idx}: long top {nL}, short top {nS}")
            variant_idx += 1
    else:
        print("[run_wf] INFO: brak DF dla entry_setups_*_good – zostaje tylko wariant pełny (idx=0).")

    # odpal backtest dla każdego wariantu entry + kombinacji exitów
    for e_idx, entry_long_path, entry_short_path in entry_variants:
        for li in long_rule_indices:
            for si in short_rule_indices:
                trades_csv = os.path.join(step_dir, f"trades_E{e_idx}_L{li}_S{si}.csv")
                trades_summary_csv = os.path.join(step_dir, f"trades_summary_E{e_idx}_L{li}_S{si}.csv")

                run_cmd_if_missing(
                    [
                        "python",
                        "strategy_backtest.py",
                        "--snapshots-enriched",
                        snapshots_test_enriched,
                        "--entry-setups-long",
                        entry_long_path,
                        "--entry-setups-short",
                        entry_short_path,
                        "--ranked-long",
                        ranked_long,
                        "--ranked-short",
                        ranked_short,
                        "--signatures-long",
                        exit_signatures_long,
                        "--signatures-short",
                        exit_signatures_short,
                        "--long-rule-index",
                        str(li),
                        "--short-rule-index",
                        str(si),
                        "--atr-col",
                        DEFAULT_ATR_COL,
                        "--risk-mult",
                        str(DEFAULT_RISK_MULT),
                        "--rsi-col",
                        DEFAULT_RSI_COL,
                        "--out-trades",
                        trades_csv,
                        "--out-summary",
                        trades_summary_csv,
                    ],
                    outputs=[trades_csv, trades_summary_csv],
                )

    # 13b) Combine results dla siatki entry/exit w tym kroku
    combined_summary_csv = os.path.join(step_dir, "trades_summary_grid.csv")
    rows = []
    pattern = os.path.join(step_dir, "trades_summary_E*_L*_S*.csv")
    for path in glob.glob(pattern):
        base = os.path.basename(path)  # np. "trades_summary_E1_L0_S2.csv"
        name = base.replace("trades_summary_", "").replace(".csv", "")  # "E1_L0_S2"
        try:
            part_e, part_l, part_s = name.split("_")
            ei = int(part_e[1:])  # "E1" -> 1
            li = int(part_l[1:])  # "L0" -> 0
            si = int(part_s[1:])  # "S2" -> 2
        except Exception:
            continue

        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            continue
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        row["entry_variant_index"] = ei
        row["long_rule_index"] = li
        row["short_rule_index"] = si
        rows.append(row)

    if rows:
        combined = pd.DataFrame(rows)
        sort_cols = [c for c in ["sum_R", "avg_R"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(
                by=sort_cols,
                ascending=[False] * len(sort_cols),
            ).reset_index(drop=True)
        combined.to_csv(combined_summary_csv, sep=";", index=False)
        print(f"[run_wf]  - combined entry/exit grid summary: {combined_summary_csv}")
    else:
        print("[run_wf]  - brak plików trades_summary_E*_L*_S*.csv do złączenia.")

    # 14) CLEANUP – kasujemy ciężkie pliki pośrednie TRAIN tylko jeśli cleanup=True
    if cleanup:
        to_delete = [
            snapshots_train,
            labeled_snapshots_train,
            labeled_paths_train,
            enriched_snapshots_train,
            paths_v3_train,
            ideal_exits_long,
            ideal_exits_short,
            entry_setups_train,
        ]
        for path in to_delete:
            safe_remove(path)
    else:
        print("[run_wf] CLEANUP wyłączony – żadne pliki pośrednie TRAIN nie zostały skasowane.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master run script for walk-forward research + backtest pipeline."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Katalog z plikami snapshotów (default: .)",
    )
    parser.add_argument(
        "--snapshots-glob",
        "-g",
        type=str,
        default=DEFAULT_SNAPSHOTS_GLOB,
        help=(
            "Wzorzec glob dla plików snapshotów "
            f"(default: {DEFAULT_SNAPSHOTS_GLOB})"
        ),
    )
    parser.add_argument(
        "--candles-5m",
        type=str,
        default=DEFAULT_CANDLES_5M,
        help=f"Plik z M5 candles (default: {DEFAULT_CANDLES_5M})",
    )
    parser.add_argument(
        "--train-len",
        type=int,
        default=DEFAULT_TRAIN_LEN,
        help=f"Liczba okresów w TRAIN (default: {DEFAULT_TRAIN_LEN})",
    )
    parser.add_argument(
        "--test-len",
        type=int,
        default=DEFAULT_TEST_LEN,
        help=f"Liczba okresów w TEST (default: {DEFAULT_TEST_LEN})",
    )
    parser.add_argument(
        "--max-bars-ahead",
        type=int,
        default=DEFAULT_MAX_BARS_AHEAD,
        help=f"max_bars_ahead do labelera / path_generator (default: {DEFAULT_MAX_BARS_AHEAD})",
    )
    parser.add_argument(
        "--checkpoint-bars",
        type=str,
        default=DEFAULT_CHECKPOINT_BARS,
        help=f"checkpoint-bars do labelera (default: '{DEFAULT_CHECKPOINT_BARS}')",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="wf_runs",
        help="Katalog, gdzie będą tworzone podkatalogi wf_step_xx (default: wf_runs)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Jeśli podane: po zakończeniu kroku WF usuń ciężkie pliki pośrednie TRAIN "
            "(snapshots_train, paths_v3_train itd.). Domyślnie NIE sprzątamy."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # 1) Znajdź okresy
    periods = discover_periods(args.directory, args.snapshots_glob)
    if not periods:
        print("[run_wf] Nie znaleziono żadnych plików snapshotów.")
        return

    print("[run_wf] Znalezione okresy:")
    for p in periods:
        print(
            f"  id={p.period_id:2d}  days_back={p.days_back:4d}  skip_days={p.skip_days:4d}  "
            f"start_off={p.start_offset_days:4d}  end_off={p.end_offset_days:4d}  "
            f"file={os.path.basename(p.filename)}"
        )

    # 2) Zbuduj kroki walk-forward
    steps = build_wf_steps(periods, args.train_len, args.test_len)
    print_wf_plan(steps)

    if not steps:
        print("[run_wf] Brak kroków WF – sprawdź train_len/test_len.")
        return

    # 3) Odpal pipeline dla każdego kroku
    for step in steps:
        print(f"\n[run_wf] === KROK WF {step.step_id:02d} ===")
        run_wf_step(
            step=step,
            workdir=args.workdir,
            candles_5m=args.candles_5m,
            max_bars_ahead=args.max_bars_ahead,
            checkpoint_bars=args.checkpoint_bars,
            cleanup=args.cleanup,
        )

    print("\n[run_wf] Wszystkie kroki WF (TRAIN + TEST pipeline) zakończone.")

    # ------------------------------------------------------------------
    # GLOBAL COMBINE: łączymy wszystkie wf_step_xx/trades_summary_grid.csv
    # w jeden plik: wf_trades_summary_global.csv
    # ------------------------------------------------------------------
    global_rows = []
    pattern = os.path.join(args.workdir, "wf_step_*", "trades_summary_grid.csv")
    for path in glob.glob(pattern):
        # wyciągnij step_id z nazwy folderu (wf_step_XX)
        step_folder = os.path.basename(os.path.dirname(path))
        m = re.match(r"wf_step_(\d+)", step_folder)
        if not m:
            continue
        step_id = int(m.group(1))

        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            continue

        if df.empty:
            continue

        df = df.copy()
        df["step_id"] = step_id
        global_rows.append(df)

    if not global_rows:
        print("[run_wf] UWAGA: nie znaleziono żadnych trades_summary_grid.csv do globalnego złączenia.")
        return

    global_df = pd.concat(global_rows, ignore_index=True)

    # Spodziewane kolumny: n_trades, avg_R, sum_R, winrate, max_DD_R, entry_variant_index, long_rule_index, short_rule_index, ...
    required = [
        "n_trades",
        "sum_R",
        "avg_R",
        "winrate",
        "max_DD_R",
        "entry_variant_index",
        "long_rule_index",
        "short_rule_index",
    ]
    for c in required:
        if c not in global_df.columns:
            print(f"[run_wf] WARN: global_df nie ma kolumny {c} – globalny combine może być niepełny.")

    grouped = global_df.groupby(
        ["entry_variant_index", "long_rule_index", "short_rule_index"],
        as_index=False,
    )

    agg_rows = []
    for (ei, li, si), g in grouped:
        # total trades + total R
        n_trades_total = g["n_trades"].sum() if "n_trades" in g else float("nan")
        sum_R_total = g["sum_R"].sum() if "sum_R" in g else float("nan")

        if n_trades_total and n_trades_total != 0:
            global_avg_R = sum_R_total / n_trades_total
            # ważona winrate po liczbie trade'ów
            if "winrate" in g and "n_trades" in g:
                winrate_weighted = (g["winrate"] * g["n_trades"]).sum() / n_trades_total
            else:
                winrate_weighted = float("nan")
        else:
            global_avg_R = float("nan")
            winrate_weighted = float("nan")

        # globalny DD – bierzemy najgorszy (najbardziej ujemny) max_DD_R ze wszystkich kroków
        if "max_DD_R" in g:
            max_DD_global = g["max_DD_R"].min()
        else:
            max_DD_global = float("nan")

        steps_total = len(g)
        steps_pos = (g["sum_R"] > 0).sum() if "sum_R" in g else float("nan")
        steps_neg = (g["sum_R"] <= 0).sum() if "sum_R" in g else float("nan")

        agg_rows.append(
            {
                "entry_variant_index": ei,
                "long_rule_index": li,
                "short_rule_index": si,
                "steps_total": steps_total,
                "steps_pos": steps_pos,
                "steps_neg": steps_neg,
                "n_trades_total": n_trades_total,
                "sum_R_total": sum_R_total,
                "global_avg_R": global_avg_R,
                "winrate_weighted": winrate_weighted,
                "max_DD_R_global": max_DD_global,
            }
        )

    global_summary = pd.DataFrame(agg_rows)
    if not global_summary.empty:
        global_summary = global_summary.sort_values(
            by=["sum_R_total", "global_avg_R"],
            ascending=[False, False],
        ).reset_index(drop=True)

        out_global = os.path.join(args.workdir, "wf_trades_summary_global.csv")
        global_summary.to_csv(out_global, sep=";", index=False)
        print(f"[run_wf] Globalny wynik WF zapisany do: {out_global}")
        print("[run_wf] TOP kombinacje ENTRY/EXIT (globalnie):")
        print(global_summary.head(10).to_string(index=False))

        # [NOWE] – odpalenie pełnego backtestu na całej próbce dla TOP kombinacji
        try:
            run_cmd(
                [
                    "python",
                    "full_backtest_from_wf.py",
                    "--wf-workdir",
                    args.workdir,
                    "--directory",
                    args.directory,
                    "--snapshots-glob",
                    args.snapshots_glob,
                    "--top-k-full",
                    "5",   # ile najlepszych kombinacji z WF testujemy na full sample
                    "--min-trades-total",
                    "40",  # minimalna liczba trade'ów w WF, żeby kombinacja weszła do full BT
                    "--atr-col",
                    DEFAULT_ATR_COL,
                    "--rsi-col",
                    DEFAULT_RSI_COL,
                    "--risk-mult",
                    str(DEFAULT_RISK_MULT),
                ]
            )
        except Exception as e:
            print(f"[run_wf] UWAGA: full_backtest_from_wf.py nie udał się: {e}")

    else:
        print("[run_wf] UWAGA: global_summary jest pusty – coś poszło nie tak z agregacją.")


if __name__ == "__main__":
    main()
