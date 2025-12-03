#!/usr/bin/env python
"""
full_backtest_from_wf.py

Cel:
- Wziąć wyniki globalnego walk-forwarda z:
    wf_workdir/wf_trades_summary_global.csv
  (robione w run_wf.py),
- Wybrać TOP kombinacje (entry_variant_index, long_rule_index, short_rule_index),
- Zbudować "pełny" zbiór snapshotów z całej próbki (all TRAIN+TEST),
- Wzbogacić go grouperem (--only-enrich),
- Dla każdej wybranej kombinacji:
    * odpalić strategy_backtest.py na CAŁEJ próbce
    * zapisać trades + summary
- Zapisać zbiorcze podsumowanie:
    wf_workdir/wf_full_backtests_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpery subprocess + pliki
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    if cwd:
        print(f"[full_bt] RUN (cwd={cwd}): {' '.join(cmd)}")
    else:
        print(f"[full_bt] RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def outputs_exist(outputs) -> bool:
    if isinstance(outputs, str):
        outputs = [outputs]
    flags = []
    for p in outputs:
        if p is None:
            continue
        flags.append(os.path.exists(p))
    return bool(flags) and all(flags)


def run_cmd_if_missing(cmd: List[str], outputs, cwd: Optional[str] = None) -> None:
    if outputs_exist(outputs):
        if isinstance(outputs, str):
            outs_str = outputs
        else:
            outs_str = ", ".join(outputs)
        print(f"[full_bt] SKIP – wyjście już istnieje: {outs_str}")
        return
    run_cmd(cmd, cwd=cwd)


# ---------------------------------------------------------------------------
# Budowa pełnego zestawu snapshotów + enrich
# ---------------------------------------------------------------------------

def build_full_snapshots(
    directory: str,
    snapshots_glob: str,
    wf_workdir: str,
    cluster_gap_bars: int = 5,
) -> str:
    """
    Łączy wszystkie snapshoty z `directory/snapshots_glob` w:
        wf_workdir/snapshots_all.csv
    i odpala:
        grouper.py --only-enrich
    -> wf_workdir/snapshots_all_enriched.csv
    Zwraca ścieżkę do snapshots_all_enriched.csv
    """
    pattern = os.path.join(directory, snapshots_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"[full_bt] Nie znaleziono snapshotów pod wzorcem: {pattern}"
        )

    snapshots_all = os.path.join(wf_workdir, "snapshots_all.csv")
    snapshots_all_enriched = os.path.join(wf_workdir, "snapshots_all_enriched.csv")

    # merge_files.py
    merge_inputs = []
    for f in files:
        merge_inputs.extend(["--inputs", f])

    run_cmd_if_missing(
        ["python", "merge_files.py", *merge_inputs, "--output", snapshots_all],
        outputs=snapshots_all,
    )

    # grouper --only-enrich
    run_cmd_if_missing(
        [
            "python",
            "grouper.py",
            "--snapshots",
            snapshots_all,
            "--out-snapshots",
            snapshots_all_enriched,
            "--cluster-gap-bars",
            str(cluster_gap_bars),
            "--only-enrich",
        ],
        outputs=snapshots_all_enriched,
    )

    return snapshots_all_enriched


# ---------------------------------------------------------------------------
# Wczytanie globalnego WF i wybór TOP kombinacji
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    entry_variant_index: int
    long_rule_index: int
    short_rule_index: int
    n_trades_total: float
    sum_R_total: float
    global_avg_R: float
    max_DD_R_global: float


def load_global_summary(wf_workdir: str) -> pd.DataFrame:
    path = os.path.join(wf_workdir, "wf_trades_summary_global.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[full_bt] Brak pliku {path} – najpierw odpal run_wf.py, "
            "żeby wygenerować globalne podsumowanie."
        )
    df = pd.read_csv(path, sep=";")
    if df.empty:
        raise RuntimeError(f"[full_bt] wf_trades_summary_global.csv jest pusty.")
    return df


def pick_top_combos(
    global_df: pd.DataFrame,
    top_k_full: int,
    min_trades_total: int,
) -> List[Combo]:
    """
    Wybiera TOP kombinacje ENTRY/EXIT do pełnego backtestu.
    Filtrowanie:
    - n_trades_total >= min_trades_total
    - sort po sum_R_total (malejąco), potem global_avg_R
    """
    required_cols = [
        "entry_variant_index",
        "long_rule_index",
        "short_rule_index",
        "n_trades_total",
        "sum_R_total",
        "global_avg_R",
        "max_DD_R_global",
    ]
    for c in required_cols:
        if c not in global_df.columns:
            raise RuntimeError(
                f"[full_bt] global_df nie ma kolumny {c}, "
                "nie mogę wybrać TOP kombinacji."
            )

    df = global_df.copy()
    df = df[df["n_trades_total"] >= min_trades_total]
    if df.empty:
        raise RuntimeError(
            f"[full_bt] Po filtrze n_trades_total >= {min_trades_total} nie zostało żadnych kombinacji."
        )

    df = df.sort_values(
        by=["sum_R_total", "global_avg_R"],
        ascending=[False, False],
    ).reset_index(drop=True)

    df_top = df.head(top_k_full).copy()
    combos: List[Combo] = []
    for _, row in df_top.iterrows():
        combos.append(
            Combo(
                entry_variant_index=int(row["entry_variant_index"]),
                long_rule_index=int(row["long_rule_index"]),
                short_rule_index=int(row["short_rule_index"]),
                n_trades_total=float(row["n_trades_total"]),
                sum_R_total=float(row["sum_R_total"]),
                global_avg_R=float(row["global_avg_R"]),
                max_DD_R_global=float(row["max_DD_R_global"]),
            )
        )

    print("[full_bt] Wybrane kombinacje do pełnego backtestu:")
    print(df_top[
        [
            "entry_variant_index",
            "long_rule_index",
            "short_rule_index",
            "n_trades_total",
            "sum_R_total",
            "global_avg_R",
            "max_DD_R_global",
        ]
    ].to_string(index=False))

    return combos


# ---------------------------------------------------------------------------
# Wybór referencyjnego kroku WF
# ---------------------------------------------------------------------------

def choose_ref_step_dir(wf_workdir: str) -> str:
    """
    Szuka katalogów wf_step_XX i bierze ostatni (najbardziej "aktualny")
    jako referencyjny do:
    - entry_setups_long_good/short_good
    - exit_lab_v3_*_ranked.csv
    - exit_signatures_*.csv
    """
    pattern = os.path.join(wf_workdir, "wf_step_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise RuntimeError(
            f"[full_bt] Nie znaleziono żadnego wf_step_* w {wf_workdir}"
        )
    ref = dirs[-1]
    print(f"[full_bt] Używam ostatniego kroku WF jako referencyjnego: {ref}")
    return ref


# ---------------------------------------------------------------------------
# Odpalanie pełnych backtestów po wybranych kombinacjach
# ---------------------------------------------------------------------------

def run_full_backtests(
    wf_workdir: str,
    snapshots_all_enriched: str,
    combos: List[Combo],
    atr_col: str,
    rsi_col: str,
    risk_mult: float,
) -> pd.DataFrame:
    """
    Dla każdej kombinacji (entry_variant_index, long_rule_index, short_rule_index):
    - używa plików z ostatniego wf_step_XX jako referencji:
        entry_setups_long_good.csv
        entry_setups_short_good.csv
        exit_lab_v3_long_ranked.csv
        exit_lab_v3_short_ranked.csv
        exit_signatures_long.csv
        exit_signatures_short.csv
    - odpala strategy_backtest.py na snapshots_all_enriched
    - zbiera wyniki summary w jedną tabelę
    """
    ref_dir = choose_ref_step_dir(wf_workdir)

    entry_setups_long_good = os.path.join(ref_dir, "entry_setups_long_good.csv")
    entry_setups_short_good = os.path.join(ref_dir, "entry_setups_short_good.csv")
    ranked_long = os.path.join(ref_dir, "exit_lab_v3_long_ranked.csv")
    ranked_short = os.path.join(ref_dir, "exit_lab_v3_short_ranked.csv")
    signatures_long = os.path.join(ref_dir, "exit_signatures_long.csv")
    signatures_short = os.path.join(ref_dir, "exit_signatures_short.csv")

    for p in [
        entry_setups_long_good,
        entry_setups_short_good,
        ranked_long,
        ranked_short,
        signatures_long,
        signatures_short,
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[full_bt] Brak wymaganego pliku w {ref_dir}: {p}"
            )

    rows = []

    for combo in combos:
        ei = combo.entry_variant_index
        li = combo.long_rule_index
        si = combo.short_rule_index

        # NOTE:
        # Na razie NIE odtwarzamy różnych wariantów entry (top3/top5/top10)
        # na całej próbce – używamy tych samych entry_setups_*_good
        # co w ref-step. entry_variant_index zachowujemy tylko jako "znacznik"
        # w wynikach i w nazwie pliku.
        entry_long_path = entry_setups_long_good
        entry_short_path = entry_setups_short_good

        out_trades = os.path.join(
            wf_workdir, f"full_bt_entry{ei}_L{li}_S{si}_trades.csv"
        )
        out_summary = os.path.join(
            wf_workdir, f"full_bt_entry{ei}_L{li}_S{si}_summary.csv"
        )

        run_cmd_if_missing(
            [
                "python",
                "strategy_backtest.py",
                "--snapshots-enriched",
                snapshots_all_enriched,
                "--entry-setups-long",
                entry_long_path,
                "--entry-setups-short",
                entry_short_path,
                "--ranked-long",
                ranked_long,
                "--ranked-short",
                ranked_short,
                "--signatures-long",
                signatures_long,
                "--signatures-short",
                signatures_short,
                "--long-rule-index",
                str(li),
                "--short-rule-index",
                str(si),
                "--atr-col",
                atr_col,
                "--risk-mult",
                str(risk_mult),
                "--rsi-col",
                rsi_col,
                "--out-trades",
                out_trades,
                "--out-summary",
                out_summary,
            ],
            outputs=[out_trades, out_summary],
        )

        # wczytaj summary i dodaj info o kombinacji
        if not os.path.exists(out_summary):
            print(f"[full_bt] WARN: {out_summary} nie powstał, pomijam.")
            continue

        try:
            df = pd.read_csv(out_summary, sep=";")
        except Exception as e:
            print(f"[full_bt] WARN: nie mogę wczytać {out_summary}: {e}")
            continue

        if df.empty:
            print(f"[full_bt] WARN: {out_summary} jest pusty.")
            continue

        r = df.iloc[0].to_dict()
        r["entry_variant_index"] = ei
        r["long_rule_index"] = li
        r["short_rule_index"] = si
        # dorzuć info z WF (opcjonalnie)
        r["wf_n_trades_total"] = combo.n_trades_total
        r["wf_sum_R_total"] = combo.sum_R_total
        r["wf_global_avg_R"] = combo.global_avg_R
        r["wf_max_DD_R_global"] = combo.max_DD_R_global

        rows.append(r)

    if not rows:
        raise RuntimeError("[full_bt] Żaden pełny backtest nie wygenerował wyników.")

    combined = pd.DataFrame(rows)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Uruchamia pełne backtesty na całej próbce "
            "dla TOP kombinacji ENTRY/EXIT z walk-forwarda."
        )
    )
    p.add_argument(
        "--wf-workdir",
        type=str,
        default="wf_runs",
        help="Katalog z wynikami walk-forward (default: wf_runs).",
    )
    p.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Katalog z plikami snapshotów źródłowych (default: .).",
    )
    p.add_argument(
        "--snapshots-glob",
        type=str,
        default="brent_snapshots_5m_100_*.csv",
        help="Glob dla plików snapshotów do zbudowania pełnej próbki.",
    )
    p.add_argument(
        "--top-k-full",
        type=int,
        default=5,
        help="Liczba najlepszych kombinacji ENTRY/EXIT do pełnego backtestu.",
    )
    p.add_argument(
        "--min-trades-total",
        type=int,
        default=40,
        help="Minimalna liczba trade'ów w WF, żeby kombinacja była brana do full backtestu.",
    )
    p.add_argument(
        "--atr-col",
        type=str,
        default="m5_atr14",
        help="Kolumna ATR do strategy_backtest.py.",
    )
    p.add_argument(
        "--rsi-col",
        type=str,
        default="m5_rsi14",
        help="Kolumna RSI do strategy_backtest.py.",
    )
    p.add_argument(
        "--risk-mult",
        type=float,
        default=1.0,
        help="risk_mult do strategy_backtest.py.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.wf_workdir, exist_ok=True)

    # 1) Zbuduj pełny zestaw snapshotów i wzbogac go
    snapshots_all_enriched = build_full_snapshots(
        directory=args.directory,
        snapshots_glob=args.snapshots_glob,
        wf_workdir=args.wf_workdir,
        cluster_gap_bars=5,
    )

    # 2) Wczytaj globalne podsumowanie WF i wybierz TOP kombinacje
    global_df = load_global_summary(args.wf_workdir)
    combos = pick_top_combos(
        global_df=global_df,
        top_k_full=args.top_k_full,
        min_trades_total=args.min_trades_total,
    )

    # 3) Odpal pełne backtesty
    combined = run_full_backtests(
        wf_workdir=args.wf_workdir,
        snapshots_all_enriched=snapshots_all_enriched,
        combos=combos,
        atr_col=args.atr_col,
        rsi_col=args.rsi_col,
        risk_mult=args.risk_mult,
    )

    # 4) Zapisz zbiorcze podsumowanie
    out_path = os.path.join(args.wf_workdir, "wf_full_backtests_summary.csv")
    combined.to_csv(out_path, sep=";", index=False)
    print(f"[full_bt] Zbiorcze podsumowanie pełnych backtestów zapisane do: {out_path}")
    print("[full_bt] TOP wyniki full BT:")
    sort_cols = [c for c in ["sum_R", "avg_R"] if c in combined.columns]
    if sort_cols:
        print(
            combined.sort_values(
                by=sort_cols,
                ascending=[False] * len(sort_cols),
            )
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
