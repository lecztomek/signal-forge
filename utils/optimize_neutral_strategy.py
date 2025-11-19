import argparse
import csv
import json
import copy
import itertools
from typing import List, Dict, Any, Tuple

# Tu importujesz swój kod z poprzedniego pliku
# ZMIEN nazwę modułu na taką, jaką masz u siebie
from run_neutral_strategy_from_snapshots_csv import (
    split_parts_from_row,
    parse_sr_multi_from_row,
    is_neutral_daily,
    check_h1_neutral,
    check_neutral_5m_trigger,
    build_neutral_signal_with_risk,
)


# ============================================================
#              Ładowanie świec i prosty backtest
# ============================================================

def load_candles_5m(candles_csv: str) -> List[Dict[str, Any]]:
    """
    Oczekuje CSV z kolumnami:
        timestamp;open;high;low;close
    Timestamp w tym samym formacie, co w sygnałach (np. 2025-11-06T05:20:00Z).
    """
    candles = []
    with open(candles_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = row.get("timestamp") or row.get("ts") or row.get("time")
            if not ts:
                continue
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
            except Exception:
                continue

            candles.append({
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            })

    candles.sort(key=lambda x: x["timestamp"])
    return candles


def backtest_signals_on_candles(
    signals: List[Dict[str, Any]],
    candles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Bardzo prosty backtest:
      - dla każdego sygnału szuka pierwszej świecy z ts >= signal.timestamp
      - iteruje do przodu aż trafi TP albo SL
      - jeśli w jednej świecy dotknięte są oba, zakładamy PESYMISTYCZNIE, że SL pierwszy
      - liczymy R = zysk / ryzyko (entry - SL)

    To jest uproszczenie – ale wystarczy do względnego porównania configów.
    """

    if not signals:
        return {
            "n_trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "total_R": 0.0,
            "avg_R": 0.0,
        }

    # upewnijmy się, że sygnały są posortowane po czasie
    signals_sorted = sorted(signals, key=lambda s: s["timestamp"])

    trades = []

    # pre-index świec po czasie – liniowe szukanie też by dało radę,
    # ale tak będzie trochę szybciej przy dużych danych
    candle_ts_list = [c["timestamp"] for c in candles]

    def find_first_candle_idx(ts: str) -> int:
        # prosty linear search; jak będzie wolno, można zrobić binary search
        for i, c_ts in enumerate(candle_ts_list):
            if c_ts >= ts:
                return i
        return -1

    for sig in signals_sorted:
        side = sig["side"]
        entry = float(sig["entry_price"])
        sl = float(sig["sl"])
        tp = float(sig["tp"])
        ts = sig["timestamp"]

        start_idx = find_first_candle_idx(ts)
        if start_idx < 0:
            # brak świec po sygnale – ignorujemy
            continue

        exit_price = None
        outcome = None
        bars_held = 0

        for c in candles[start_idx:]:
            bars_held += 1
            high = c["high"]
            low = c["low"]

            if side == "BUY":
                hit_sl = (low <= sl)
                hit_tp = (high >= tp)
            else:
                hit_sl = (high >= sl)
                hit_tp = (low <= tp)

            # Jeśli w tej samej świecy są oba – zakładamy najgorszy case: SL first
            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                outcome = "SL"
                exit_price = sl
                break
            if hit_tp:
                outcome = "TP"
                exit_price = tp
                break

        if outcome is None:
            # brak TP/SL – możesz tu dodać logikę typu: zamknięcie na ostatniej świecy
            continue

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        if side == "BUY":
            R = (exit_price - entry) / risk
        else:
            R = (entry - exit_price) / risk

        trades.append({
            "outcome": outcome,
            "R": R,
            "bars": bars_held,
        })

    if not trades:
        return {
            "n_trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "total_R": 0.0,
            "avg_R": 0.0,
        }

    wins = sum(1 for t in trades if t["outcome"] == "TP")
    losses = sum(1 for t in trades if t["outcome"] == "SL")
    total_R = sum(t["R"] for t in trades)
    avg_R = total_R / len(trades)
    winrate = wins / len(trades) * 100.0

    return {
        "n_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "total_R": total_R,
        "avg_R": avg_R,
    }


# ============================================================
#           Generowanie sygnałów dla danego configu
# ============================================================

def generate_signals_for_config(
    config: Dict[str, Any],
    snapshots_csv: str,
    instrument_override: str | None = None,
) -> List[Dict[str, Any]]:
    """
    To jest mini-wersja Twojego main() bez debug_csv/statystyk.
    Wykorzystuje te same funkcje: is_neutral_daily, check_h1_neutral, check_neutral_5m_trigger, build_neutral_signal_with_risk.
    """

    cfg_daily = config.get("daily", {})
    cfg_h1 = config.get("h1", {})
    cfg_m5 = config.get("m5", {})
    cfg_risk = config.get("risk", {})

    signals: List[Dict[str, Any]] = []

    with open(snapshots_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    for i, row in enumerate(rows):
        snap_instrument = row.get("instrument")
        instrument = instrument_override or snap_instrument or "BZ=F"

        daily_part, h1_part, m5_part = split_parts_from_row(row)
        sr_multi = parse_sr_multi_from_row(row)

        snap_ts = (
            m5_part.get("ts")
            or daily_part.get("ts")
            or row.get("run_at")
            or row.get("ts")
        )

        price = m5_part.get("close")

        # 1) DAILY neutral
        is_neutral, daily_reason = is_neutral_daily(daily_part, cfg_daily)
        if not is_neutral:
            continue

        # 2) H1 neutral
        h1_ok, h1_reason = check_h1_neutral(h1_part, cfg_h1)
        if not h1_ok:
            continue

        # 3) M5 trigger
        raw_signal, m5_debug = check_neutral_5m_trigger(m5_part, cfg_m5)
        if not raw_signal:
            continue

        # 4) Risk / SR
        full_signal = build_neutral_signal_with_risk(
            raw_signal=raw_signal,
            cfg_risk=cfg_risk,
            instrument=instrument,
            strategy_hash="DUMMY",  # nie potrzebujemy tego do optymalizacji
            strategy_name=config.get("name", "neutral_strategy"),
            config_json=json.dumps(config, sort_keys=True),
            sr_multi=sr_multi,
            h1_part=h1_part,
            m5_part=m5_part,
        )

        if not full_signal:
            continue

        # Upewniamy się że mamy timestamp (raw_signal go ma)
        full_signal["timestamp"] = raw_signal["timestamp"]
        signals.append(full_signal)

    return signals


# ============================================================
#           Tworzenie siatki parametrów (grid search)
# ============================================================

def generate_config_variants(
    base_config: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Tutaj definiujesz siatkę parametrów do przetestowania.
    Zwraca listę (config, param_combo_info).
    param_combo_info – słownik z parametrami, żeby łatwo było potem wydrukować.
    """

    # ---- TU TUNINGUJESZ SIATKĘ ----
    # Każdy klucz to ścieżka w configu "sekcja.parametr"
    grid = {
        "daily.max_atr_pct":       [30.0, 40.0, 50.0],
        "daily.rsi_mid_min_pct":   [30.0, 35.0],
        "daily.rsi_mid_max_pct":   [65.0, 70.0],
        "daily.max_ema_distance_pct": [2.0, 3.0],

        "m5.score.min_score":      [25.0, 30.0, 35.0],
        "risk.min_rr":             [1.3, 1.5],
        "risk.min_score":          [65.0, 70.0, 75.0],
    }

    keys = list(grid.keys())
    values_product = list(itertools.product(*(grid[k] for k in keys)))

    variants: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for values in values_product:
        cfg = copy.deepcopy(base_config)
        info = {}
        for key, v in zip(keys, values):
            section, param = key.split(".", 1)
            if section not in cfg:
                cfg[section] = {}
            cfg[section][param] = v
            info[key] = v
        variants.append((cfg, info))

    return variants


# ============================================================
#                           main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_csv", required=True,
                        help="CSV ze snapshotami M5/D1/H1 (jak w Twoim generatorze)")
    parser.add_argument("--candles_csv", required=True,
                        help="CSV ze świecami 5m (timestamp;open;high;low;close)")
    parser.add_argument("--base_config_json", required=True,
                        help="Plik JSON z bazowym configiem strategii")
    parser.add_argument("--instrument", default="BZ=F",
                        help="Instrument (opcjonalnie override)")
    parser.add_argument("--min_trades", type=int, default=20,
                        help="Minimalna liczba tradów, żeby config był brany na poważnie")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Ilu najlepszych configów wydrukować")
    args = parser.parse_args()

    # 1) baza configu
    with open(args.base_config_json, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    # 2) siatka configów
    variants = generate_config_variants(base_config)
    print(f"[INFO] Liczba kombinacji do przetestowania: {len(variants)}")

    # 3) świeczki 5m
    candles = load_candles_5m(args.candles_csv)
    print(f"[INFO] Załadowano {len(candles)} świec z {args.candles_csv}")

    results = []

    for idx, (cfg, info) in enumerate(variants, start=1):
        print(f"\n=== Kombinacja {idx}/{len(variants)} ===")
        print("Parametry:")
        for k, v in info.items():
            print(f"  {k} = {v}")

        # a) generowanie sygnałów
        signals = generate_signals_for_config(
            config=cfg,
            snapshots_csv=args.snapshots_csv,
            instrument_override=args.instrument,
        )
        print(f"[INFO] Wygenerowano sygnałów: {len(signals)}")

        if len(signals) < args.min_trades:
            print("[INFO] Za mało sygnałów – pomijam backtest.")
            continue

        # b) backtest
        stats = backtest_signals_on_candles(signals, candles)
        print(f"[INFO] Backtest: n={stats['n_trades']}, "
              f"winrate={stats['winrate']:.2f}%, "
              f"total_R={stats['total_R']:.3f}, "
              f"avg_R={stats['avg_R']:.3f}")

        # zbieramy
        result_row = {
            "n_trades": stats["n_trades"],
            "winrate": stats["winrate"],
            "total_R": stats["total_R"],
            "avg_R": stats["avg_R"],
            "params": info,
        }
        results.append(result_row)

    if not results:
        print("\n[INFO] Brak configów spełniających min_trades. Zmień siatkę parametrów lub próg.")
        return

    # sortujemy po total_R, potem po avg_R
    results_sorted = sorted(
        results,
        key=lambda r: (r["total_R"], r["avg_R"]),
        reverse=True,
    )

    print("\n========== TOP CONFIGI ==========")
    for i, r in enumerate(results_sorted[: args.top_n], start=1):
        print(f"\n--- #{i} ---")
        print(f"Trades: {r['n_trades']}, winrate={r['winrate']:.2f}%, "
              f"total_R={r['total_R']:.3f}, avg_R={r['avg_R']:.3f}")
        print("Parametry:")
        for k, v in r["params"].items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
