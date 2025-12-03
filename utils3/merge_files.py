# merge_snapshots.py
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="lista plików snapshotów do sklejenia")
    parser.add_argument("--output", required=True, help="plik wynikowy")
    args = parser.parse_args()

    all_rows = []
    fieldnames = None

    for path in args.inputs:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=';')
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    # sortowanie po run_at (i ewentualnie instrument, idx itd.)
    all_rows.sort(key=lambda r: (r.get("run_at"), r.get("instrument")))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Zapisano {len(all_rows)} wierszy do {args.output}")

if __name__ == "__main__":
    main()
