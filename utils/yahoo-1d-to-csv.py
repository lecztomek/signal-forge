import re
import csv
from datetime import datetime

INPUT = "yahoo-brent-1d-table.txt"              # tu wklejasz całą wiadomość/tabelę
OUTPUT = "bz_1d_for_dynamodb.csv"
PK_VALUE = "BZ=F#1d"

row_pattern = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>(?P<date>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<open>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<high>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<low>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<close>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<adjclose>[^<]*)</td>\s*"
    r"<td[^>]*>(?P<volume>[^<]*)</td>\s*"
    r"</tr>",
    re.IGNORECASE
)

def parse_date_to_iso(date_str: str) -> str:
    # "Nov 14, 2025" -> "2025-11-14T00:00:00Z"
    dt = datetime.strptime(date_str.strip(), "%b %d, %Y")
    return dt.strftime("%Y-%m-%dT00:00:00Z")

def parse_float(s: str) -> float:
    return float(s.strip())

def parse_volume(vol_str: str) -> int:
    vol_str = vol_str.strip()
    if vol_str == "-" or vol_str == "":
        return 0
    return int(vol_str.replace(",", ""))

rows = []

with open(INPUT, encoding="utf-8") as f:
    html = f.read()

for m in row_pattern.finditer(html):
    date = m.group("date")
    ts = parse_date_to_iso(date)
    open_ = parse_float(m.group("open"))
    high = parse_float(m.group("high"))
    low = parse_float(m.group("low"))
    close = parse_float(m.group("close"))
    volume = parse_volume(m.group("volume"))

    rows.append({
        "pk": PK_VALUE,
        "ts": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

# zapisz do CSV
with open(OUTPUT, "w", newline="", encoding="utf-8") as fout:
    fieldnames = ["pk", "ts", "open", "high", "low", "close", "volume"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Zapisano {len(rows)} rekordów do {OUTPUT}")
