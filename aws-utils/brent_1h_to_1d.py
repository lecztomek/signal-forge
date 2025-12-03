import sys
import pandas as pd

# Użycie:
# python brent_1h_to_1d.py input.csv output.csv
# Jeśli nie podasz argumentów, użyje domyślnych nazw
if len(sys.argv) >= 2:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = "brent_1h.csv"

if len(sys.argv) >= 3:
    OUTPUT_FILE = sys.argv[2]
else:
    OUTPUT_FILE = "brent_1d.csv"

print(f"Czytam dane 1h z: {INPUT_FILE}")

# Format: Date;Time;Open;High;Low;Close;Volume
df = pd.read_csv(
    INPUT_FILE,
    sep=';',
    header=None,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
    parse_dates={'DateTime': ['Date', 'Time']},
    dayfirst=True
)

# Ustawiamy DateTime jako indeks
df.set_index('DateTime', inplace=True)

# Konwersja na typy liczbowe
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

# === RESAMPLE DO 1D ===
# Open  = pierwsza świeca dnia
# High  = max z dnia
# Low   = min z dnia
# Close = ostatnia świeca dnia
# Volume= suma wolumenu w dniu
df_1d = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Usuwamy dni bez danych (Open = NaN)
df_1d = df_1d.dropna(subset=['Open'])

# Przywracamy DateTime jako kolumnę
df_1d = df_1d.reset_index()

# Formatowanie daty i czasu
df_1d['Date'] = df_1d['DateTime'].dt.strftime('%d/%m/%Y')
# Godzina po resamplu to 00:00:00 – tak zostawimy
df_1d['Time'] = df_1d['DateTime'].dt.strftime('%H:%M:%S')

# Kolejność kolumn jak w wejściu
df_1d = df_1d[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Zapis do CSV (bez nagłówka, średnik jako separator)
df_1d.to_csv(OUTPUT_FILE, sep=';', index=False, header=False)

print(f"Zapisano dane 1d do pliku: {OUTPUT_FILE}")
