import pandas as pd

# === USTAWIENIA ===
INPUT_FILE = "brent_5m.csv"   # tu wpisz nazwę swojego pliku 5m
OUTPUT_FILE = "brent_1h.csv"  # nazwa pliku wyjściowego 1h

# === WCZYTANIE DANYCH 5m ===
# Format: Data;Czas;Open;High;Low;Close;Volume
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

# Upewniamy się, że wartości liczbowe są liczbami
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

# === RESAMPLE DO 1H ===
# Open  = pierwsza wartość z godziny
# High  = max z godziny
# Low   = min z godziny
# Close = ostatnia wartość z godziny
# Volume= suma z godziny
df_1h = df.resample('1H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Usuwamy godziny bez żadnych danych (Open = NaN)
df_1h = df_1h.dropna(subset=['Open'])

# === FORMATOWANIE WYJŚCIA ===
df_1h = df_1h.reset_index()

df_1h['Date'] = df_1h['DateTime'].dt.strftime('%d/%m/%Y')
df_1h['Time'] = df_1h['DateTime'].dt.strftime('%H:%M:%S')

# Kolejność kolumn jak w wejściu
df_1h = df_1h[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Zapis do CSV (bez nagłówka, średnik jako separator)
df_1h.to_csv(OUTPUT_FILE, sep=';', index=False, header=False)

print(f"Zapisano dane 1h do pliku: {OUTPUT_FILE}")
