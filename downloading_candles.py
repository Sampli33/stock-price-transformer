"""
Загружает дневные OHLCV-свечи (2010-2019) для 86 тикеров США из 11 секторов
через yfinance и сохраняет их в CSV-файл.

Выход: CSV со столбцами begin, ticker, open, close, high, low, volume, value.
"""

import yfinance as yf
import pandas as pd

DATE_FROM = "2010-01-01"
DATE_TO   = "2019-12-31"
MIN_ROWS  = 1800
OUTPUT    = "all_tickers_candles_2010_2019.csv"

TICKERS = [
    # Технологии
    "AAPL", "MSFT", "GOOGL", "INTC", "IBM", "CSCO",
    "TXN", "QCOM", "ADI", "AMAT", "KLAC", "HPQ", "ADP",

    # Финансы
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C",
    "USB", "PNC", "AXP", "COF", "AON",

    # Здравоохранение
    "JNJ", "PFE", "MRK", "ABT", "UNH", "MDT",
    "TMO", "BMY", "AMGN", "GILD",

    # Потребительские товары
    "PG", "KO", "PEP", "WMT", "COST", "CL",
    "GIS", "HSY", "SYY",

    # Промышленность
    "GE", "HON", "CAT", "MMM", "BA", "LMT",
    "NOC", "DE", "EMR",

    # Энергетика
    "XOM", "CVX", "COP", "SLB", "EOG", "VLO",

    # Телекоммуникации
    "VZ", "T", "CMCSA",

    # Материалы
    "LIN", "NEM", "APD", "ECL",

    # Коммунальные услуги
    "NEE", "DUK", "SO", "D",

    # Недвижимость
    "PLD", "SPG",

    # Дискреционный потребительский сектор
    "MCD", "NKE", "TGT", "LOW", "HD", "YUM", "VFC", "HAS",
]

all_parts = []
skipped   = []

for ticker in TICKERS:
    print(f"{ticker:<6} ...", end=" ", flush=True)

    try:
        raw = yf.download(
            ticker,
            start=DATE_FROM,
            end=DATE_TO,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

    except Exception as e:
        print(f"Ошибка: {e}")
        skipped.append(ticker)
        continue

    if raw.empty or len(raw) < MIN_ROWS:
        print(f"пропущен (только {len(raw)} строк)")
        skipped.append(ticker)
        continue

    raw = raw.copy()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df_t = pd.DataFrame({
        "begin": raw.index.strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "open": raw["Open"],
        "close": raw["Close"],
        "high": raw["High"],
        "low": raw["Low"],
        "volume": raw["Volume"],
        "value": (raw["Close"] * raw["Volume"]),
    })

    print(f"OK - {len(df_t)} свечей ({df_t['begin'].iloc[0][:10]} - {df_t['begin'].iloc[-1][:10]})")
    all_parts.append(df_t)

if not all_parts:
    print("Ничего не загружено")
else:
    result = (
        pd.concat(all_parts, ignore_index=True)
        .sort_values(["ticker", "begin"])
        .reset_index(drop=True)
    )
    result.to_csv(OUTPUT, index=False)

    print(f"Тикеров загружено : {len(all_parts)}")
    if skipped:
        print(f"Пропущено : {skipped}")
    print(f"Всего строк : {len(result):,}")
    print(f"Диапазон дат : {result['begin'].min()[:10]} — {result['begin'].max()[:10]}")
    print(f"Сохранено в : {OUTPUT}")
