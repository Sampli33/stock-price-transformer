"""
Формирует датасет для обучения модели бинарной классификации направления
цены закрытия (вырастет/упадёт через HORIZON торговых дней) на основе
дневных OHLCV-свечей.

Пайплайн:
    1. Загрузка свечей и расчёт целевой переменной (per-ticker).
    2. Построение технических индикаторов: доходности, SMA-ratio, MACD,
       RSI, полосы Боллинджера, ATR, стохастик, объёмные и свечные признаки,
       циклическое кодирование дня недели/месяца.
    3. Временной train/val/test сплит отдельно по каждому тикеру.
    4. Масштабирование объёма (RobustScaler).
    5. Нарезка на скользящие окна длиной SEQ_LEN для каждого тикера.

Вход: CSV со свечами (DATA_PATH), колонки begin/ticker/open/high/low/close/volume.

Выход:
    DATASET_PATH — pickle с окнами X, метками y, тикерами и датами
                   отдельно для train/val/test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle

DATA_PATH = "all_tickers_candles_2010_2019.csv"
DATASET_PATH = "dataset_ready.pkl"

SEQ_LEN = 60            # длина входного окна (торговых дней)
HORIZON = 1             # горизонт прогноза
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ЗАГРУЗКА И БАЗОВАЯ ОЧИСТКА

df = pd.read_csv(DATA_PATH)
df["ticker"] = df["ticker"].astype(object)
df["begin"]  = pd.to_datetime(df["begin"])
df = df.rename(columns={"begin": "date"})
df = (df.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"])
        .reset_index(drop=True))

print(f"Загружено строк: {len(df)}")
print(f"Тикеры: {sorted(df['ticker'].unique())}")
print(f"Диапазон дат: {df['date'].min().date()} — {df['date'].max().date()}")

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ

df["_fc"] = df.groupby("ticker")["close"].shift(-HORIZON)
df = df.dropna(subset=["_fc"]).reset_index(drop=True)
df["target"] = (df["_fc"] > df["close"]).astype(int)
df = df.drop(columns=["_fc"])

# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (per-ticker)

parts = []
for t in df["ticker"].unique():
    g = df[df["ticker"] == t].copy().reset_index(drop=True)
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]

    # Доходности
    g["log_ret_1d"] = np.log(c / (c.shift(1) + 1e-9))
    g["log_ret_5d"] = np.log(c / (c.shift(5) + 1e-9))
    g["log_ret_20d"] = np.log(c / (c.shift(20) + 1e-9))

    # SMA ratio
    for w in [5, 10, 20, 50]:
        sma = c.rolling(w).mean()
        g[f"sma_{w}_ratio"] = (c - sma) / sma

    # MACD (только гистограмма)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    g["macd_hist"] = (macd - signal) / (c + 1e-9)

    # RSI (14)
    c_1 = c.shift(1)
    d = np.log(c_1 / (c_1.shift(1) + 1e-9))
    gain = d.clip(lower=0)
    loss = (-d.clip(upper=0))
    alpha = 1 / 14
    gain_ema = gain.ewm(alpha=alpha, adjust=False).mean()
    loss_ema = loss.ewm(alpha=alpha, adjust=False).mean()
    rsi = gain_ema / (gain_ema + loss_ema + 1e-9)
    g["rsi_logit"] = np.log((rsi + 1e-9) / (1 - rsi + 1e-9))

    # Полосы Боллинджера
    sma20, std20 = c.shift(1).rolling(20).mean(), c.shift(1).rolling(20).std(ddof=0)
    bb_u, bb_l = sma20 + 2 * std20, sma20 - 2 * std20
    g["bb_pct"] = (c - bb_l) / (bb_u - bb_l + 1e-9)
    g["bb_pct_centered"] = g["bb_pct"] - 0.5
    g["bb_width"] = std20 / (sma20 + 1e-9)
    g = g.drop(columns=["bb_pct"])

    # ATR (14)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    g["atr_14_pct"] = tr.shift(1).rolling(14).mean() / (c + 1e-9)

    # Стохастик (K, D)
    low14, high14 = l.shift(1).rolling(14).min(), h.shift(1).rolling(14).max()
    g["stoch_k"] = ((c - low14) / (high14 - low14 + 1e-9)) - 0.5
    g["stoch_d"] = g["stoch_k"].rolling(3).mean()

    # Объём
    g["volume_log"] = np.log1p(v)
    vol_mean = v.shift(1).rolling(20).mean()
    vol_std = v.shift(1).rolling(20).std()
    g["vol_ratio"] = (v - vol_mean) / (vol_std + 1e-9)

    # Свечные признаки
    o = g["open"].shift(1)
    c_ = c.shift(1)
    h_ = h.shift(1)
    l_ = l.shift(1)
    candle = h_ - l_ + 1e-9
    g["upper_wick"] = (h_ - np.maximum(c_, o)) / candle
    g["lower_wick"] = (np.minimum(c_, o) - l_) / candle
    g["candle_dir"] = (c_ - o) / candle

    parts.append(g)

df = pd.concat(parts, ignore_index=True)

# ВРЕМЕННЫЕ ПРИЗНАКИ

# Синус/косинус-кодирование
_dow   = df["date"].dt.dayofweek
_month = df["date"].dt.month

df["dow_sin"]   = np.sin(2 * np.pi * _dow   / 5)
df["dow_cos"]   = np.cos(2 * np.pi * _dow   / 5)
df["month_sin"] = np.sin(2 * np.pi * _month / 12)
df["month_cos"] = np.cos(2 * np.pi * _month / 12)

# СПИСОК ПРИЗНАКОВ

FEATURE_COLS = [
    "log_ret_1d", "log_ret_5d", "log_ret_20d",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "macd_hist", "rsi_logit",
    "bb_width", "atr_14_pct",
    "stoch_k", "stoch_d",
    "volume_log", "vol_ratio",
    "upper_wick", "lower_wick", "candle_dir",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]
TARGET_COL = "target"

# УДАЛЕНИЕ NaN

print(f"\nДо удаления NaN: {len(df)} строк")

df = pd.concat(
    [g.dropna() for _, g in df.groupby("ticker")],
    ignore_index=True
)
print(f"\nПосле удаления NaN: {len(df)} строк")

# ВРЕМЕННОЙ SPLIT (per-ticker)

def temporal_split(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    n = len(g)
    i_tr = int(n * TRAIN_RATIO)
    i_v = int(n * (TRAIN_RATIO + VAL_RATIO))
    g["split"] = "test"
    g.iloc[:i_tr, g.columns.get_loc("split")] = "train"
    g.iloc[i_tr:i_v, g.columns.get_loc("split")] = "val"
    return g

df = pd.concat(
    [temporal_split(g) for _, g in df.groupby("ticker")],
    ignore_index=True
)

print("\nРаспределение по split:")
print(df.groupby(["ticker", "split"]).size().unstack(fill_value=0))

# МАСШТАБИРОВАНИЕ

scalers = {}
scaled_parts = []
for t, g in df.groupby("ticker"):
    g = g.copy()
    sc = RobustScaler()
    sc.fit(g.loc[g["split"] == "train", ["volume_log"]])
    g["volume_log"] = sc.transform(g[["volume_log"]])
    scalers[t] = sc
    scaled_parts.append(g)
df = (
    pd.concat(scaled_parts)
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

# СКОЛЬЗЯЩИЕ ОКНА

datasets = {
    "train": {"X": [], "y": [], "tickers": [], "dates": []},
    "val": {"X": [], "y": [], "tickers": [], "dates": []},
    "test": {"X": [], "y": [], "tickers": [], "dates": []},
}

for t, g in df.groupby("ticker"):
    g = g.sort_values("date").reset_index(drop=True)

    feats  = g[FEATURE_COLS].values
    target = g[TARGET_COL].values
    split  = g["split"].values
    dates  = g["date"].values

    for i in range(SEQ_LEN, len(g)):

        X = feats[i - SEQ_LEN:i]
        y = target[i]
        s = split[i]
        d = dates[i]

        datasets[s]["X"].append(X)
        datasets[s]["y"].append(y)
        datasets[s]["tickers"].append(t)
        datasets[s]["dates"].append(d)

for split in ["train", "val", "test"]:
    datasets[split]["X"] = np.array(datasets[split]["X"], dtype=np.float32)
    datasets[split]["y"] = np.array(datasets[split]["y"], dtype=np.int64)

    bc = np.bincount(datasets[split]["y"])
    print(
        f"{split:5s}: X={datasets[split]['X'].shape} | "
        f"class 0: {bc[0]} | class 1: {bc[1]}"
    )

with open(DATASET_PATH, "wb") as f:
    pickle.dump(datasets, f)

print(f"Датасет сохранён: {DATASET_PATH}")
