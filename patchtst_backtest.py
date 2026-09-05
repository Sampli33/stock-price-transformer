"""
Оценивает экономическую эффективность обученной модели PatchTST через
бэктест торговых сигналов на тестовой выборке.

Вход:
    PKL_PATH   — dataset_ready.pkl (тестовые окна X/y, тикеры, даты).
    DATA_PATH  — CSV с исходными свечами (нужен для расчёта дневных
                 доходностей, на которых считается PnL стратегии).
    MODEL_PATH — веса обученной модели PatchTST.

Пайплайн:
    1. Инференс модели на test-выборке: предсказанный класс (рост/падение)
       и вероятность роста для каждого (тикер, дата).
    2. Расчёт дневных forward-доходностей по каждому тикеру из сырых цен.
    3. Построение "слоистого" (layered) портфеля: так как модель
       прогнозирует направление на HORIZON дней вперёд, каждая открытая
       позиция держится HORIZON дней, и на каждый день приходится
       несколько параллельных ("слои") позиций, открытых в разные дни —
       усреднение по слоям даёт дневной PnL стратегии. Тестируются 3
       варианта: long-only, long-short и long-short с фильтром по
       уверенности модели (CONF_THRESHOLD).
    4. Equal-Weight Benchmark как бенчмарк (равновзвешенный по всем тикерам с сигналом).
    5. Расчёт стандартных метрик риска/доходности (Sharpe, Sortino, Calmar,
       max drawdown, win rate, profit factor) для всех стратегий.
    6. Визуализация: кривые капитала, просадки, сводная таблица метрик.

Все метрики и PnL — gross, без учёта транзакционных издержек.

Выход: три графика (equity curve, drawdown, таблица метрик), отображаются
на экране.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PKL_PATH = 'dataset_ready.pkl'
DATA_PATH = 'all_tickers_candles_2010_2019.csv'
MODEL_PATH = 'best_model_patchtst.pt'

HORIZON = 5
RISK_FREE = 0.0
TRADING_DAYS = 252

# Позиция открывается только если |вероятность - 0.5| превышает порог,
# т.е. модель должна быть уверена хотя бы на 5 п.п. сильнее случайного угадывания
CONF_THRESHOLD = 0.05

with open(PKL_PATH, 'rb') as f:
    datasets = pickle.load(f)

N_FEATURES = datasets["train"]["X"].shape[2]
SEQ_LEN = datasets["train"]["X"].shape[1]
print(f"Форма входа: (batch, {SEQ_LEN}, {N_FEATURES})")


class PatchTST(nn.Module):
    """Трансформер-классификатор PatchTST для многомерных временных рядов.

    Архитектура идентична использованной при обучении (см. train_model.py):
    """

    def __init__(self,
                 n_features = N_FEATURES,
                 seq_len = SEQ_LEN,
                 patch_len = 12,
                 stride = 6,
                 d_model = 64,
                 nhead = 4,
                 num_layers = 2,
                 dim_ff = 128,
                 dropout = 0.3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.d_model = d_model
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Embedding(self.num_patches, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        head_in = n_features * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Dropout(dropout),
            nn.Linear(head_in, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        _, C, N, P = x.shape
        x = x.reshape(B * C, N, P)
        x = self.patch_proj(x)
        x = x + self.pos_emb(torch.arange(N, device=x.device))
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = x.reshape(B, C * self.d_model)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

model = PatchTST().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

X_test = torch.tensor(datasets["test"]["X"], dtype=torch.float32)
y_test = torch.tensor(datasets["test"]["y"], dtype=torch.long)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=512, shuffle=False)

# Инференс на test: предсказанный класс и вероятность роста
all_preds, all_probs, all_true = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        out   = model(X.to(device))
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(y.numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true = np.array(all_true)
print(f"Тестовых примеров: {len(all_preds)}")

print("\nЗагрузка исходных цен для расчёта дневных доходностей...")
df_prices = pd.read_csv(DATA_PATH)
df_prices["ticker"] = df_prices["ticker"].astype(object)
df_prices["date"]   = pd.to_datetime(df_prices["begin"])
df_prices = (df_prices.sort_values(["ticker", "date"])
                      .drop_duplicates(["ticker", "date"])
                      .reset_index(drop=True))

# Доходность закрытия на 1 день вперёд — базовый "строительный блок" PnL:
# каждый день удержания позиции даёт вклад pos * fwd_ret_1d в этот день.
df_prices["fwd_ret_1d"] = df_prices.groupby("ticker")["close"].transform(
    lambda x: x.shift(-1) / x - 1
)

# Быстрый доступ к forward-доходности по (тикер, дата) при построении PnL
price_map = df_prices.set_index(["ticker", "date"])["fwd_ret_1d"].to_dict()

test_tickers = np.array(datasets["test"]["tickers"])
test_dates = pd.to_datetime(datasets["test"]["dates"])

signals = pd.DataFrame({
    "date": test_dates,
    "ticker": test_tickers,
    "pred": all_preds.astype(int),
    "prob": all_probs,
}).sort_values(["date", "ticker"]).reset_index(drop=True)

print(f"Сигналов всего: {len(signals)}")
print(f"Тикеров в тесте: {signals['ticker'].nunique()}")
print(f"Диапазон тестовых дат: {signals['date'].min().date()} — {signals['date'].max().date()}")

print(f"\nРаспределение предсказаний модели:")
print(f"класс 0: {(all_preds == 0).sum()} ({(all_preds == 0).mean():.1%})")
print(f"класс 1: {(all_preds == 1).sum()} ({(all_preds == 1).mean():.1%})")


def build_layered_strategy(signals, mode="long_short", conf_threshold=None,
                           horizon=HORIZON, price_map=price_map):
    """Строит дневной PnL "слоистой" (layered) стратегии по сигналам модели.

    Модель прогнозирует направление на `horizon` дней вперёд, поэтому
    каждый сигнал (тикер, дата) открывает позицию, которая держится
    `horizon` торговых дней подряд и в каждый из них даёт вклад
    position * дневная_доходность. Поскольку новый сигнал появляется
    каждый день, в любой момент времени одновременно "живут" до `horizon`
    таких перекрывающихся позиций (слоёв) по каждому тикеру — итоговый
    дневной PnL стратегии считается как среднее по всем активным на этот
    день слоям. Такой подход не даёт результату зависеть от того, что
    сигнал выдаётся каждый день при многодневном горизонте прогноза.

    mode:
        "long_only"  — позиция 1 при предсказанном росте, иначе 0.
        "long_short" — позиция 1 при росте, -1 при падении.
    conf_threshold:
        если задан, позиции с |prob - 0.5| <= conf_threshold обнуляются
        (сделка не открывается при низкой уверенности модели).

    Возвращает pd.Series дневных доходностей стратегии, индекс — даты.
    """
    signals = signals.copy()

    if mode == "long_only":
        raw_pos = np.where(signals["pred"] == 1, 1, 0)
    elif mode == "long_short":
        raw_pos = np.where(signals["pred"] == 1, 1, -1)
    else:
        raise ValueError(f"Неизвестный mode: {mode}")

    if conf_threshold is not None:
        confident = np.abs(signals["prob"] - 0.5) > conf_threshold
        raw_pos = np.where(confident, raw_pos, 0)

    signals["position"] = raw_pos

    records = []
    signals_sorted = signals.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Отсортированные торговые даты по каждому тикеру — нужны, чтобы найти
    # следующие `horizon` реальных торговых дней после даты сигнала
    # (календарные дни здесь не подходят из-за выходных/праздников).
    ticker_dates = {t: np.sort(df_prices[df_prices["ticker"] == t]["date"].values)
                    for t in signals["ticker"].unique()}

    for _, row in signals_sorted.iterrows():
        if row["position"] == 0:
            continue

        t   = row["date"]
        tkr = row["ticker"]
        pos = row["position"]

        td  = ticker_dates[tkr]
        idx = np.searchsorted(td, np.datetime64(t))
        if idx >= len(td):
            continue
        # Следующие `horizon` торговых дней — окно удержания этой позиции
        layer_dates = td[idx : idx + horizon]

        for ta in layer_dates:
            ta_ts = pd.Timestamp(ta)
            r1 = price_map.get((tkr, ta_ts), np.nan)
            if np.isnan(r1):
                continue
            records.append({"date": ta_ts, "pnl": pos * r1})

    pnl_df = pd.DataFrame(records)
    if len(pnl_df) == 0:
        return pd.Series(dtype=float, name="ret")

    # Усреднение по всем активным в этот день "слоям" (тикер x момент входа)
    daily = pnl_df.groupby("date")["pnl"].mean().sort_index().rename("ret")
    return daily


def build_buy_hold(signals, price_map=price_map):
    """Бенчмарк Equal-Weight Benchmark: равновзвешенная дневная доходность по всем
    (тикер, дата) из test-сигналов, без использования предсказаний модели.
    """
    records = []
    for _, row in signals[["date", "ticker"]].drop_duplicates().iterrows():
        r = price_map.get((row["ticker"], row["date"]), np.nan)
        if not np.isnan(r):
            records.append({"date": row["date"], "ret": r})
    daily = pd.DataFrame(records).groupby("date")["ret"].mean().sort_index().rename("ret")
    return daily


# --- Стандартные метрики риска/доходности для рядов дневных доходностей ---

def annualized_return(rets, periods=TRADING_DAYS):
    """Геометрическая годовая доходность по ряду дневных доходностей."""
    if len(rets) == 0: return np.nan
    cum = (1 + rets).prod()
    return cum ** (periods / len(rets)) - 1

def annualized_volatility(rets, periods=TRADING_DAYS):
    """Годовая волатильность (std дневных доходностей, приведённая к году)."""
    return rets.std() * np.sqrt(periods)

def sharpe(rets, rf=RISK_FREE, periods=TRADING_DAYS):
    """Коэффициент Шарпа (годовой) относительно безрисковой ставки rf."""
    excess = rets - rf / periods
    if excess.std() == 0 or len(rets) == 0:
        return np.nan
    return np.sqrt(periods) * excess.mean() / excess.std()

def sortino(rets, rf=RISK_FREE, periods=TRADING_DAYS):
    """Коэффициент Сортино — как Шарп, но волатильность считается только
    по отрицательным (downside) отклонениям."""
    excess = rets - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0: return np.nan
    down_std = np.sqrt((downside**2).mean())
    if down_std == 0: return np.nan
    return np.sqrt(periods) * excess.mean() / down_std

def max_drawdown(rets):
    """Максимальная просадка от исторического пика кривой капитала."""
    if len(rets) == 0: return np.nan
    cum = (1 + rets).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def calmar(rets, periods=TRADING_DAYS):
    """Коэффициент Калмара: годовая доходность / |максимальная просадка|."""
    mdd = max_drawdown(rets)
    if mdd == 0 or np.isnan(mdd): return np.nan
    return annualized_return(rets, periods) / abs(mdd)

def win_rate(rets):
    """Доля дней с положительным PnL среди дней с ненулевой позицией."""
    active = rets[rets != 0]
    if len(active) == 0: return np.nan
    return (active > 0).mean()

def profit_factor(rets):
    """Отношение суммы положительных доходностей к сумме отрицательных
    (по модулю); чем выше, тем лучше соотношение прибыли к убыткам."""
    gains  = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    if losses == 0: return np.inf
    return gains / losses

def compute_all_metrics(rets, name):
    """Собирает все метрики выше в один словарь для сводной таблицы."""
    return {
        "Стратегия": name,
        "Cumulative Return": f"{(1 + rets).prod() - 1:.4f}" if len(rets) else "—",
        "Annualized Return": f"{annualized_return(rets):.4f}",
        "Annualized Volatility": f"{annualized_volatility(rets):.4f}",
        "Sharpe Ratio": f"{sharpe(rets):.4f}",
        "Sortino Ratio": f"{sortino(rets):.4f}",
        "Max Drawdown": f"{max_drawdown(rets):.4f}",
        "Calmar Ratio": f"{calmar(rets):.4f}",
        "Win Rate (daily)": f"{win_rate(rets):.4f}",
        "Profit Factor": f"{profit_factor(rets):.4f}",
        "N дней": len(rets),
    }

print(f"Расчёт стратегий (5-слойный портфель, gross - без TC)")

ret_lo = build_layered_strategy(signals, mode="long_only")
ret_ls = build_layered_strategy(signals, mode="long_short")
ret_ls_cf = build_layered_strategy(signals, mode="long_short",
                                    conf_threshold=CONF_THRESHOLD)
ret_bh    = build_buy_hold(signals)

print(f"Long-Only: {len(ret_lo)} дней")
print(f"Long-Short: {len(ret_ls)} дней")
print(f"Long-Short (|p−.5|>{CONF_THRESHOLD}): {len(ret_ls_cf)} дней")
print(f"Equal-Weight Benchmark: {len(ret_bh)} дней")

print("ЭКОНОМИЧЕСКИЕ МЕТРИКИ — PatchTST (gross, без транзакционных издержек)")

metrics_list = [
    compute_all_metrics(ret_lo, "PatchTST Long-Only"),
    compute_all_metrics(ret_ls, "PatchTST Long-Short"),
    compute_all_metrics(ret_ls_cf, f"Long-Short (|p−.5|>{CONF_THRESHOLD})"),
    compute_all_metrics(ret_bh, "Equal-Weight Benchmark (бенчмарк)"),
]
metrics_df = pd.DataFrame(metrics_list).set_index("Стратегия").T
print(metrics_df.to_string())

cum_lo = (1 + ret_lo).cumprod()
cum_ls = (1 + ret_ls).cumprod()
cum_ls_cf = (1 + ret_ls_cf).cumprod()
cum_bh = (1 + ret_bh).cumprod()

def drawdown_series(rets):
    """Ряд просадок от текущего максимума кривой капитала (для графика)."""
    cum  = (1 + rets).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak

dd_lo = drawdown_series(ret_lo)
dd_ls = drawdown_series(ret_ls)
dd_ls_cf = drawdown_series(ret_ls_cf)
dd_bh = drawdown_series(ret_bh)

# --- Визуализация: кривые капитала, просадки, сводная таблица метрик ---
fig, axes = plt.subplots(3, 1, figsize=(14, 13))

axes[0].plot(cum_lo.index, cum_lo.values, label="Long-Only", color="steelblue")
axes[0].plot(cum_ls.index, cum_ls.values, label="Long-Short", color="darkorange")
axes[0].plot(cum_ls_cf.index, cum_ls_cf.values, label=f"LS (|p−.5|>{CONF_THRESHOLD})", color="seagreen")
axes[0].plot(cum_bh.index, cum_bh.values, label="Equal-Weight Benchmark", color="gray", linestyle="--")
axes[0].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
axes[0].set_title("Кривая капитала (gross, без TC)", fontsize=13)
axes[0].set_ylabel("Стоимость портфеля (нач. = 1)")
axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.4)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

axes[1].fill_between(dd_lo.index, dd_lo.values, 0, alpha=0.35, color="steelblue",  label="Long-Only")
axes[1].fill_between(dd_ls.index, dd_ls.values, 0, alpha=0.35, color="darkorange", label="Long-Short")
axes[1].fill_between(dd_ls_cf.index, dd_ls_cf.values, 0, alpha=0.35, color="seagreen",   label=f"LS (|p−.5|>{CONF_THRESHOLD})")
axes[1].plot(dd_bh.index, dd_bh.values, color="gray", linestyle="--", label="Equal-Weight Benchmark")
axes[1].set_title("Просадки (Drawdown)", fontsize=13)
axes[1].set_ylabel("Просадка от пика")
axes[1].legend(loc="best")
axes[1].grid(True, alpha=0.4)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)

axes[2].axis("off")
col_labels = ["Метрика"] + list(metrics_df.columns)
row_data   = [[idx] + list(metrics_df.loc[idx]) for idx in metrics_df.index]

tbl = axes[2].table(cellText=row_data, colLabels=col_labels,
                    cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(row_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor("#f0f4f8")

axes[2].set_title("Сводная таблица метрик (gross)", fontsize=13, pad=20)

plt.suptitle(f"PatchTST - экономический бэктест (горизонт {HORIZON}д, {signals['ticker'].nunique()} тикеров)",
             fontsize=14, y=1.00)
plt.tight_layout(h_pad=3)
plt.show()
