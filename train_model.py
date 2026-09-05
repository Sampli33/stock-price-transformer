"""
Обучает PatchTST-подобный трансформер для бинарной классификации направления
цены (рост/падение) на скользящих окнах технических индикаторов.

Вход: dataset_ready.pkl — датасет, подготовленный build_dataset.py
       (окна X, метки y, разбитые на train/val/test).

Пайплайн:
    1. Расчёт весов классов для борьбы с дисбалансом (train split).
    2. Определение модели PatchTST: вход нарезается на патчи по времени,
       патчи прогоняются через трансформер-энкодер, затем классификационную
       голову.
    3. Обучение с early stopping по macro F1 на валидации и
       ReduceLROnPlateau-шедулером.
    4. Оценка лучшей модели на test: accuracy, macro F1, ROC-AUC,
       classification report.
    5. Визуализация кривых обучения (loss, F1) и confusion matrix.

Выход:
    best_model_patchtst.pt — веса лучшей по val macro F1 модели.
    Графики обучения и confusion matrix.
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

with open('dataset_ready.pkl', 'rb') as f:
    datasets = pickle.load(f)

# Веса классов для компенсации дисбаланса
# -100 зарезервировано как метка "нет валидного таргета"
# и должно быть исключено из расчёта весов.
y_train = datasets["train"]["y"]
active_labels = y_train[y_train != -100]
counts = np.bincount(active_labels)

total_active = len(active_labels)
w0 = total_active / (2.0 * counts[0])
w1 = total_active / (2.0 * counts[1])
class_weights = torch.tensor([w0, w1], dtype=torch.float32)

print(f"Рассчитанные веса: Класс 0 (Падение) = {w0:.2f}, Класс 1 (Рост) = {w1:.2f}")
print(f"Train: class 0={counts[0]}  class 1={counts[1]}")


def to_tensors(split: str) -> TensorDataset:
"""Конвертирует один из сплитов датасета (train/val/test) в TensorDataset."""
    X = torch.tensor(datasets[split]["X"], dtype=torch.float32)
    y = torch.tensor(datasets[split]["y"], dtype=torch.long)
    return TensorDataset(X, y)


train_ds = to_tensors("train")
val_ds = to_tensors("val")
test_ds = to_tensors("test")

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

N_FEATURES = datasets["train"]["X"].shape[2]
SEQ_LEN = datasets["train"]["X"].shape[1]

print(f"Форма входа: (batch, {SEQ_LEN}, {N_FEATURES})")


class PatchTST(nn.Module):
    """Трансформер-классификатор PatchTST для многомерных временных рядов.

    Каждый из n_features признаков обрабатывается независимо: временная ось
    нарезается на перекрывающиеся патчи длиной patch_len (шаг stride), патчи
    проецируются в d_model-мерное пространство, кодируются позиционными
    эмбеддингами и прогоняются через общий (shared) трансформер-энкодер.
    Затем эмбеддинги всех признаков конкатенируются и подаются в
    классификационную голову (2 класса: падение/рост).
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        head_in = n_features * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features) -> logits (batch, 2)
        """
        B = x.size(0)

        # (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)

        # Нарезка на патчи по временной оси:
        # (batch, n_features, seq_len) -> (batch, n_features, num_patches, patch_len)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        _, C, N, P = x.shape

        # Каждый признак трактуется как отдельный "канал" в батче,
        # чтобы прогнать все каналы через один и тот же трансформер разом
        x = x.reshape(B * C, N, P)

        x = self.patch_proj(x)

        positions = torch.arange(N, device=x.device)
        x = x + self.pos_emb(positions)

        x = self.transformer(x)

        x = x.mean(dim=1)
        x = x.reshape(B, C * self.d_model)

        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

model = PatchTST().to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Параметров модели: {n_params:,}")
print(f"Патчей в последовательности: {model.num_patches}  "
      f"(patch_len={model.patch_len}, stride={model.stride})")

opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# ignore_index=-100 исключает из лосса примеры без валидной метки
crit = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=-100,
                           label_smoothing=0.2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', patience=4, factor=0.5)


def run_epoch(loader, train=True):
    """Прогоняет одну эпоху (обучение или инференс) по переданному loader'у.

    Возвращает (средний loss, accuracy, macro F1) по всем валидным
    (не -100) примерам эпохи.
    """
    model.train() if train else model.eval()
    total_loss, preds, trues = 0, [], []
    n_active = 0

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = crit(out, y)

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            active_mask = (y != -100).cpu().numpy()
            batch_preds = out.argmax(1).cpu().numpy()
            batch_true = y.cpu().numpy()

            preds.extend(batch_preds[active_mask])
            trues.extend(batch_true[active_mask])

            n_active += active_mask.sum()
            total_loss += loss.item() * active_mask.sum()

    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    return total_loss / n_active, accuracy_score(trues, preds), macro_f1


EPOCHS = 100
PATIENCE = 12
best_val_f1 = 0.0
patience_ctr = 0
history = []

print(f"\n{'Epoch':>5} | {'tr_loss':>8} {'tr_acc':>7} {'tr_f1':>7} | "
      f"{'vl_loss':>8} {'vl_acc':>7} {'vl_f1':>7} | {'lr':>8}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_f1 = run_epoch(val_loader, train=False)

    current_lr = opt.param_groups[0]["lr"]
    print(f"{epoch:5d} | {tr_loss:8.4f} {tr_acc:7.4f} {tr_f1:7.4f} | "
          f"{vl_loss:8.4f} {vl_acc:7.4f} {vl_f1:7.4f} | {current_lr:8.2e}")

    history.append(dict(
        epoch=epoch,
        tr_loss=tr_loss, tr_acc=tr_acc, tr_f1=tr_f1,
        vl_loss=vl_loss, vl_acc=vl_acc, vl_f1=vl_f1
    ))

    scheduler.step(vl_f1)

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        patience_ctr = 0
        torch.save(model.state_dict(), "best_model_patchtst.pt")
        print(f"новый лучший val macro_f1={best_val_f1:.4f}")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping на эпохе {epoch} (нет улучшений {PATIENCE} эпох)")
            break

model.load_state_dict(torch.load("best_model_patchtst.pt"))
model.eval()

all_preds, all_probs, all_true = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        out = model(X.to(device))
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(y.numpy())

all_true_np = np.array(all_true)
all_preds_np = np.array(all_preds)
all_probs_np = np.array(all_probs)

mask = (all_true_np == 0) | (all_true_np == 1)
all_true = all_true_np[mask]
all_preds = all_preds_np[mask]
all_probs = all_probs_np[mask]

test_acc = accuracy_score(all_true, all_preds)
test_macro_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
test_roc_auc = roc_auc_score(all_true, all_probs)

print(f"\nТест: accuracy={test_acc:.4f}  macro_F1={test_macro_f1:.4f}  ROC-AUC={test_roc_auc:.4f}")
print(f"\n{classification_report(all_true, all_preds, target_names=['падение', 'рост'], zero_division=0)}")

epochs_done = [h["epoch"] for h in history]
tr_losses = [h["tr_loss"] for h in history]
vl_losses = [h["vl_loss"] for h in history]
tr_f1s = [h["tr_f1"] for h in history]
vl_f1s = [h["vl_f1"] for h in history]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(epochs_done, tr_losses, label="Train")
axes[0].plot(epochs_done, vl_losses, label="Val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_done, tr_f1s, label="Train macro F1")
axes[1].plot(epochs_done, vl_f1s, label="Val macro F1")
axes[1].set_title("Macro F1"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(True)

cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Пред: 0", "Пред: 1"],
            yticklabels=["Факт: 0", "Факт: 1"])
axes[2].set_title("Confusion Matrix (Test)")

plt.tight_layout()
plt.show()
