import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from katransformer import create_kat_amp_model  # model factory

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
PAD_LENGTH = 3000  # fixed sequence length
NEURONS = 125  # number of internal neurons (GroupKAN)
ACTIVATION = "SoftSign"
DROPOUT_RATE = 0.48  # dropout on CLS token
BATCH_SIZE = 86
LR = 1.792e-6  # learning rate
EPOCHS = 47

print(
    f"Hyperparams → neurons={NEURONS}, act={ACTIVATION}, dropout={DROPOUT_RATE}, "
    f"batch={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}"
)


# ──────────────────────────────────────────────────────────────────────────────
#  DATASET & COLLATION
# ──────────────────────────────────────────────────────────────────────────────
class AMPDataset(Dataset):
    """CSV must contain columns 'aa_seq' and 'AMP' (0 or 1)."""

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["aa_seq"]
        label = torch.tensor(int(self.df.iloc[idx]["AMP"]), dtype=torch.float)
        # encode A-Z → 0-25, else 0
        enc = torch.tensor(
            [ord(c) - ord("A") if "A" <= c <= "Z" else 0 for c in seq], dtype=torch.long
        )
        return enc, label


def collate_fn(batch, pad_len: int = PAD_LENGTH):
    seqs, labels = zip(*batch)
    x = torch.zeros(len(seqs), pad_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        ln = min(len(s), pad_len)
        x[i, :ln] = s[:ln]
    return x, torch.tensor(labels)


# ──────────────────────────────────────────────────────────────────────────────
#  CONFUSION METRICS
# ──────────────────────────────────────────────────────────────────────────────
def confusion_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    def safe(n, d):
        return n / d if d else float("nan")

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": safe(tp, tp + fn),  # recall
        "TNR": safe(tn, tn + fp),  # specificity
        "PPV": safe(tp, tp + fp),  # precision
        "NPV": safe(tn, tn + fn),
        "FNR": safe(fn, fn + tp),
        "FPR": safe(fp, fp + tn),
        "FDR": safe(fp, fp + tp),
        "FOR": safe(fn, fn + tn),
        "ACC": safe(tp + tn, tp + tn + fp + fn),
        "F1": safe(2 * tp, 2 * tp + fp + fn),
        "MCC": safe(
            tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        ),
        "BM": safe(tp, tp + fn) + safe(tn, tn + fp) - 1,
        "MK": safe(tp, tp + fp) + safe(tn, tn + fn) - 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────
csv_path = "dataset_benchmarking/data.csv"
train_df, val_df = train_test_split(
    pd.read_csv(csv_path), test_size=0.3, random_state=42
)
train_df.to_csv("dataset_benchmarking/train_amp.csv", index=False)
val_df.to_csv("dataset_benchmarking/val_amp.csv", index=False)

train_ds = AMPDataset("dataset_benchmarking/train_amp.csv")
val_ds = AMPDataset("dataset_benchmarking/val_amp.csv")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# ──────────────────────────────────────────────────────────────────────────────
#  MODEL, OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_kat_amp_model(
    seq_length=PAD_LENGTH,
    embed_dim=768,
    num_classes=1,
    neurons=NEURONS,
    activation=ACTIVATION,
    dropout_rate=DROPOUT_RATE,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ──────────────────────────────────────────────────────────────────────────────
#  TRAIN & COLLECT PER-SEQUENCE EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────
sequence_vectors = []  # will collect [B, embed_dim] arrays

epoch_labels, epoch_preds = [], []
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = correct = count = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits, token_emb = model(x_batch)
        loss = criterion(logits.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds.squeeze() == y_batch).sum().item()
        count += y_batch.size(0)
        total_loss += loss.item() * y_batch.size(0)

        # mean-pool token embeddings → [B, embed_dim]
        seq_vecs = token_emb.mean(dim=1).detach().cpu().numpy()
        sequence_vectors.append(seq_vecs)

    scheduler.step()
    train_acc = correct / count
    train_loss = total_loss / count

    # validation
    model.eval()
    v_loss = v_correct = v_count = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, _ = model(x_batch)
            loss = criterion(logits.squeeze(), y_batch)
            v_loss += loss.item() * y_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            v_correct += (preds.squeeze() == y_batch).sum().item()
            v_count += y_batch.size(0)
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.squeeze().cpu().numpy())
    val_acc = v_correct / v_count
    print(
        f"Epoch {ep:2d}/{EPOCHS}  train-loss {train_loss:.4f}  "
        f"train-acc {train_acc:.4f}  val-loss {v_loss/v_count:.4f}  val-acc {val_acc:.4f}"
    )

# ──────────────────────────────────────────────────────────────────────────────
#  SAVE MODEL & METRICS
# ──────────────────────────────────────────────────────────────────────────────
model_path = "dataset_benchmarking/amp_prediction_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved → {model_path}")

# save confusion metrics on last epoch
y_true = np.array(all_labels, dtype=int)
y_pred = np.array(all_preds, dtype=int)
stats = confusion_metrics(y_true, y_pred)
with open("dataset_benchmarking/metrics.txt", "w") as f:
    f.write("\t".join(stats.keys()) + "\n")
    f.write("\t".join(map(str, stats.values())) + "\n")
print("Metrics written → dataset_benchmarking/metrics.txt")

# ──────────────────────────────────────────────────────────────────────────────
#  EXPORT PER-SEQUENCE EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────
seq_mat = np.concatenate(sequence_vectors, axis=0)
cols = [f"F_{i}" for i in range(seq_mat.shape[1])]
pd.DataFrame(seq_mat, columns=cols).to_csv(
    "dataset_benchmarking/per_sequence_weights.csv", index=False
)
print("Per-sequence weights saved → dataset_benchmarking/per_sequence_weights.csv")
