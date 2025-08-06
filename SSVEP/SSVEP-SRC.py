import warnings  # to control warning messages
warnings.filterwarnings('ignore')  # ignore all warnings

# ============================ ssvep_model_preprocess.py ============================
import torch  # main PyTorch library
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # functional API for layers
import pandas as pd  # data handling
import os  # file system operations
import numpy as np  # numerical operations
from sklearn.preprocessing import StandardScaler  # feature scaling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set compute device

# ========== Model ==========
class PositionalEncoding(nn.Module):
    """Add positional information to embeddings for transformer."""
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # dropout layer
        pe = torch.zeros(max_len, model_dim)  # prepare positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # positions
        # compute divisor term for sine/cosine frequencies
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cosine to odd indices
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)  # store as buffer (not a parameter)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # add positional encoding
        return self.dropout(x)  # apply dropout


class TransformerEEGClassifier(nn.Module):
    """Transformer-based classifier for SSVEP EEG data."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.3):
        super(TransformerEEGClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)  # linear projection to model dimension
        self.pos_encoder = PositionalEncoding(model_dim, dropout)  # positional encoding layer
        # build a stack of transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),  # reduce to 64 dims
            nn.ReLU(),                 # activation
            nn.Dropout(dropout),       # dropout
            nn.Linear(64, num_classes) # final logits
        )

    def forward(self, x):
        x = self.embedding(x)  # embed input
        x = self.pos_encoder(x)  # add positional info
        x = self.transformer_encoder(x)  # transformer encoding
        x = x.mean(dim=1)  # global average pooling over time
        return self.classifier(x)  # output logits


# ========== Preprocessing ==========
from scipy.signal import iirnotch, filtfilt  # notch filter design and apply
from sklearn.preprocessing import StandardScaler  # scaling
import torch
import os
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # compute device

class Preprocess:
    """Data loading and preprocessing for SSVEP."""
    def load_data(base_path, subjects, sessions, samples_per_trial=1750):
        X = []  # list to collect DataFrames
        for s in subjects:
            for subfolder in sessions:
                path = os.path.join(base_path, f"S{s}", str(subfolder), "EEGdata.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)  # read CSV
                    X.append(df)
        X = pd.concat(X, ignore_index=True)  # combine all sessions
        columns_to_drop = ['Battery', 'Counter', 'Validation']  # non-EEG columns
        # split into individual trials
        X_split = [
            X.iloc[n * samples_per_trial: (n + 1) * samples_per_trial]
             .drop(columns=columns_to_drop)
            for n in range(len(X) // samples_per_trial)
        ]
        return X_split

    def load_labels(csv_path, nrows):
        Y = pd.read_csv(csv_path).tail(nrows)[['label']]  # read last nrows labels
        # map string labels to integers
        Y['label'] = Y['label'].map({'Left': 0, 'Right': 1, 'Forward': 2, 'Backward': 3})
        return Y

    def preprocess_sessions(dfs_list, labels):
        processed = []  # store processed arrays

        # design notch filter @50Hz
        fs, f0, Q = 250, 50, 30
        b_notch, a_notch = iirnotch(f0, Q, fs)

        eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']  # EEG channels

        for df in dfs_list:
            time = df['Time'].values  # extract time stamps
            delta_t = np.diff(time, prepend=time[0])  # compute time differences
            df_proc = df.drop(columns=['Time'])  # drop time col

            # apply notch filter to EEG channels
            eeg_only = df_proc[eeg_channels]
            eeg_filtered = eeg_only.apply(lambda ch: filtfilt(b_notch, a_notch, ch), axis=0)

            # handle other sensor columns
            other_cols = [col for col in df_proc.columns if col not in eeg_channels]
            others = df_proc[other_cols]
            others['delta_t'] = delta_t  # add time delta

            df_filtered = pd.concat([eeg_filtered, others], axis=1)  # combine all features
            processed.append(df_filtered.values)

        X_np = np.stack(processed)  # stack into array
        # scale each trial
        X_scaled = [StandardScaler().fit_transform(session) for session in X_np]
        X_np = np.stack(X_scaled)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)  # to tensor
        Y_tensor = torch.tensor(np.array(labels), dtype=torch.long).to(device).squeeze()  # labels
        return X_tensor, Y_tensor


# ============================ ssvep_train_with_seeds.py ============================
import random  # for seeding
from torch.utils.data import TensorDataset, DataLoader  # data utilities
from sklearn.metrics import f1_score  # evaluation metric

def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load and preprocess train data
train_X = Preprocess.load_data("/kaggle/input/mtcaic3-phase-ii/SSVEP/train", range(1, 31), range(1, 9))
train_Y = Preprocess.load_labels("/kaggle/input/mtcaic3-phase-ii/train.csv", 2400)
X_train, Y_train = Preprocess.preprocess_sessions(train_X, train_Y['label'])

# load and preprocess validation data
val_X = Preprocess.load_data("/kaggle/input/mtcaic3-phase-ii/SSVEP/validation", range(31, 36), [1])
val_Y = Preprocess.load_labels("/kaggle/input/mtcaic3-phase-ii/validation.csv", 50)
X_val, Y_val = Preprocess.preprocess_sessions(val_X, val_Y['label'])

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=100, shuffle=True)  # data loader

best_val_acc = 0  # best validation accuracy
best_seed = None  # best seed

for seed in [13]:
    print(f"\n🔁 Training with seed {seed}")
    set_seeds(seed)  # reproducible seed

    # instantiate model and training tools
    model = TransformerEEGClassifier(15, 64, 4, 1, 4, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)
    criterion = nn.CrossEntropyLoss()  # standard cross-entropy

    best_seed_val = 0
    patience = 0

    # training loop
    for epoch in range(200):
        model.train()
        correct = total = loss_sum = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()  # reset gradients
            preds = model(xb)  # forward pass
            loss = criterion(preds, yb)  # compute loss
            loss.backward()  # backprop
            optimizer.step()  # update weights
            loss_sum += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = 100 * correct / total  # train accuracy

        # validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_acc = 100 * (val_preds.argmax(1) == Y_val).float().mean().item()
            val_f1 = f1_score(Y_val.cpu(), val_preds.argmax(1).cpu(), average='macro')

        scheduler.step(val_acc)  # adjust LR
        print(f"📚 Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f} | Loss: {loss_sum:.4f}")

        # checkpointing
        if val_acc > best_seed_val:
            best_seed_val = val_acc
            torch.save(model.state_dict(), f"ssvep_model_seed{seed}.pt")
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print("⏹️ Early stopping")
                break

    print(f"✅ Seed {seed} finished with Best Val Acc = {best_seed_val:.2f}%")

    # keep best overall
    if best_seed_val > best_val_acc:
        best_val_acc = best_seed_val
        best_seed = seed

print(f"\n🏆 Best seed: {best_seed} with Validation Accuracy: {best_val_acc:.2f}%")


# ============================ ssvep_inference.py ============================
import pandas as pd  # for DataFrame
from sklearn.preprocessing import StandardScaler  # for scaling
import os  # file ops
from scipy.signal import iirnotch, filtfilt  # filtering

# === Load Best Model ===
best_model_path = f"ssvep_model_seed{best_seed}.pt"
inference_model = TransformerEEGClassifier(15, 64, 4, 1, 4, dropout=0.3).to(device)
inference_model.load_state_dict(torch.load(best_model_path, map_location=device))
inference_model.eval()  # set to evaluation mode

# === Load Test Data ===
dfs = []  # store raw trial DataFrames
ids = []  # store trial IDs
trial_id = 5001  # starting ID for SSVEP test

for s in range(36, 46):  # subjects S36–S45
    path = f"/kaggle/input/mtcaic3-phase-ii/SSVEP/test/S{s}/1/EEGdata.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)  # read full session
        for t in range(10):  # split into 10 trials
            start = t * samples_per_trial
            end = start + samples_per_trial
            trial_df = df.iloc[start:end].drop(columns=['Battery', 'Counter', 'Validation'])
            dfs.append(trial_df)
            ids.append(trial_id)
            trial_id += 1

# === Preprocess Test Data ===
def preprocess_test_sessions(dfs_list):
    """Apply same filtering & scaling as training."""
    processed = []
    fs, f0, Q = 250, 50, 30  # filter params
    b_notch, a_notch = iirnotch(f0, Q, fs)  # design notch filter
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']

    for df in dfs_list:
        time = df['Time'].values
        delta_t = np.diff(time, prepend=time[0])
        df_proc = df.drop(columns=['Time'])
        eeg_only = df_proc[eeg_channels]
        eeg_filtered = eeg_only.apply(lambda ch: filtfilt(b_notch, a_notch, ch), axis=0)
        other_cols = [col for col in df_proc.columns if col not in eeg_channels]
        others = df_proc[other_cols]
        others['delta_t'] = delta_t
        df_filtered = pd.concat([eeg_filtered, others], axis=1)
        processed.append(df_filtered.values)

    X_np = np.stack(processed)
    X_scaled = [StandardScaler().fit_transform(session) for session in X_np]
    X_np = np.stack(X_scaled)
    return torch.tensor(X_np, dtype=torch.float32).to(device)

X_test = preprocess_test_sessions(dfs)  # get test tensor

# === Predict ===
with torch.no_grad():
    outputs = inference_model(X_test)  # forward pass
    preds = outputs.argmax(dim=1).cpu().numpy()  # predicted classes

# === Prepare Submission ===
label_map = {0: "Left", 1: "Right", 2: "Forward", 3: "Backward"}
labels = [label_map[p] for p in preds]  # map indices to labels

submission = pd.DataFrame({"id": ids, "label": labels})  # build DataFrame
submission.to_csv("ssvep_submission.csv", index=False)  # write CSV
print(f"✅ Saved submission with {len(labels)} predictions to ssvep_submission.csv")
