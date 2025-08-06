import warnings  # suppress warnings
warnings.filterwarnings('ignore')  # ignore all warnings

# ============================ mi_model.py ============================
import torch  # PyTorch core
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # functional API
import random  # for seeding
import numpy as np  # numerical operations

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.75, gamma=4.0):
        super().__init__()
        self.alpha = alpha  # weighting factor
        self.gamma = gamma  # focusing parameter

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # compute cross-entropy per sample
        pt = torch.exp(-ce_loss)  # probability of correct class
        # apply focal formula
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

class EEG_MI_GRUBinary(nn.Module):
    """GRU-based binary classifier for MI EEG data."""
    def __init__(self, input_channels=15, hidden_size=256, num_classes=2, dropout=0.5):
        super().__init__()
        # bidirectional GRU layer
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # simple classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # dropout for regularization
            nn.Linear(hidden_size * 2, num_classes)  # map to output classes
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # rearrange to (batch, time, channels)
        _, h_n = self.gru(x)  # run through GRU
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # concatenate final states
        return self.classifier(h_cat)  # output logits


# ============================ mi_preprocess.py ============================
import pandas as pd  # data handling
import os  # file operations
from sklearn.preprocessing import StandardScaler  # feature scaling
from scipy.signal import iirnotch, filtfilt  # signal filtering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device

class Preprocess:
    """Data loading and preprocessing utilities."""
    def load_data(base_path, subjects, sessions, samples_per_trial=2250):
        X = []  # list to collect DataFrames
        for s in subjects:
            s_folder = f"S{s}"  # subject folder
            for subfolder in sessions:
                file_path = os.path.join(base_path, s_folder, str(subfolder), "EEGdata.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)  # read session CSV
                    X.append(df)
        X = pd.concat(X, ignore_index=True)  # combine all sessions
        columns_to_drop = ['Battery', 'Counter', 'Validation']  # drop non-EEG cols
        # split into trials
        X_split = [
            X.iloc[(n * samples_per_trial):(n + 1) * samples_per_trial]
             .drop(columns=columns_to_drop)
            for n in range(len(X) // samples_per_trial)
        ]
        return X_split

    def load_labels(csv_path, nrows):
        df = pd.read_csv(csv_path, nrows=nrows)[['label']]  # read labels
        df['label'] = df['label'].map({'Left': 0, 'Right': 1})  # map to integers
        return df

    def preprocess_sessions(dfs_list, labels):
        processed = []  # store processed arrays
        fs, f0, Q = 250, 50, 30  # filter parameters
        b_notch, a_notch = iirnotch(f0, Q, fs)  # design notch filter
        eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']  # EEG channel names

        for df in dfs_list:
            time = df['Time'].values  # extract time column
            delta_t = np.diff(time, prepend=time[0])  # compute time deltas
            df_proc = df.drop(columns=['Time'])  # drop time from features

            eeg_only = df_proc[eeg_channels]  # select EEG data
            # apply notch filter on each channel
            filtered = eeg_only.apply(lambda ch: filtfilt(b_notch, a_notch, ch), axis=0)

            rest_cols = [col for col in df_proc.columns if col not in eeg_channels]  # non-EEG cols
            rest = df_proc[rest_cols]  # select rest
            rest['delta_t'] = delta_t  # add time delta

            full_df = pd.concat([filtered, rest], axis=1)  # combine all features
            processed.append(full_df.values)

        X_np = np.stack(processed)  # stack into array
        Y_np = np.array(labels).squeeze()  # labels array
        scaler = StandardScaler()  # instantiate scaler
        # scale each trial separately
        X_scaled = [scaler.fit_transform(session) for session in X_np]
        X_np = np.stack(X_scaled)
        # convert to tensor and transpose to (batch, channels, time)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).permute(0, 2, 1).to(device)
        Y_tensor = torch.tensor(Y_np, dtype=torch.long).to(device).squeeze()
        return X_tensor, Y_tensor


# ============================ mi_train_with_seeds.py ============================
from torch.utils.data import TensorDataset, DataLoader  # data utilities

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load and preprocess training data
train_X = Preprocess.load_data(
    "/kaggle/input/mtcaic3-phase-ii/MI/train", range(1, 31), range(1, 9)
)
train_Y = Preprocess.load_labels('/kaggle/input/mtcaic3-phase-ii/train.csv', 2400)
X_tensor, Y_tensor = Preprocess.preprocess_sessions(train_X, train_Y['label'])

# Load and preprocess validation data
val_X = Preprocess.load_data(
    "/kaggle/input/mtcaic3-phase-ii/MI/validation", range(31, 36), [1]
)
val_Y = Preprocess.load_labels('/kaggle/input/mtcaic3-phase-ii/validation.csv', 50)
X_val_tensor, Y_val_tensor = Preprocess.preprocess_sessions(val_X, val_Y['label'])

# prepare DataLoader
train_ds = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

best_val_acc = 0  # track best validation accuracy
best_seed = None  # track best seed

for seed in [13]:
    print(f"\n🔁 Training with seed {seed}")
    set_seeds(seed)  # apply seed

    # instantiate model and optimizer
    model = EEG_MI_GRUBinary(
        input_channels=15, num_classes=2, hidden_size=256, dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.95, patience=3
    )
    criterion = FocalLoss(alpha=0.75, gamma=4.0)  # focal loss

    patience = 0
    best_seed_val = 0

    for epoch in range(100):
        model.train()  # set train mode
        total_loss = correct = total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()  # clear gradients
            preds = model(xb)  # forward pass
            loss = criterion(preds, yb.squeeze())  # compute loss
            loss.backward()  # backpropagation
            optimizer.step()  # update params
            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        train_acc = 100 * correct / total  # compute train accuracy

        model.eval()  # set eval mode
        with torch.no_grad():
            val_preds = model(X_val_tensor)  # val forward
            val_acc = 100 * (val_preds.argmax(dim=1) == Y_val_tensor).float().mean().item()

        scheduler.step(val_acc)  # adjust lr

        print(f"📚 Epoch {epoch+1} | Seed {seed} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_seed_val:
            best_seed_val = val_acc
            torch.save(model.state_dict(), f"mi_model_seed{seed}.pt")  # save checkpoint
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break  # early stopping

    print(f"✅ Seed {seed} finished with Best Val Acc = {best_seed_val:.2f}%")

    if best_seed_val > best_val_acc:
        best_val_acc = best_seed_val
        best_seed = seed

print(f"\n🏆 BEST SEED = {best_seed} with Val Acc = {best_val_acc:.2f}%")


# ============================ mi_inference.py ============================
# load best model for inference
mi_model = EEG_MI_GRUBinary(
    input_channels=15, num_classes=2, hidden_size=256, dropout=0.2
).to(device)
mi_model.load_state_dict(torch.load(f"mi_model_seed{best_seed}.pt", map_location=device))
mi_model.eval()  # set eval mode

dfs = []  # list to hold test DataFrames
ids = []  # list to hold IDs
label_map = {0: "Left", 1: "Right"}  # map indices back to labels
trial_id = 4901  # starting trial ID

# collect test trials
for s in range(36, 46):
    eeg_path = f"/kaggle/input/mtcaic3-phase-ii/MI/test/S{s}/1/EEGdata.csv"
    if os.path.exists(eeg_path):
        df = pd.read_csv(eeg_path)  # read test session
        for t in range(10):
            start = t * 2250
            end = start + 2250
            trial_df = df.iloc[start:end].drop(columns=["Battery", "Counter", "Validation"])
            dfs.append(trial_df)
            ids.append(trial_id)
            trial_id += 1

def preprocess_test_sessions(dfs_list):
    """Preprocess test sessions similar to training."""
    processed = []
    fs, f0, Q = 250, 50, 30  # filter params
    b_notch, a_notch = iirnotch(f0, Q, fs)  # design notch
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']  # channel list

    for df in dfs_list:
        time = df['Time'].values  # extract time
        delta_t = np.diff(time, prepend=time[0])  # compute delta_t
        df_proc = df.drop(columns=['Time'])  # drop time col

        eeg_only = df_proc[eeg_channels]  # select EEG cols
        filtered = eeg_only.apply(lambda ch: filtfilt(b_notch, a_notch, ch), axis=0)  # filter

        rest_cols = [col for col in df_proc.columns if col not in eeg_channels]  # other cols
        rest = df_proc[rest_cols]
        rest['delta_t'] = delta_t  # add delta_t

        full_df = pd.concat([filtered, rest], axis=1)  # combine
        processed.append(full_df.values)

    X_np = np.stack(processed)  # stack trials
    scaler = StandardScaler()  # scale
    X_scaled = [scaler.fit_transform(session) for session in X_np]
    X_np = np.stack(X_scaled)
    # convert to tensor, permute to (batch, channels, time)
    X_tensor = torch.tensor(X_np, dtype=torch.float32).permute(0, 2, 1).to(device)
    return X_tensor

X_test = preprocess_test_sessions(dfs)  # preprocess

with torch.no_grad():
    outputs = mi_model(X_test)  # forward pass
    preds = outputs.argmax(dim=1).cpu().numpy()  # get predictions

labels = [label_map[p] for p in preds]  # map to labels

# save submission
submission = pd.DataFrame({"id": ids, "label": labels})
submission.to_csv("mi_submission.csv", index=False)
print("✅ Saved mi_submission.csv with", len(labels), "predictions")
