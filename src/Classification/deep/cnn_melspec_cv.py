import os
import random
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader


# ============================================================
# General configuration
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_06_cnn_melspec"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

MODEL_OUT = "models/cnn_melspec/cnn_melspec.pt"

SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 256

BATCH_SIZE = 8
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """
    Make the experiment more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Audio and feature extraction
# ============================================================

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load one audio file and resample it to 16 kHz.
    """

    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def extract_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Convert an audio waveform into a fixed-size log-mel spectrogram.

    Output shape:
        (N_MELS, MAX_FRAMES)

    This representation can be seen as an audio image:
    - vertical axis: mel frequency bins
    - horizontal axis: time frames
    - values: normalized log energy
    """

    # Remove leading and trailing silence if possible
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio_trimmed) > int(0.2 * SAMPLE_RATE):
        audio = audio_trimmed

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize each spectrogram independently
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    # Pad or truncate to a fixed duration
    if log_mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    return log_mel.astype(np.float32)


# ============================================================
# Dataset preparation
# ============================================================

def build_dataset():
    """
    Load labels_clean.csv and build mel-spectrogram features for all audios.
    """

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    df = pd.read_csv(LABELS_PATH)

    required_columns = {"filename", "label_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"labels_clean.csv must contain columns {required_columns}. "
            f"Missing columns: {missing_columns}"
        )

    X = []
    y = []
    filenames = []
    missing_files = []
    failed_files = []

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue

        try:
            audio = load_audio(audio_path, sr=SAMPLE_RATE)
            mel = extract_mel_spectrogram(audio)

            X.append(mel)
            y.append(label)
            filenames.append(filename)

        except Exception as error:
            failed_files.append((audio_path, str(error)))

    if missing_files:
        print("\nMissing audio files:")
        for path in missing_files:
            print("-", path)

    if failed_files:
        print("\nFiles with mel-spectrogram extraction errors:")
        for path, error in failed_files:
            print(f"- {path}: {error}")

    return np.array(X), np.array(y), np.array(filenames)


def filter_classes_with_enough_samples(X, y, filenames, min_samples=2):
    """
    Remove classes with fewer than min_samples examples.

    StratifiedKFold with 2 folds requires every evaluated class
    to have at least 2 samples.
    """

    class_counts = pd.Series(y).value_counts().sort_index()

    valid_classes = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples]

    mask = pd.Series(y).isin(valid_classes).to_numpy()

    X_filtered = X[mask]
    y_filtered = y[mask]
    filenames_filtered = filenames[mask]

    return X_filtered, y_filtered, filenames_filtered, class_counts, removed_classes


# ============================================================
# PyTorch dataset
# ============================================================

class MelDataset(Dataset):
    """
    Simple PyTorch dataset for log-mel spectrograms.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# CNN model
# ============================================================

class SmallMelCNN(nn.Module):
    """
    A compact CNN for log-mel spectrogram classification.

    The model is intentionally small because the dataset is very limited.
    A larger network would likely overfit.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (N_MELS // 8) * (MAX_FRAMES // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Training utilities
# ============================================================

def compute_class_weights(y_train, num_classes, device):
    """
    Compute class weights to reduce the effect of class imbalance.
    """

    counts = np.bincount(y_train, minlength=num_classes)
    counts = np.maximum(counts, 1)

    weights = len(y_train) / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights


def train_one_fold(X_train, y_train, X_test, y_test, num_classes, device):
    """
    Train and evaluate the CNN for one fold.
    """

    train_dataset = MelDataset(X_train, y_train)
    test_dataset = MelDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SmallMelCNN(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(
        y_train=y_train,
        num_classes=num_classes,
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            references.extend(batch_y.numpy())

    accuracy = accuracy_score(references, predictions)
    macro_f1 = f1_score(
        references,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        references,
        predictions,
        zero_division=0
    )

    return accuracy, macro_f1, report, model


# ============================================================
# Results saving
# ============================================================

def save_results(
    raw_class_counts,
    evaluated_class_counts,
    removed_classes,
    accuracies,
    macro_f1s,
    fold_reports,
    n_splits
):
    """
    Save a clear results file for the CNN experiment.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Experiment: CNN on Log-Mel Spectrograms\n")
        f.write("=======================================\n\n")

        f.write("Description:\n")
        f.write(
            "This experiment converts each audio file into a fixed-size log-mel "
            "spectrogram and trains a compact CNN for whistled sentence "
            "classification. The network is intentionally small because the "
            "dataset is limited.\n\n"
        )

        f.write("Input representation:\n")
        f.write(f"- Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"- Number of mel bins: {N_MELS}\n")
        f.write(f"- Maximum number of frames: {MAX_FRAMES}\n")
        f.write(f"- Input shape: (1, {N_MELS}, {MAX_FRAMES})\n\n")

        f.write("Training configuration:\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Max epochs: {EPOCHS}\n")
        f.write(f"- Early stopping patience: {PATIENCE}\n")
        f.write(f"- Learning rate: {LEARNING_RATE}\n")
        f.write(f"- Weight decay: {WEIGHT_DECAY}\n")
        f.write("- Loss: CrossEntropyLoss with class weights\n\n")

        f.write("Cross-validation:\n")
        f.write(f"- StratifiedKFold with n_splits = {n_splits}\n")
        f.write("- Classes with fewer than 2 samples are excluded from evaluation\n\n")

        f.write("Original class distribution:\n")
        f.write(raw_class_counts.to_string())
        f.write("\n\n")

        f.write("Removed classes with fewer than 2 samples:\n")
        if len(removed_classes) > 0:
            f.write(removed_classes.to_string())
        else:
            f.write("None")
        f.write("\n\n")

        f.write("Evaluated class distribution:\n")
        f.write(evaluated_class_counts.to_string())
        f.write("\n\n")

        f.write("Fold results:\n")
        for index, (acc, f1) in enumerate(zip(accuracies, macro_f1s), start=1):
            f.write(f"- Fold {index}: Accuracy={acc:.4f}, Macro F1={f1:.4f}\n")

        f.write("\nFinal results:\n")
        f.write(f"Accuracy mean: {np.mean(accuracies):.4f}\n")
        f.write(f"Accuracy std : {np.std(accuracies):.4f}\n")
        f.write(f"Macro F1 mean: {np.mean(macro_f1s):.4f}\n")
        f.write(f"Macro F1 std : {np.std(macro_f1s):.4f}\n")

        f.write("\nDetailed classification reports:\n")
        for fold_id, report in fold_reports:
            f.write("\n")
            f.write(f"Fold {fold_id}\n")
            f.write("----------------------------------------\n")
            f.write(report)
            f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_STATE)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    print("Building log-mel spectrogram dataset...")

    X, y, filenames = build_dataset()

    print("\nRaw dataset:")
    print("Number of audios found:", len(X))
    print("Input shape:", X.shape)
    print("Number of classes:", len(set(y)))

    raw_class_counts = pd.Series(y).value_counts().sort_index()

    print("\nRaw class distribution:")
    print(raw_class_counts)

    X_eval, y_eval, filenames_eval, raw_counts, removed_classes = filter_classes_with_enough_samples(
        X,
        y,
        filenames,
        min_samples=2
    )

    evaluated_class_counts = pd.Series(y_eval).value_counts().sort_index()

    print("\nDataset used for evaluation:")
    print("Number of audios used:", len(X_eval))
    print("Input shape:", X_eval.shape)
    print("Number of classes:", len(set(y_eval)))

    print("\nEvaluated class distribution:")
    print(evaluated_class_counts)

    print("\nRemoved classes with fewer than 2 samples:")
    print(removed_classes if len(removed_classes) > 0 else "None")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_eval)

    num_classes = len(label_encoder.classes_)

    min_class_count = evaluated_class_counts.min()
    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "StratifiedKFold is impossible even after filtering."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    accuracies = []
    macro_f1s = []
    fold_reports = []

    best_model_state = None
    best_macro_f1 = -1.0

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_eval, y_encoded), start=1):
        print(f"\nFold {fold_id}/{n_splits}")

        X_train = X_eval[train_idx]
        X_test = X_eval[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        acc, macro_f1, report, model = train_one_fold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            device=device
        )

        accuracies.append(acc)
        macro_f1s.append(macro_f1)
        fold_reports.append((fold_id, report))

        print(f"Accuracy fold {fold_id}: {acc:.4f}")
        print(f"Macro F1 fold {fold_id}: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    print("\n=== CNN Mel-Spectrogram Cross-Validation Results ===")
    print(f"Accuracy mean: {np.mean(accuracies):.4f}")
    print(f"Accuracy std : {np.std(accuracies):.4f}")
    print(f"Macro F1 mean: {np.mean(macro_f1s):.4f}")
    print(f"Macro F1 std : {np.std(macro_f1s):.4f}")

    torch.save({
        "model_state_dict": best_model_state,
        "label_encoder_classes": label_encoder.classes_,
        "n_mels": N_MELS,
        "max_frames": MAX_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "num_classes": num_classes,
        "removed_classes": removed_classes.to_dict(),
    }, MODEL_OUT)

    save_results(
        raw_class_counts=raw_counts,
        evaluated_class_counts=evaluated_class_counts,
        removed_classes=removed_classes,
        accuracies=accuracies,
        macro_f1s=macro_f1s,
        fold_reports=fold_reports,
        n_splits=n_splits
    )

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Best model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()