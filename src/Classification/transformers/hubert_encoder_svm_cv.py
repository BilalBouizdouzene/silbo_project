import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
import torch

from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# General configuration
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_05_hubert_encoder_svm"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

EMBEDDINGS_OUT = "data/processed/embeddings/hubert_base_embeddings.npz"
MODEL_OUT = "models/hubert_encoder_svm/hubert_encoder_svm.joblib"

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
RANDOM_STATE = 42


# ============================================================
# Audio loading
# ============================================================

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and resample it to 16 kHz.

    HuBERT expects 16 kHz input audio. We use librosa to keep
    preprocessing consistent with the other experiments.
    """

    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


# ============================================================
# HuBERT embedding extraction
# ============================================================

def extract_hubert_embedding(audio, feature_extractor, model, device) -> np.ndarray:
    """
    Extract one fixed-size embedding from HuBERT.

    Workflow:
    audio waveform
        -> HuBERT feature extractor
        -> HuBERT encoder
        -> hidden states
        -> mean pooling over time

    The final output is one embedding vector per audio file.
    """

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # Shape: [batch, time, hidden_dim]
    hidden_states = outputs.last_hidden_state

    # Mean pooling over time
    embedding = hidden_states.mean(dim=1).squeeze(0)

    return embedding.cpu().numpy()


def build_embeddings():
    """
    Read labels_clean.csv and extract HuBERT embeddings for all available audios.

    Returns
    -------
    X : np.ndarray
        Embedding matrix.
    y : np.ndarray
        Label IDs.
    filenames : np.ndarray
        Filenames corresponding to the embeddings.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    print(f"Loading HuBERT model: {HUBERT_MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    model = HubertModel.from_pretrained(HUBERT_MODEL_NAME)

    model.to(device)
    model.eval()

    X = []
    y = []
    filenames = []
    missing_files = []
    failed_files = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HuBERT embeddings"):
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue

        try:
            audio = load_audio(audio_path, sr=SAMPLE_RATE)
            embedding = extract_hubert_embedding(
                audio=audio,
                feature_extractor=feature_extractor,
                model=model,
                device=device
            )

            X.append(embedding)
            y.append(label)
            filenames.append(filename)

        except Exception as error:
            failed_files.append((audio_path, str(error)))

    if missing_files:
        print("\nMissing audio files:")
        for path in missing_files:
            print("-", path)

    if failed_files:
        print("\nFiles with embedding extraction errors:")
        for path, error in failed_files:
            print(f"- {path}: {error}")

    return np.array(X), np.array(y), np.array(filenames)


# ============================================================
# Embedding cache
# ============================================================

def load_or_build_embeddings():
    """
    Load existing HuBERT embeddings if available, otherwise compute them.

    Important:
    If labels_clean.csv changes, delete the old .npz file before running
    this script. The script prints the number of loaded samples so old
    caches can be detected easily.
    """

    os.makedirs(os.path.dirname(EMBEDDINGS_OUT), exist_ok=True)

    if os.path.exists(EMBEDDINGS_OUT):
        print("Loading existing HuBERT embeddings:", EMBEDDINGS_OUT)

        data = np.load(EMBEDDINGS_OUT, allow_pickle=True)

        X = data["X"]
        y = data["y"]
        filenames = data["filenames"]

        print("Loaded embeddings:", len(X))
        return X, y, filenames

    print("No saved HuBERT embeddings found. Extracting embeddings...")

    X, y, filenames = build_embeddings()

    np.savez(
        EMBEDDINGS_OUT,
        X=X,
        y=y,
        filenames=filenames
    )

    print("HuBERT embeddings saved:", EMBEDDINGS_OUT)

    return X, y, filenames


# ============================================================
# Dataset filtering
# ============================================================

def filter_classes_with_enough_samples(X, y, filenames, min_samples=2):
    """
    Remove classes with fewer than min_samples examples.

    StratifiedKFold requires at least one sample of each class in each fold.
    With n_splits=2, every evaluated class must have at least 2 samples.

    The full labels_clean.csv remains unchanged. The filtering only affects
    the evaluation subset.
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
# Cross-validation
# ============================================================

def run_cross_validation(X, y):
    """
    Evaluate several SVM configurations using StratifiedKFold.

    The goal is to compare simple SVM configurations on top of frozen
    HuBERT embeddings and select the best one using Macro F1.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "StratifiedKFold is impossible even after filtering. "
            "At least one remaining class has fewer than 2 samples."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    svm_configs = [
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10, "gamma": "scale"},
        {"kernel": "rbf", "C": 100, "gamma": "scale"},
        {"kernel": "linear", "C": 1, "gamma": "scale"},
        {"kernel": "linear", "C": 10, "gamma": "scale"},
    ]

    all_results = []
    best_result = None

    for config in svm_configs:
        fold_accuracies = []
        fold_macro_f1s = []
        fold_reports = []

        print("\n========================================")
        print("Testing configuration:", config)
        print("========================================")

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_encoded[train_idx]
            y_test = y_encoded[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=config["kernel"],
                    C=config["C"],
                    gamma=config["gamma"],
                    class_weight="balanced"
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

            fold_accuracies.append(accuracy)
            fold_macro_f1s.append(macro_f1)

            labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
            target_names = label_encoder.inverse_transform(labels_present)

            report = classification_report(
                y_test,
                y_pred,
                labels=labels_present,
                target_names=target_names,
                zero_division=0
            )

            fold_reports.append((fold_id, report))

            print(f"Fold {fold_id}/{n_splits}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Macro F1 : {macro_f1:.4f}")

        result = {
            "config": config,
            "accuracy_mean": float(np.mean(fold_accuracies)),
            "accuracy_std": float(np.std(fold_accuracies)),
            "macro_f1_mean": float(np.mean(fold_macro_f1s)),
            "macro_f1_std": float(np.std(fold_macro_f1s)),
            "reports": fold_reports,
        }

        all_results.append(result)

        print("\nConfiguration summary:")
        print(f"Accuracy mean : {result['accuracy_mean']:.4f}")
        print(f"Accuracy std  : {result['accuracy_std']:.4f}")
        print(f"Macro F1 mean : {result['macro_f1_mean']:.4f}")
        print(f"Macro F1 std  : {result['macro_f1_std']:.4f}")

        if best_result is None or result["macro_f1_mean"] > best_result["macro_f1_mean"]:
            best_result = result

    return all_results, best_result, label_encoder, n_splits


# ============================================================
# Final model training
# ============================================================

def train_final_model(X, y, best_config):
    """
    Train the final model on all evaluable samples using the best SVM config.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=best_config["kernel"],
            C=best_config["C"],
            gamma=best_config["gamma"],
            class_weight="balanced"
        ))
    ])

    final_pipeline.fit(X, y_encoded)

    return final_pipeline, label_encoder


# ============================================================
# Save results
# ============================================================

def save_results(
    all_results,
    best_result,
    raw_class_counts,
    evaluated_class_counts,
    removed_classes,
    n_splits,
    embedding_dim
):
    """
    Save a transparent and readable results file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Experiment: HuBERT Encoder + SVM\n")
        f.write("================================\n\n")

        f.write("Description:\n")
        f.write(
            "This experiment uses HuBERT as a frozen feature extractor. "
            "For each audio file, the hidden states are mean-pooled over time "
            "to obtain one fixed-size embedding. A Support Vector Machine is then "
            "trained on top of these embeddings for whistled sentence classification.\n\n"
        )

        f.write("Embedding extraction:\n")
        f.write(f"- HuBERT model: {HUBERT_MODEL_NAME}\n")
        f.write(f"- Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"- Embedding dimension: {embedding_dim}\n")
        f.write("- Pooling: mean pooling over encoder time frames\n\n")

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

        f.write("All tested SVM configurations:\n")
        for result in all_results:
            f.write("\n")
            f.write(f"Config: {result['config']}\n")
            f.write(f"Accuracy mean: {result['accuracy_mean']:.4f}\n")
            f.write(f"Accuracy std : {result['accuracy_std']:.4f}\n")
            f.write(f"Macro F1 mean: {result['macro_f1_mean']:.4f}\n")
            f.write(f"Macro F1 std : {result['macro_f1_std']:.4f}\n")

        f.write("\n\nBest configuration:\n")
        f.write(str(best_result["config"]))
        f.write("\n")
        f.write(f"Best Accuracy mean: {best_result['accuracy_mean']:.4f}\n")
        f.write(f"Best Accuracy std : {best_result['accuracy_std']:.4f}\n")
        f.write(f"Best Macro F1 mean: {best_result['macro_f1_mean']:.4f}\n")
        f.write(f"Best Macro F1 std : {best_result['macro_f1_std']:.4f}\n")

        f.write("\n\nDetailed classification reports for the best configuration:\n")
        for fold_id, report in best_result["reports"]:
            f.write("\n")
            f.write(f"Fold {fold_id}\n")
            f.write("----------------------------------------\n")
            f.write(report)
            f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print("Loading or extracting HuBERT embeddings...")

    X, y, filenames = load_or_build_embeddings()

    print("\nRaw dataset:")
    print("Number of audios found:", len(X))
    print("Embedding dimension:", X.shape[1])
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
    print("Number of classes:", len(set(y_eval)))

    print("\nEvaluated class distribution:")
    print(evaluated_class_counts)

    print("\nRemoved classes with fewer than 2 samples:")
    print(removed_classes if len(removed_classes) > 0 else "None")

    all_results, best_result, _, n_splits = run_cross_validation(
        X_eval,
        y_eval
    )

    print("\n========================================")
    print("Best configuration")
    print("========================================")
    print(best_result["config"])
    print(f"Accuracy mean : {best_result['accuracy_mean']:.4f}")
    print(f"Accuracy std  : {best_result['accuracy_std']:.4f}")
    print(f"Macro F1 mean : {best_result['macro_f1_mean']:.4f}")
    print(f"Macro F1 std  : {best_result['macro_f1_std']:.4f}")

    final_model, label_encoder = train_final_model(
        X_eval,
        y_eval,
        best_result["config"]
    )

    joblib.dump({
        "pipeline": final_model,
        "label_encoder": label_encoder,
        "feature_type": "hubert_encoder_mean_pooling",
        "hubert_model": HUBERT_MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "removed_classes": removed_classes.to_dict(),
        "best_config": best_result["config"],
        "evaluated_labels": list(label_encoder.classes_),
    }, MODEL_OUT)

    save_results(
        all_results=all_results,
        best_result=best_result,
        raw_class_counts=raw_counts,
        evaluated_class_counts=evaluated_class_counts,
        removed_classes=removed_classes,
        n_splits=n_splits,
        embedding_dim=X.shape[1]
    )

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Final model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()