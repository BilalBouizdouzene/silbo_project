# Silbo Whistled Speech Classification Experiments

This repository contains the first classification experiments for the project:

**Deep Learning for the Automatic Transcription of Whistled Spanish**

The goal of this stage is to build a clean experimental pipeline for whistled speech processing.  
Before moving to full automatic transcription, we start with a sentence classification task in order to verify whether the acoustic signal contains enough information to distinguish different whistled sentences.

---

## 1. Project Context

Whistled speech is a special communication modality where linguistic information is transmitted through a whistled signal instead of ordinary vocal speech.

In this project, we work on Spanish whistled speech, especially Silbo-related recordings. The final long-term objective is automatic transcription. However, because the dataset is small and the signal is very different from standard speech, the first step is to build reliable baselines.

The current repository focuses on:

- cleaning and preparing labels;
- extracting acoustic and neural representations;
- training classical and neural classifiers;
- comparing several classification baselines;
- preparing the code for reproducible experiments.

---

## 2. Dataset

The cleaned dataset contains:

```text
65 audio files
21 real sentence classes
```
The labels are stored in:
```text
data/metadata/labels_clean.csv
```
The mapping between class IDs and sentence transcriptions is stored in:
```text
data/metadata/label_mapping.csv
```
The class labels are created from the normalized phrase content, not from the filename prefix.
This is important because some files share the same numeric prefix but correspond to different sentences.

Example:
```text
01-D-2017-...  -> Claudia tiene dos libros grandes
01-JL-2018-... -> Juan encontró cinco perdices
```
These are different sentences and must not be grouped into the same class.

---

## 3. Evaluation Protocol
The cleaned dataset contains one class with only one example:
```text
P03: 1 sample
```
Since StratifiedKFold with 2 folds requires at least 2 examples per class, this class is excluded only during evaluation.

Therefore, all reported experiments use:
```text
64 audio files
20 evaluable classes
```
The full metadata file is kept complete, but the evaluation scripts automatically filter classes with fewer than 2 examples.

Cross-validation protocol:
```text
StratifiedKFold
n_splits = 2
shuffle = True
random_state = 42
```
Main metrics:
```text
Accuracy
Macro F1-score
```
Macro F1 is especially important because the classes are imbalanced.

---

## 4. Label Preparation
The label cleaning script is:
```text
src/data/create_labels_clean.py
```
It produces:
```text
data/metadata/labels_clean.csv
data/metadata/label_mapping.csv
```
The script normalizes small transcription inconsistencies, such as:

secondary stress marks;
small IPA variants;
accidental annotation artifacts;
inconsistent phrase-level transcriptions.

The labels are created based on the normalized IPA phrase.

To regenerate the labels:
```text
python src/create_labels.py
python src/data/create_labels_clean.py
```
Expected output:
```text
Number of audio files: 65
Number of classes: 21
No suspicious label issues detected
```

---

# 5. Experiments
The following experiments were implemented and evaluated.
---
## Experiment 1 — MFCC + Delta + Delta-Delta + SVM

Script:
```text
src/models/classical/mfcc_svm_cv.py
```
Feature extraction:
```text
MFCC coefficients: 20
Delta MFCC
Delta-delta MFCC
Statistics: mean + standard deviation over time
Final feature dimension: 120
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
Linear SVM, C=1
```
Results:
```text
Accuracy mean : 0.6250
Accuracy std  : 0.0000
Macro F1 mean : 0.6487
Macro F1 std  : 0.0196
```
This is the best result among the current experiments.

## Experiment 2 — Whisper Encoder + SVM

Script:
```text
src/models/transformers/whisper_encoder_svm_cv.py
```
Model:
```text
openai/whisper-tiny
```
Feature extraction:
```text
Whisper encoder hidden states
Mean pooling over time
Embedding dimension: 384
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
Linear SVM, C=1
```
Results:
```text
Accuracy mean : 0.5469
Accuracy std  : 0.0156
Macro F1 mean : 0.5475
Macro F1 std  : 0.0142
```
Whisper embeddings are useful, but they do not outperform the MFCC-based baseline on the current dataset.

## Experiment 3 — wav2vec2 Encoder + SVM

Script:
```text
src/models/transformers/wav2vec2_encoder_svm_cv.py
```
Model:
```text
facebook/wav2vec2-base-960h
```
Feature extraction:
```text
wav2vec2 hidden states
Mean pooling over time
Embedding dimension: 768
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
Linear SVM, C=1
```
Results:
```text
Accuracy mean : 0.4219
Accuracy std  : 0.1094
Macro F1 mean : 0.3953
Macro F1 std  : 0.1164
```
wav2vec2 performs weaker than the other representations in this setup. This suggests that its pretrained representation does not directly transfer well to whistled speech without adaptation.

## Experiment 4 — Fusion MFCC + Whisper Encoder + SVM

Script:
```text
src/models/transformers/fusion_mfcc_whisper_svm_cv.py
```
Feature extraction:
```text
MFCC + delta + delta-delta: 120 dimensions
Whisper encoder embedding: 384 dimensions
Total feature dimension: 504
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
Linear SVM, C=1
```
Results:
```text
Accuracy mean : 0.5625
Accuracy std  : 0.0000
Macro F1 mean : 0.5763
Macro F1 std  : 0.0187
```
The fusion improves over Whisper alone, but it still does not outperform the MFCC + delta + delta-delta baseline.

## Experiment 5 — HuBERT Encoder + SVM

Script:
```text
src/models/transformers/hubert_encoder_svm_cv.py
```
Model:
```text
facebook/hubert-base-ls960
```
Feature extraction:
```text
HuBERT hidden states
Mean pooling over time
Embedding dimension: 768
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
Linear SVM, C=1
```
Results:
```text
Accuracy mean : 0.5312
Accuracy std  : 0.0938
Macro F1 mean : 0.5164
Macro F1 std  : 0.0962
```
HuBERT performs better than wav2vec2 but remains below Whisper and the MFCC-based baseline.

## Experiment 6 — CNN on Log-Mel Spectrograms

Script:
```text
src/models/deep/cnn_melspec_cv.py
```
Input representation:
```text
Log-mel spectrogram
Sample rate: 16000 Hz
Number of mel bins: 64
Maximum frames: 256
Input shape: (1, 64, 256)
```
Training configuration:
```text
Batch size: 8
Max epochs: 60
Early stopping patience: 10
Learning rate: 1e-3
Weight decay: 1e-4
Loss: CrossEntropyLoss with class weights
```
Results:
```text
Accuracy mean : 0.5625
Accuracy std  : 0.0312
Macro F1 mean : 0.5255
Macro F1 std  : 0.0262
```
The CNN gives reasonable accuracy, but it does not outperform the MFCC-based baseline. This is expected because the dataset is still very small for training a neural model from scratch.

## Experiment 7 — Strict Train-Only Data Augmentation + MFCC + SVM

Script:
```text
src/models/classical/mfcc_svm_augmentation_strict_cv.py
```
This is the official augmentation experiment.

Protocol:
```text
1. Split original audios into train/test folds.
2. Apply augmentation only to the training fold.
3. Keep the test fold original and non-augmented.
```
This avoids data leakage.

Augmentations applied to training audio only:
```text
original
noise_0005
noise_0010
amp_080
amp_120
stretch_095
stretch_105
pitch_minus1
pitch_plus1
```
Feature extraction:
```text7
MFCC + delta + delta-delta
Statistics: mean + standard deviation
Feature dimension: 120
```
SVM configurations tested:
```text
RBF SVM, C=0.1, gamma=scale
RBF SVM, C=1, gamma=scale
RBF SVM, C=10, gamma=scale
RBF SVM, C=100, gamma=scale
Linear SVM, C=1
Linear SVM, C=10
```
Best configuration:
```text
RBF SVM, C=100, gamma=scale
```
Results:
```text
Accuracy mean : 0.6094
Accuracy std  : 0.0156
Macro F1 mean : 0.6308
Macro F1 std  : 0.0275
```
This result is close to the best MFCC baseline, but it does not outperform it. This suggests that augmentation is promising but must be tuned carefully, because some transformations may alter important acoustic cues in whistled speech.
---
# 6. Main Observations

The strongest baseline is the classical MFCC-based system with dynamic features.

This is an important result because the dataset is small. In low-resource conditions, classical acoustic features can be more robust than large pretrained speech models.

Whisper embeddings perform better than wav2vec2 and HuBERT in the current setup, but they still do not outperform MFCC dynamic features.

The fusion of MFCC and Whisper improves over Whisper alone, but not over MFCC alone.

The CNN on log-mel spectrograms gives reasonable accuracy, but it likely needs more data to become competitive.

The strict data augmentation experiment is close to the best baseline, but it does not improve it yet. More careful augmentation strategies may be needed.
---
# 7. Data Leakage Note

A previous naive augmentation experiment was removed from the main experiments because it could introduce data leakage.

The problem with naive augmentation is that augmented versions of the same original audio can appear in both training and test folds. This makes the evaluation artificially optimistic.

The current repository keeps only the strict augmentation protocol:
```text
Split original audios first
Then augment training audios only
Never augment test audios
```
This is the scientifically correct protocol for evaluating data augmentation.
---
# 8. Next Steps

The next steps are:

1. Continue annotating more audio files.
2. Increase the number of examples per sentence class.
3. Test word-level or syllable-level classification.
4. Explore transcription-oriented approaches.
5. Fine-tune pretrained models on whistled speech if enough data becomes available.
6. Compare phrase-level classification with word-level recognition.
7. Prepare a cleaner benchmark split once more data is available.

The current repository represents the first clean and reproducible baseline stage of the project.