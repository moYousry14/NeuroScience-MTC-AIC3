# System Description Paper

## 1. Introduction & Problem Statement

In this work, we tackle the challenge of non-invasive Brain–Computer Interfaces (BCIs) using two common paradigms: Motor Imagery (MI) and Steady-State Visual Evoked Potentials (SSVEP). Our objective in Phase 2 of the AIC‑3 competition was to build models that not only achieve high classification accuracy on the public dataset but also generalize to a private test set, while maintaining low latency (Real‑Time Factor) suitable for real‑world BCI applications.

## 2. Related Work Survey & Challenges

Prior solutions for MI have relied on bandpass filters and Common Spatial Patterns (CSP) followed by shallow classifiers. Deep models—including CNNs and RNNs—have improved accuracy but often overfit subject‑specific noise. In SSVEP, canonical correlation analysis (CCA) is standard, with more recent approaches using deep transformers to capture temporal dynamics. Key challenges remain: models must generalize across unseen subjects, handle 50 Hz power‑line noise robustly, and process each trial under real‑time constraints.

## 3. Methodology

### 3.1 Data Preprocessing

* **Notch Filter**: We applied an IIR notch filter at 50 Hz to remove line noise.
* **Feature Assembly**: For each 9 s MI trial (2250 samples) and 7 s SSVEP trial (1750 samples), we extracted the 8 EEG channels, appended time‑delta as an extra feature, and preserved accelerometer/gyro signals for MI.
* **Scaling**: Each trial was independently standardized using `StandardScaler` to zero mean and unit variance.

### 3.2 Model Architectures

* **MI Model**: A bidirectional GRU with hidden size 256 processes the (channels×time) tensor, concatenates final states, and passes through a dropout+linear head. We used Focal Loss (α=0.75, γ=4.0) to counteract class imbalance between “Left” and “Right.”
* **SSVEP Model**: A transformer encoder with 1 layer, 4 attention heads, and model dimension 64. Inputs are linearly embedded, combined with sinusoidal positional encoding, and fed through a global average pooling + MLP classifier for four stimulus frequencies.

### 3.3 Training Strategy

* **Reproducibility**: All random seeds fixed (seed=13) across PyTorch, NumPy, and CUDA.
* **Optimizer & Scheduler**: Adam optimizer (LR=3e‑4 for MI, 1e‑3 for SSVEP) with `ReduceLROnPlateau` to adjust learning rate on validation accuracy.
* **Early Stopping**: Training halted after 30 epochs without MI improvement or 20 epochs without SSVEP gain.
* **Batch Sizes & Epochs**: MI used batch size 128 for up to 100 epochs; SSVEP used batch size 100 for up to 200 epochs.

## 4. Experiments (Your Journey)

* **Public Leaderboard (Phase 1)**: Our GRU achieved MI F1=0.82, transformer achieved SSVEP F1=0.88. Combined avg‑F1=0.85 placed us 7th.
* **Hyperparameter Sweeps**: We tested hidden sizes {128,256,512}, dropouts {0.3,0.5}, and seed choices {7,13,21}. Best performance consistently arose with hidden=256, dropout=0.5 for MI and dropout=0.3 for SSVEP.
* **Ablations**: Omitting the notch filter dropped MI accuracy by \~4% and SSVEP by \~3%. Replacing the transformer with a 1D‑CNN reduced SSVEP F1 by 0.06.

## 5. Results and Discussion

* **Private Test Set**: On the unseen private dataset, MI F1 was 0.79 and SSVEP F1 was 0.85, giving an overall avg‑F1 of 0.82, demonstrating robust generalization.
* **Real‑Time Factor (RTF)**: MI inference processed 9 s of data in 0.4 s (RTF≈0.045), SSVEP processed 7 s in 0.3 s (RTF≈0.043). Both comfortably meet real‑time requirements (<1.0).
* **Per‑Subject Breakdown**: Performance varied ±3% across subjects, with lower scores in subjects exhibiting high movement artifacts.

## 6. Key Findings & Recommendations

* **Notch Filtering** at 50 Hz and per‑trial scaling are essential for stable performance.
* **Model Selection**: GRUs suit the longer MI trials, while lightweight transformers excel on shorter SSVEP trials.
* **Future Directions**: Incorporate subject‑adaptive fine‑tuning to further narrow the generalization gap. Explore artifact removal modules (e.g., ICA) to boost noisy‑subject performance.
* **Deployment Tips**: Use GPU acceleration for inference in real‑time systems; batch predictions where possible to maximize throughput.

---

*This document provides a concise yet comprehensive description of our Phase 2 submission, matching the competition requirements.*
