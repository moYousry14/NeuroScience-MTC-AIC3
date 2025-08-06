# AIC-3 BCI Final Submission

**Deadline:** July 24, 2025, 11:45 PM (Cairo Time)

## Repository Structure

```
NEUROSCIENCE/
├── MI/
│   ├── MI-SRC.py            # Combined MI train + inference script
│   ├── mi_model_seed13.pt   # Trained MI model checkpoint
│   └── mi_submission.csv    # MI test predictions
├── SSVEP/
│   ├── SSVEP-SRC.py         # Combined SSVEP train + inference script
│   ├── ssvep_model_seed13.pt# Trained SSVEP model checkpoint
│   └── ssvep_submission.csv # SSVEP test predictions
├── requirements.txt         # Project dependencies
└── submission.csv           # (Optional) merged MI + SSVEP submissions
```

## Setup

1. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Place the dataset** in `./mtc-aic3_dataset/` matching this layout:

   ```
   mtc-aic3_dataset/
   ├── train.csv
   ├── validation.csv
   ├── MI/
   │   ├── train/
   │   ├── validation/
   │   └── test/
   └── SSVEP/
       ├── train/
       ├── validation/
       └── test/
   ```

## Usage

Each of the two pipelines (MI and SSVEP) supports both training and inference in a single script.

### Motor Imagery (MI)

* **Train & Validate**

  ```bash
  python MI/MI-SRC.py --mode train --data_dir ./mtc-aic3_dataset --seed 13
  ```

  * Loads MI data, trains the GRU model, and saves `MI/mi_model_seed13.pt`.

* **Inference**

  ```bash
  python MI/MI-SRC.py --mode inference --data_dir ./mtc-aic3_dataset --checkpoint MI/mi_model_seed13.pt
  ```

  * Generates `MI/mi_submission.csv` with columns `id,label`.

### SSVEP

* **Train & Validate**

  ```bash
  python SSVEP/SSVEP-SRC.py --mode train --data_dir ./mtc-aic3_dataset --seed 13
  ```

  * Loads SSVEP data, trains the Transformer model, and saves `SSVEP/ssvep_model_seed13.pt`.

* **Inference**

  ```bash
  python SSVEP/SSVEP-SRC.py --mode inference --data_dir ./mtc-aic3_dataset --checkpoint SSVEP/ssvep_model_seed13.pt
  ```

  * Generates `SSVEP/ssvep_submission.csv` with columns `id,label`.

## Merging Submissions

If a single `submission.csv` is required, concatenate the two outputs:

```bash
cat MI/mi_submission.csv SSVEP/ssvep_submission.csv > submission.csv
```

## Dependencies

Install these packages (latest versions will be used):

```
torch
numpy
pandas
scipy
scikit-learn
tqdm
```

```bash
pip install -r requirements.txt
```

## Notes

* Random seeds fixed (`--seed 13`) for reproducibility.
* Scripts print progress and save best checkpoints automatically.
* Ensure your `data_dir` matches the structure above.

---

*Validate each step in a fresh environment to guarantee reproducibility.*

