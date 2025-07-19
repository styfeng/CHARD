# CHARD Dataset Files

This directory contains the data used in the paper:  
[**"CHARD: Clinical Health-Aware Reasoning Across Dimensions for Text Generation Models"**  ](https://aclanthology.org/2023.eacl-main.24/)
Work by Carnegie Mellon University and Accenture Labs

---

## 📁 `CHARDAT/` – Raw Dataset

This folder contains the core **CHARDAT** dataset, split by dimension:

- `prevention/`
- `risk-factor/`
- `treatment/`

Each subfolder includes:
- `train.txt`
- `val.txt`
- `test.txt`

Each line in these files follows the format:

```
input_ID <sep> condition <sep> dimension attribute <sep> passage containing explanation
```

- `input_ID`: A unique identifier from AMT collection (can be ignored for most uses).
- The passage contains a natural language explanation relevant to the input condition and dimension attribute.

---

## 📁 `HF_training_data/` – Model Training Data (HuggingFace Format)

This folder contains `.json` files prepared for training and evaluation with HuggingFace BART/T5 models.

### Subfolders:
- **Per-dimension folders** (`prevention/`, `risk-factor/`, `treatment/`) – for dimension-specific training.
- **`combined/`** – used to train models on all three dimensions jointly.
- **`augmented/`** – includes data augmentation variants.

#### `augmented/2x/`
- Contains 2x backtranslation-augmented training data across different temperatures.
- Other subfolders explore alternative augmentation scales and strategies (see paper for details).

---

## 📁 `input_GT-output_txt_files/` – Inputs and Ground Truth Outputs

This folder provides `.txt` files with **inputs** and **ground-truth outputs** (one per line) for all test splits:

- Full combined split
- Each individual dimension
- `test-seen` and `test-unseen` subsets

These files are used for:
- Easy inspection
- Evaluation (e.g., BLEU, METEOR, CIDEr, SPICE)

---

## 📁 `all_generation_txt_files/` – Model Generations

This folder contains **model outputs** generated from different methods and models.

### Key Subfolders:
- `humans/`: Ground-truth human-written outputs
- `retrieval/`: Outputs from Google Search retrieval-based baseline
- `best_T5_model/`: Generations from the best-performing T5-large model (used for human eval)

### Other Subfolders:
- `prevention/`, `risk-factor/`, `treatment/`, and `combined/` – model outputs split by test set:
  - `test-combined` (full)
  - `test-seen`
  - `test-unseen`

> 🔎 The `combined/` subfolder includes outputs from several models. Refer to the paper to determine the best model per type and size (e.g., BART-base, T5-large).

---

## 📁 `backtranslation_data/` – UDA Backtranslation Inputs & Outputs

This folder includes all data used for **backtranslation-based data augmentation**.

### Structure:
- **`inputs/`**:
  - `combined_train_explanations.txt`: Explanation portions to be backtranslated.
  - `combined_train_initials.txt`: Template/initial portion that is kept fixed.
- **`backtranslations/`**:
  - Contains backtranslated explanations for various temperatures and versions (e.g., 9 variations for `temp=0.9`).

> See the paper for full methodology and backtranslation usage.

