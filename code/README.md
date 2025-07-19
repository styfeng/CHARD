# Project Overview

This repository contains scripts and tools for training, evaluating, and analyzing **BART** and **T5** models across various tasks and datasets.

---

## 🛠 Setup Instructions

### 🔧 Environment Setup for BART and T5 Training + Inference

Create a Conda environment with the necessary packages:

```bash
conda create -n T5_training python=3.8.5
conda activate T5_training

# Optional but useful
conda install -c conda-forge tmux

# Core dependencies
pip3 install torch torchvision torchaudio
pip install scipy nltk

# REQUIRED for FP16 T5-large training
pip install git+https://github.com/huggingface/transformers@t5-fp16-no-nans

# Evaluation libraries
pip install rouge_score
pip install bert_score==0.3.8
```

### 📦 Additional (Potentially Required) Packages

Install these only if there are no conflicts:

```bash
pip install datasets
pip install sentencepiece==0.1.95
pip install spacy==2.1.0
pip install gitpython

python -m spacy download en
python -m spacy validate
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz
```

> ⚠️ Check individual scripts (especially in `BART-T5-scripts/`) for additional dependencies.

---

### ⚙️ UDA Backtranslation Setup

UDA backtranslation works best in Python 2 and has separate requirements.  
See `UDA_backtrans/main_notebook_UDA_backtranslation.ipynb` for full setup instructions.

---

### 🧪 CommonGen Eval Environment Setup

`commongen_eval` scripts require Python 2 and specific dependencies.  
Create a separate environment as follows:

```bash
conda create -n coco_score python=2.7
conda activate coco_score

pip install numpy
pip install -U spacy
python -m spacy download en_core_web_sm
bash get_stanford_models.sh
```

These instructions are based on the CommonGen repo:  
📎 https://github.com/INK-USC/CommonGen/tree/master/evaluation/Traditional

---

## 📁 `BART-T5-scripts/`

This folder contains scripts related to:
- Training BART and T5 models
- Running inference
- Evaluating model outputs

---

### 🔧 `training/` – Model Training

This subfolder includes scripts for training BART and T5. Each script contains inline comments explaining every argument in detail.

#### Example Commands

```bash
# Train BART model
python bart_training.py base 3e-05 42 32 128 combined_baseline \
  HF_training_data/combined/combined_train_HF.json \
  HF_training_data/combined/combined_val_HF.json 32 30 0

# Train T5 model
python t5_training.py base 1e-05 42 35 128 combined_2x_temp0.4 \
  HF_training_data/augmented/2x/combined_train_temp0.4_2x.json \
  HF_training_data/combined/combined_val_HF.json 64 25 0
```

#### `loop_training.sh`

This script trains a model over multiple learning rates in one go.

```bash
bash loop_training.sh t5 large 42 35 128 combined_diff-temps_3x \
  HF_training_data/augmented/diff-temps-t5/combined_train_diff-temps-t5_3x.json \
  HF_training_data/combined/combined_val_HF.json 16 12 0
```

You’ll be prompted to enter a list of learning rates, e.g.:

```text
1e-05 5e-05 1e-04 5e-04 1e-06 5e-06 1e-03 5e-03 5e-07 1e-02
```

---

### 🔍 `inference_evaluation/` – Inference + Evaluation

This subfolder includes scripts for running inference with trained models and computing various evaluation metrics.

#### Key Scripts

- `bart_inference_evaluate.py`
- `t5_inference_evaluate.py`

These scripts:
- Load a model and perform beam search decoding.
- Compute metrics: **ROUGE, BERTScore, PPL, Length, TTR, UTR**.
- Save:
  - A `.json` file with inputs, ground-truths, and model outputs
  - A `.txt` file with model outputs (one per line)
  - A per-example metrics file
  - A summary file with overall metrics

##### Sample Usage

```bash
python bart_inference_evaluate.py base 42 32 128 risk-factor_baseline \
  HF_training_data/risk-factor/risk-factor_test_HF.json 3e-03 40 32 0
```

#### `inference_three_test_splits.sh`

There are four variants for different test subsets. These scripts evaluate models across:
- Full test set
- Seen test split
- Unseen test split

##### Sample Usage

```bash
bash BART-T5-scripts/inference_three_test_splits_prevention.sh t5 base \
  42 35 128 prevention_baseline 3e-03 12 32 0
```

#### Other Evaluation Tools

- `evaluate_from_file.py`: Evaluate metrics from a `.json` file of saved generations.
- `eval_diversity_PPL_from_json_file.py` / `eval_diversity_PPL_from_txt_file.py`: Compute **diversity** and **PPL** when there is no ground truth.
- `ROUGE_BERTScore_from_files.py`: Compare ROUGE and BERTScore across two generation files.

---

## 📁 `commongen_eval/`

This folder supports evaluation of additional metrics:
- **BLEU, METEOR, CIDEr, SPICE**

The scripts rely on the [CommonGen evaluation toolkit](https://github.com/INK-USC/CommonGen).

### Setup

1. Clone the CommonGen repo.
2. Navigate to:  
   ```
   evaluation/Traditional/eval_metrics
   ```
3. Paste the scripts from `commongen_eval` here.
4. Create a subfolder named `Accenture_data` containing:
   - Input files (e.g., `combined_test_HF_inputs.txt`)
   - Ground truth files (e.g., `combined_test_HF_outputs.txt`)
5. Optionally, create additional subfolders for model outputs.

### Evaluation Scripts

- `eval_individual_X.py`: Computes metric **X** per example (useful for statistical tests).
- `loop_scripts/`: Runs all metrics over a folder of `.txt` files using the above scripts.

---

## 📁 `data_processing/`

This folder contains scripts for:
- Preprocessing
- Postprocessing
- General data transformation

Each subfolder corresponds to a specific processing goal (names are self-explanatory).  
Please refer to each script for details, as usage and purpose are commented at the top of each file.

---

## 📁 `UDA_backtrans/`

Contains code and instructions for **UDA backtranslation-based data augmentation**, adapted from the official [UDA repo](https://github.com/google-research/uda).

### Key Resource

- `main_notebook_UDA_backtranslation.ipynb`:  
  A comprehensive notebook (based on Colab) that:
  - Provides setup instructions
  - Runs backtranslation
  - Guides you through the full process

Refer to the notebook for detailed steps and required environment setup.
