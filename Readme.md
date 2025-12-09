# CLARITY: Detecting Ambiguity and Evasion in Political Discourse

**SemEval-2026 Task 6 – LexiClarity Track**

This repository contains our team’s implementation for the **CLARITY Shared Task**, which focuses on detecting **response ambiguity and evasion techniques** in political discourse (e.g., presidential debates and interviews).

The goal is to classify whether a politician’s answer is:

- **Clear Reply**
- **Ambiguous**
- **Clear Non-Reply**

and further identify the **specific evasion strategy** used (9 fine-grained evasion categories).

This work is part of **SemEval-2026 Task 6: CLARITY – Unmasking Political Question Evasion**.

---

## Team Members

| Name                     | Institution                    |
| ------------------------ | ------------------------------ |
| **Mahendra Varma Vaddi** | University of Colorado Boulder |
| **Shivani Madan**        | University of Colorado Boulder |
| **Anirudh Kakati**       | University of Colorado Boulder |

---

## Overview

Our project aims to:

1. Detect **clarity level** in question–answer pairs
   _(Clear Reply, Ambiguous, Clear Non-Reply)_
2. Detect **evasion techniques** _(9-class classification task)_
3. Compare:

   - Feature-enhanced transformers
   - Pure transformer baselines

4. Evaluate using **Macro-F1** for fair multi-class performance

We build upon:

- **QEvasion Dataset** (Thomas et al., 2024)
- **Transformer architectures**:

  - RoBERTa + Feature Engineering
  - DeBERTa + Feature Engineering
  - ELECTRA (No Features baseline)

- **HuggingFace Transformers + PyTorch**
- **Macro-F1 as the primary evaluation metric**

---

## Dataset

We use the official **QEvasion dataset** hosted on HuggingFace:

```python
from datasets import load_dataset

ds = load_dataset("ailsntua/QEvasion")
```

Each sample consists of:

- `question`
- `interview_answer`
- `clarity_label`, `clarity_label_id`
- `evasion_label`, `evasion_label_id`

We also maintain preprocessed CSV files for training:

```
csv_files/
├── training_data.csv
├── validation_data.csv
├── FE_training_data.csv
└── FE_test_data.csv
```

---

## Models Implemented

### 1. RoBERTa + Feature Engineering

- Uses handcrafted linguistic + statistical features
- Trained separately for:

  - **Clarity**
  - **Evasion**

- Implemented using HuggingFace `Trainer`

---

### 2. DeBERTa + Feature Engineering

- Strong contextual baseline
- Same feature-enhanced setup as RoBERTa
- Often performs better on subtle ambiguity patterns

---

### 3. ELECTRA (No Features – Baseline)

A clean transformer-only baseline using:

- **`google/electra-base-discriminator`**
- No handcrafted features
- Input format:

  ```
  Question: <question> [SEP] Answer: <answer>
  ```

- Includes:

  - Class-weighted loss for imbalance
  - Mixed precision (`fp16`)
  - Early stopping
  - Macro-F1 model selection

---

## Repository Structure

```
.
├── src/
│   ├── train_task1_roberta.py
│   ├── train_task2_roberta.py
│   ├── train_task1_deberta.py
│   ├── train_task2_deberta.py
│   ├── eval_valid_task1.py
│   ├── eval_valid_task2.py
│   ├── train_electra.py
│   └── utils/
├── csv_files/
├── models/
├── results/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Varmah14/SemEval-task-6.git
cd SemEval-task-6
```

---

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

### RoBERTa (Clarity)

```bash
python3 -m src.train_task1_roberta --epochs 3 --bs 8 --lr 2e-5
```

---

<!--
### DeBERTa (Evasion)

```bash
python3 -m src.train_task2_deberta --epochs 3 --bs 8 --lr 2e-5
``` -->

---

### ELECTRA Baseline (Both Tasks)

```bash
python3 src/train_electra.py
```

Outputs:

- Trained models → `models/`
- Logs → `results/`
- Metrics CSV → `csv_files/`

---

## Evaluation

Primary Metric: **Macro-F1 Score**
Secondary Metric: Weighted-F1

Evaluation scripts:

```bash
python3 -m src.eval_valid_task1
python3 -m src.eval_valid_task2
```

---

<!--
## Results Summary (Fill Final Scores)

| Model                 | Task    | Macro-F1 | Weighted-F1 |
| --------------------- | ------- | -------- | ----------- |
| RoBERTa + FE          | Clarity | TODO     | TODO        |
| RoBERTa + FE          | Evasion | TODO     | TODO        |
| DeBERTa + FE          | Clarity | TODO     | TODO        |
| DeBERTa + FE          | Evasion | TODO     | TODO        |
| ELECTRA (No Features) | Clarity | TODO     | TODO        |
| ELECTRA (No Features) | Evasion | TODO     | TODO        |

--- -->

## 🔬 Limitations

- No debate-level context modeling yet
- Feature engineering is static
- No cross-domain generalization testing yet

---

## 🔮 Future Work

- Multi-task learning between clarity & evasion
- Speaker-aware modeling
- Cross-lingual evaluation
- Error analysis of evasion strategies

---

## 📬 Contact

**Mahendra Varma Vaddi**
MS Data Science — University of Colorado Boulder
GitHub: [https://github.com/Varmah14](https://github.com/Varmah14)
