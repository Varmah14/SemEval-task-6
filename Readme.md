# 🧠 CLARITY: Detecting Ambiguity and Evasion in Political Discourse

This repository contains our team's implementation for the **CLARITY Shared Task**, which focuses on detecting **response ambiguity and evasion techniques** in political discourse (e.g., presidential debates and interviews).  
The goal is to classify whether a politician’s answer is _clear_, _ambiguous_, or a _non-reply_, and further identify specific **evasion strategies**.

---

## 👥 Team Members

| Name                     | Institution                    |
| ------------------------ | ------------------------------ |
| **Mahendra Varma Vaddi** | University of Colorado Boulder |
| **Shivani Madan**        | University of Colorado Boulder |
| **Anirudh Kakati**       | University of Colorado Boulder |

---

## 📘 Overview

Our project aims to:

1. Detect **clarity level** in Q/A pairs (Clear Reply, Ambiguous, Clear Non-Reply).
2. Detect **evasion technique** (nine fine-grained categories).
3. Explore hierarchical multitask modeling for improved interpretability.

We build upon:

- The [QEvasion Dataset (Thomas et al., 2024)](https://huggingface.co/datasets/ailsntua/QEvasion)
- Transformer-based models (RoBERTa, DeBERTa)
- Macro-F1 evaluation for fair multi-class scoring

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Varmah14/SemEval-task-6.git
```
