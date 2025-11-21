import argparse
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from src.data import get_text_pair, label_maps_evasion


@torch.no_grad()
def main(
    model_dir: str = "out/task2-roberta",
    seed: int = 42,
    valid_size: float = 0.15,
    batch_size: int = 16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Loading Task 2 model from: {model_dir}")
    print(f"🔹 Seed = {seed}, valid_size = {valid_size}, batch_size = {batch_size}")

    # ---------- 1. Load dataset & split same as training ----------
    print("📥 Loading ailsntua/QEvasion dataset...")
    ds_all = load_dataset("ailsntua/QEvasion")
    split = ds_all["train"].train_test_split(test_size=valid_size, seed=seed)
    train_ds, valid_ds = split["train"], split["test"]
    print(f"✅ Train size: {len(train_ds)}, raw valid size: {len(valid_ds)}")

    # Label maps (based on train)
    label2id, id2label = label_maps_evasion(train_ds)
    print(f"✅ Number of evasion classes: {len(label2id)}")
    print(f"Classes: {list(label2id.keys())}")

    # ---------- 2. Load tokenizer + model ----------
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    model.eval()
    print("✅ Model loaded.")

    # ---------- 3. Build evaluation lists ----------
    Xq, Xa, y_true = [], [], []

    for ex in valid_ds:
        # Some examples may not have evasion_label -> skip them
        if not ex.get("evasion_label"):
            continue
        q, a = get_text_pair(ex)
        Xq.append(q)
        Xa.append(a)
        y_true.append(ex["evasion_label"])

    print(f"✅ Valid examples with evasion_label: {len(y_true)}")
    if len(y_true) == 0:
        print(
            "⚠️ No validation examples with evasion_label found. Check dataset / split."
        )
        return

    # ---------- 4. Run model in batches ----------
    preds_ids = []

    for i in range(0, len(Xq), batch_size):
        batch_q = Xq[i : i + batch_size]
        batch_a = Xa[i : i + batch_size]

        enc = tok(
            batch_q,
            batch_a,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        preds_ids.extend(batch_preds)

    # Map ids -> label strings
    y_pred = [id2label[int(i)] for i in preds_ids]

    # ---------- 5. Metrics ----------
    print("\n=== Task 2: Evasion-level classification on validation split ===\n")
    print(classification_report(y_true, y_pred, digits=4))

    labels_sorted = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="out/task2-roberta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(**vars(args))
