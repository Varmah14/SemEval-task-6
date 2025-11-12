import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from src.data import get_text_pair, label_maps_clarity, prepare_qevasion


@torch.no_grad()
def main(model_dir="out/task1-roberta", seed=42, valid_size=0.15, batch_size=16):
    # ---- load cached splits (same seed/valid_size as training!) ----
    train, valid = prepare_qevasion(valid_size=valid_size, seed=seed, revision="main")
    valid = valid.filter(
        lambda ex: ex.get("clarity_label") is not None and ex["clarity_label"] != ""
    )

    # ---- label maps consistent with training ----
    label2id, id2label = label_maps_clarity(train)
    num_labels = len(label2id)

    # ---- model/tokenizer + device ----
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ---- build inputs & gold labels ----
    qs, as_, y_true_ids = [], [], []
    for row in valid:
        q, a = get_text_pair(row)
        qs.append(q)
        as_.append(a)
        y_true_ids.append(label2id[row["clarity_label"]])

    # ---- batched inference ----
    preds_ids = []
    for i in range(0, len(qs), batch_size):
        enc = tok(
            qs[i : i + batch_size],
            as_[i : i + batch_size],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds_ids.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    # ---- metrics: keep ids, provide names for readability ----
    target_names = [id2label[i] for i in range(num_labels)]
    labels_order = list(range(num_labels))

    print(
        classification_report(
            y_true_ids,
            preds_ids,
            labels=labels_order,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true_ids, preds_ids, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="out/task1-roberta")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_size", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    main(**vars(args))
