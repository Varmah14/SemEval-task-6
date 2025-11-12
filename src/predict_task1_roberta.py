import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from src.data import get_text_pair, prepare_qevasion


@torch.no_grad()
def main(
    model_dir="out/task1-roberta",
    out_csv="out/task1_roberta_test_preds.csv",
    batch_size=16,
    split="test",
    seed=42,
    valid_size=0.15,
):

    # ---- Load model/tokenizer + device ----
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ---- Pick dataset split ----
    if split in {"train", "valid"}:
        train, valid = prepare_qevasion(
            valid_size=valid_size, seed=seed, revision="main"
        )
        ds = train if split == "train" else valid
        # keep only rows with label (for clarity task)
        ds = ds.filter(
            lambda ex: ex.get("clarity_label") is not None and ex["clarity_label"] != ""
        )
    elif split == "test":
        ds = load_dataset("ailsntua/QEvasion")["test"]
    else:
        raise ValueError("--split must be one of: train, valid, test")

    # ---- Build ids & text pairs ----
    has_id = "id" in ds.column_names
    ids = ds["id"] if has_id else list(range(len(ds)))
    texts = [get_text_pair(row) for row in ds]  # (question, answer)

    # ---- Batch inference ----
    def make_batches(ix, n):
        for i in range(0, n, batch_size):
            yield ix[i : i + batch_size]

    preds = []
    for batch_ix in make_batches(list(range(len(ids))), len(ids)):
        q = [texts[i][0] for i in batch_ix]
        a = [texts[i][1] for i in batch_ix]
        enc = tok(
            q, a, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p = logits.argmax(-1).cpu().numpy().tolist()
        preds.extend(p)

    # ---- Map to string labels via model config ----
    id2label = model.config.id2label
    labels = [id2label[int(i)] for i in preds]

    # ---- Save ----
    df = pd.DataFrame({"id": ids, "prediction": labels})
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} predictions for split='{split}' to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="out/task1-roberta")
    parser.add_argument(
        "--out_csv", type=str, default="out/task1_roberta_test_preds.csv"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "valid", "test"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.15)
    args = parser.parse_args()
    main(**vars(args))
