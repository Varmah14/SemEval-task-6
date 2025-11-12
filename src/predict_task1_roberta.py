import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from src.data import get_text_pair

@torch.no_grad()
def main(model_dir="out/task1-roberta", out_csv="out/task1_roberta_test_preds.csv", batch_size=16):
    ds_test = load_dataset("ailsntua/QEvasion")["test"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    def make_batches(ix, n):
        for i in range(0, n, batch_size):
            yield ix[i:i+batch_size]

    texts = [get_text_pair(r) for r in ds_test]
    ids = list(range(len(texts)))
    preds = []

    for batch_ix in make_batches(ids, len(ids)):
        q = [texts[i][0] for i in batch_ix]
        a = [texts[i][1] for i in batch_ix]
        enc = tok(q, a, return_tensors="pt", truncation=True, padding=True, max_length=512)
        logits = model(**enc).logits
        p = logits.argmax(-1).cpu().numpy().tolist()
        preds.extend(p)

    id2label = model.config.id2label
    labels = [id2label[int(i)] for i in preds]
    df = pd.DataFrame({"id": ids, "prediction": labels})
    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="out/task1-roberta")
    parser.add_argument("--out_csv", type=str, default="out/task1_roberta_test_preds.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(**vars(args))
