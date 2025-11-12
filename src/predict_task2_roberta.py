import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from src.data import get_text_pair, prepare_qevasion


@torch.no_grad()
def main(
    model_dir="out/task2-roberta",
    out_csv="out/task2_roberta_test_preds.csv",
    batch_size=16,
    split="test",
    seed=42,
    valid_size=0.15,
):

    # --- Model & device ---
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # --- Pick dataset split ---
    if split in {"train", "valid"}:
        train, valid = prepare_qevasion(
            valid_size=valid_size, seed=seed, revision="main"
        )
        ds = train if split == "train" else valid
        # For evasion train/valid, keep only rows that have a label
        ds = ds.filter(
            lambda ex: ex.get("evasion_label") is not None and ex["evasion_label"] != ""
        )
    elif split == "test":
        ds = load_dataset("ailsntua/QEvasion")["test"]
    else:
        raise ValueError("--split must be one of: train, valid, test")

    # --- Build ids & text pairs ---
    has_id = "id" in ds.column_names
    ids = ds["id"] if has_id else list(range(len(ds)))
    texts = [get_text_pair(row) for row in ds]  # (question, answer)

    # --- Batched inference ---
    preds = []
    for i in range(0, len(texts), batch_size):
        q = [t[0] for t in texts[i : i + batch_size]]
        a = [t[1] for t in texts[i : i + batch_size]]
        enc = tok(
            q, a, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, -1).cpu().numpy().tolist())

    # --- Map ids to string labels & save ---
    id2label = model.config.id2label
    labels = [id2label[int(i)] for i in preds]
    df = pd.DataFrame({"id": ids, "prediction": labels})
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} predictions for split='{split}' to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="out/task2-roberta")
    parser.add_argument(
        "--out_csv", type=str, default="out/task2_roberta_test_preds.csv"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "valid", "test"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.15)
    args = parser.parse_args()
    main(**vars(args))
