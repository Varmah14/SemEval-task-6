import argparse, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data import get_text_pair

@torch.no_grad()
def main(model_dir="out/task2-roberta", out_csv="out/task2_roberta_test_preds.csv", batch_size=16):
    ds_test = load_dataset("ailsntua/QEvasion")["test"]

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = [get_text_pair(r) for r in ds_test]
    preds = []
    for i in range(0, len(texts), batch_size):
        q = [t[0] for t in texts[i:i+batch_size]]
        a = [t[1] for t in texts[i:i+batch_size]]
        enc = tok(q, a, return_tensors="pt", truncation=True, padding=True, max_length=512)
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, -1).cpu().numpy().tolist())

    id2label = model.config.id2label
    labels = [id2label[int(i)] for i in preds]
    df = pd.DataFrame({"id": range(len(labels)), "prediction": labels})
    df.to_csv(out_csv, index=False)
    print("✅ Saved predictions to", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="out/task2-roberta")
    parser.add_argument("--out_csv", type=str, default="out/task2_roberta_test_preds.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(**vars(args))
