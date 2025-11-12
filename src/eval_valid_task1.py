import numpy as np, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from src.data import get_text_pair, label_maps_clarity

def main(model_dir="out/task1-roberta", seed=42, valid_size=0.15):
    ds = load_dataset("ailsntua/QEvasion")
    split = ds["train"].train_test_split(test_size=valid_size, seed=seed)
    train, valid = split["train"], split["test"]
    label2id, id2label = label_maps_clarity(train)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    Xq, Xa, y = [], [], []
    for r in valid:
        q,a = get_text_pair(r)
        Xq.append(q); Xa.append(a); y.append(label2id[r["clarity_label"]])

    import torch
    preds = []
    for i in range(0, len(Xq), 16):
        enc = tok(Xq[i:i+16], Xa[i:i+16], return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, -1).cpu().numpy().tolist())

    y_true = [id2label[i] for i in y]
    y_pred = [id2label[i] for i in preds]
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=sorted(set(y_true)), columns=sorted(set(y_true))))

if __name__ == "__main__":
    main()
