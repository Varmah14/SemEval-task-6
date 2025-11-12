import argparse
from datasets import DatasetDict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report
from src.data import load_qevasion, get_text_pair

def pack_text(example):
    q,a = get_text_pair(example)
    return f"{q} [SEP] {a}"

def main(seed=42, valid_size=0.15):
    ds = load_qevasion(splits=("train","test"))
    # Make a validation split from train
    split = ds["train"].train_test_split(test_size=valid_size, seed=seed)
    train, valid = split["train"], split["test"]

    X_tr = [pack_text(x) for x in train]
    y_tr = train["clarity_label"]
    X_va = [pack_text(x) for x in valid]
    y_va = valid["clarity_label"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000)),
        ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=seed))
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)

    macro = f1_score(y_va, pred, average="macro")
    print(f"Macro F1 (Task1 / TF-IDF): {macro:.4f}\n")
    print(classification_report(y_va, pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.15)
    args = parser.parse_args()
    main(seed=args.seed, valid_size=args.valid_size)
