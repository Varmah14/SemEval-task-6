import argparse, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load as load_metric
from src.data import get_text_pair, label_maps_evasion

def tokenize(batch, tokenizer, label2id):
    if not batch["evasion_label"]:
        return None
    q,a = get_text_pair(batch)
    enc = tokenizer(q, a, truncation=True, padding="max_length", max_length=512)
    enc["labels"] = label2id[batch["evasion_label"]]
    return enc

def main(model_name="roberta-base", out_dir="out/task2-roberta", seed=42, epochs=5, bs=8, lr=2e-5, valid_size=0.15):
    ds_all = load_dataset("ailsntua/QEvasion")
    split = ds_all["train"].train_test_split(test_size=valid_size, seed=seed)
    train, valid = split["train"], split["test"]

    label2id, id2label = label_maps_evasion(train)
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = train.map(lambda b: tokenize(b, tok, label2id), remove_columns=train.column_names)
    va = valid.map(lambda b: tokenize(b, tok, label2id), remove_columns=valid.column_names)
    tr = tr.filter(lambda x: x is not None); va = va.filter(lambda x: x is not None)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    f1_metric = load_metric("f1")
    def comp(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=seed,
        report_to="none"
    )

    Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, compute_metrics=comp).train()
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print("Task2 model saved to", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--out_dir", type=str, default="out/task2-roberta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--valid_size", type=float, default=0.15)
    args = p.parse_args()
    main(**vars(args))
