import argparse, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric
from src.data import (
    get_text_pair,
    label_maps_evasion,
    prepare_qevasion,
)  # use cached loader


def tokenize(batch, tokenizer, label2id):
    q, a = get_text_pair(batch)
    enc = tokenizer(q, a, truncation=True, padding="max_length", max_length=512)
    enc["labels"] = label2id[batch["evasion_label"]]
    return enc


def main(
    model_name="roberta-base",
    out_dir="out/task2-roberta",
    seed=42,
    epochs=5,
    bs=8,
    lr=2e-5,
    valid_size=0.15,
):
    # Load cached train/valid (or create & save on first run)
    train, valid = prepare_qevasion(valid_size=valid_size, seed=seed, revision="main")

    # Ensure rows have a label before mapping
    train = train.filter(
        lambda ex: ex.get("evasion_label") is not None and ex["evasion_label"] != ""
    )
    valid = valid.filter(
        lambda ex: ex.get("evasion_label") is not None and ex["evasion_label"] != ""
    )

    label2id, id2label = label_maps_evasion(train)

    tok = AutoTokenizer.from_pretrained(model_name)

    # map per-example (batched=False) is fine here since we use pair encoding per row
    tr = train.map(
        lambda b: tokenize(b, tok, label2id), remove_columns=train.column_names
    )
    va = valid.map(
        lambda b: tokenize(b, tok, label2id), remove_columns=valid.column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    f1_metric = load_metric("f1")

    def comp(eval_pred):
        # eval_pred may be (preds, labels) or object with .predictions/.label_ids
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"]
        }

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
        report_to="none",
        save_total_limit=2,  # optional: keep disk small
        remove_unused_columns=True,  # fine since we mapped to model inputs
        # fp16=True,                     # uncomment if you have CUDA + want mixed precision
    )

    Trainer(
        model=model, args=args, train_dataset=tr, eval_dataset=va, compute_metrics=comp
    ).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
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
