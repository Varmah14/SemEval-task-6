import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric

from src.data import (
    get_text_pair,
    label_maps_clarity,
    prepare_qevasion,
)


def tokenize(batch, tokenizer, label2id):
    q, a = get_text_pair(batch)
    enc = tokenizer(q, a, truncation=True, padding="max_length", max_length=512)
    enc["labels"] = label2id[batch["clarity_label"]]
    return enc


def main(
    model_name="roberta-base",
    out_dir="out/task1-roberta",
    seed=42,
    epochs=3,
    bs=8,
    lr=2e-5,
    valid_size=0.15,
):
    # Load cached train/valid (or create & save on first run)
    train, valid = prepare_qevasion(valid_size=valid_size, seed=seed, revision="main")

    # Ensure examples have a label
    train = train.filter(
        lambda ex: ex.get("clarity_label") is not None and ex["clarity_label"] != ""
    )
    valid = valid.filter(
        lambda ex: ex.get("clarity_label") is not None and ex["clarity_label"] != ""
    )

    label2id, id2label = label_maps_clarity(train)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tr = train.map(
        lambda b: tokenize(b, tokenizer, label2id), remove_columns=train.column_names
    )
    va = valid.map(
        lambda b: tokenize(b, tokenizer, label2id), remove_columns=valid.column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    f1_metric = load_metric("f1")

    def comp(eval_pred):
        # Handles both tuple and EvalPrediction
        logits = (
            eval_pred[0]
            if isinstance(eval_pred, (list, tuple))
            else eval_pred.predictions
        )
        labels = (
            eval_pred[1]
            if isinstance(eval_pred, (list, tuple))
            else eval_pred.label_ids
        )
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
        save_total_limit=2,  #  keeps disk usage in check
        remove_unused_columns=True,  # safe since we mapped to model inputs
        # fp16=True,                   # (optional) if you have a CUDA GPU
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tr, eval_dataset=va, compute_metrics=comp
    )
    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Training complete. Best model saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--out_dir", type=str, default="out/task1-roberta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--valid_size", type=float, default=0.15)
    args = parser.parse_args()
    main(**vars(args))
