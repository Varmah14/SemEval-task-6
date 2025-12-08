import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report
import warnings
import os
from safetensors.torch import load_file
warnings.filterwarnings('ignore')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS=["q_word_count", "a_word_count", "a_negation_count", "qa_cosine_sim"]
FEATURE_DIM=len(FEATURE_COLS)


class QADataset(Dataset):    
    def __init__(self, df, tokenizer, max_length=512, task='clarity'):
        self.data=df.reset_index(drop=True)
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.task=task
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row=self.data.iloc[idx]
        
        text=f"Question: {row['question']} [SEP] Answer: {row['interview_answer']}"
        
        encoding=self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label=int(row['clarity_label_id'] if self.task=='clarity' else row['evasion_label_id'])
        features=row[FEATURE_COLS].values.astype(np.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ELECTRAWithFeatures(nn.Module):    
    def __init__(self, model_name, num_labels, feature_dim, dropout=0.1, class_weights=None):
        super().__init__()
        self.electra=AutoModel.from_pretrained(model_name)
        hidden_size=self.electra.config.hidden_size
        
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_size + feature_dim, num_labels)
        self.class_weights=class_weights
        
    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs=self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_output=outputs.last_hidden_state[:, 0, :]
        cls_output=self.dropout(cls_output)
        
        combined=torch.cat([cls_output, features], dim=1)
        logits=self.classifier(combined)
        
        loss=None
        if labels is not None:
            if self.class_weights is not None:
                loss_fn=nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fn=nn.CrossEntropyLoss()
            loss=loss_fn(logits, labels)
            
        return {'loss': loss, 'logits': logits}


class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels=inputs.pop("labels")
        features=inputs.pop("features")
        
        outputs=model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            features=features,
            labels=labels
        )
        
        loss=outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def evaluate_on_test(task='clarity'):    
    print(f"\n{'='*70}")
    print(f"Evaluating ELECTRA-base on TEST set: {task.upper()} task")
    print(f"{'='*70}")
    
    test_df=pd.read_csv("../csv_files/FE_test_data.csv")
    test_df[FEATURE_COLS]=test_df[FEATURE_COLS].astype(float)
    
    if task=='clarity':
        num_labels=test_df['clarity_label_id'].nunique()
        label_col='clarity_label'
    else:
        num_labels=test_df['evasion_label_id'].nunique()
        label_col='evasion_label'
    
    label_names=sorted(test_df[label_col].unique())
    print(f"Test samples: {len(test_df)}, Classes: {num_labels}")
    
    model_path=f"../models/{task}_electra"
    tokenizer=AutoTokenizer.from_pretrained(model_path)

    base_model_name="google/electra-base-discriminator"
    model=ELECTRAWithFeatures(base_model_name, num_labels, FEATURE_DIM)
    
    safetensors_path=os.path.join(model_path, "model.safetensors")
    bin_path=os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        print(f"Loading weights from: {safetensors_path}")
        state_dict=load_file(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"Loading weights from: {bin_path}")
        state_dict=torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}. Looked for 'model.safetensors' or 'pytorch_model.bin'")
    
    model.load_state_dict(state_dict)
    model.to(device)

    test_dataset=QADataset(test_df, tokenizer, task=task)
    
    training_args=TrainingArguments(
        output_dir=f"../results/{task}_test_eval",
        per_device_eval_batch_size=16,
        report_to="none"
    )
    
    trainer=CustomTrainer(
        model=model,
        args=training_args
    )
    
    print("Running predictions on test set...")
    predictions=trainer.predict(test_dataset)
    preds=np.argmax(predictions.predictions, axis=1)
    labels=predictions.label_ids
    
    macro_f1=f1_score(labels, preds, average='macro')
    weighted_f1=f1_score(labels, preds, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS - {task.upper()}")
    print(f"{'='*70}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nPer-class breakdown:")
    
    report=classification_report(labels, preds, target_names=label_names, output_dict=True)
    df_report=pd.DataFrame(report).transpose()
    df_report.loc["macro_f1", "f1-score"]=macro_f1
    
    output_file=f"../csv_files/{task}_electra_test_results.csv"
    df_report.to_csv(output_file)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\nPer-class F1 scores:")
    for label_name in label_names:
        f1=report[label_name]['f1-score']
        support=report[label_name]['support']
        print(f"  {label_name:25s}: {f1:.4f} (n={int(support)})")
    
    return macro_f1, weighted_f1, df_report


if __name__=="__main__":
    clarity_macro, clarity_weighted, _=evaluate_on_test(task='clarity')
    evasion_macro, evasion_weighted, _=evaluate_on_test(task='evasion')
    
    print(f"\n{'='*70}")
    print("FINAL TEST SET SUMMARY")
    print(f"{'='*70}")
    print(f"Clarity Task  - Macro F1: {clarity_macro:.4f}, Weighted F1: {clarity_weighted:.4f}")
    print(f"Evasion Task  - Macro F1: {evasion_macro:.4f}, Weighted F1: {evasion_weighted:.4f}")