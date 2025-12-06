import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)

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
        
        label=int(row['clarity_label_id'] if self.task == 'clarity' else row['evasion_label_id'])

        features=row[FEATURE_COLS].values.astype(np.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ELECTRAWithFeatures(nn.Module):
    """ELECTRA model with engineered features"""
    
    def __init__(self, model_name, num_labels, feature_dim, dropout=0.1, class_weights=None):
        super().__init__()
        self.electra=AutoModel.from_pretrained(model_name)
        hidden_size=self.electra.config.hidden_size
        
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_size + feature_dim, num_labels)
        self.class_weights=class_weights
        
    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs=self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_output=outputs.last_hidden_state[:, 0, :]  # [CLS] token
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


def compute_metrics(eval_pred):
    predictions, labels=eval_pred
    preds=np.argmax(predictions, axis=1)
    
    macro_f1=f1_score(labels, preds, average='macro')
    weighted_f1=f1_score(labels, preds, average='weighted')
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


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


def load_data(task):
    train_df=pd.read_csv("../csv_files/FE_training_data.csv")
    val_df=pd.read_csv("../csv_files/FE_validation_data.csv")
    
    for df in [train_df, val_df]:
        df[FEATURE_COLS]=df[FEATURE_COLS].astype(float)
    
    if task == 'clarity':
        num_labels=train_df['clarity_label_id'].nunique()
        label_col='clarity_label'
    else:
        num_labels=train_df['evasion_label_id'].nunique()
        label_col='evasion_label'
    
    label_names=sorted(train_df[label_col].unique())
    
    return train_df, val_df, num_labels, label_names


def train_model(task='clarity', model_name='google/electra-base-discriminator'):
    
    print(f"\n{'='*70}")
    print(f"Training ELECTRA-base: {task.upper()} task")
    print(f"{'='*70}")
    
    train_df, val_df, num_labels, label_names=load_data(task)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Classes: {num_labels}")
    
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    train_dataset=QADataset(train_df, tokenizer, task=task)
    val_dataset=QADataset(val_df, tokenizer, task=task)
    
    #compute class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    label_col='clarity_label_id' if task == 'clarity' else 'evasion_label_id'
    class_weights=compute_class_weight(
        'balanced',
        classes=np.unique(train_df[label_col]),
        y=train_df[label_col]
    )
    class_weights=torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    model=ELECTRAWithFeatures(model_name, num_labels, FEATURE_DIM, class_weights=class_weights).to(device)
    
    training_args=TrainingArguments(
        output_dir=f"../results/{task}_electra",
        num_train_epochs=5,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=16,  
        gradient_accumulation_steps=1,  
        learning_rate=2e-5,  
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,  
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=SEED,
        dataloader_num_workers=4,
        report_to="none"
    )
    
    trainer=CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )
    
    print("\nTraining...")
    trainer.train()
    
    print("\nEvaluating...")
    eval_results=trainer.evaluate()
    
    predictions=trainer.predict(val_dataset)
    preds=np.argmax(predictions.predictions, axis=1)
    labels=predictions.label_ids
    
    macro_f1=f1_score(labels, preds, average='macro')
    report=classification_report(labels, preds, target_names=label_names, output_dict=True)
    
    df_report=pd.DataFrame(report).transpose()
    df_report.loc["macro_f1", "f1-score"]=macro_f1
    df_report.to_csv(f"../csv_files/{task}_electra_results.csv")
    
    model_path=f"../models/{task}_electra"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"\nResults:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {eval_results['eval_weighted_f1']:.4f}")
    print(f"  Model saved to: {model_path}")
    
    return trainer, model, eval_results


if __name__ == "__main__":
    train_model(task='clarity')
    train_model(task='evasion')