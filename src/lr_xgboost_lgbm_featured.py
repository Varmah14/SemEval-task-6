import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def get_featured_data():
    train_df=pd.read_csv("../csv_files/FE_training_data.csv")
    val_df=pd.read_csv("../csv_files/FE_validation_data.csv")
    
    feature_cols=['q_word_count', 'a_word_count', 'a_negation_count', 'qa_cosine_sim']
    embedding_cols=[f'q_embedding_{i}' for i in range(384)] + [f'a_embedding_{i}' for i in range(384)]
    feature_cols.extend(embedding_cols)
    
    return train_df, val_df, feature_cols


def get_data_clarity_task():
    train_df, val_df, feature_cols=get_featured_data()
    
    X_train=train_df[feature_cols]
    X_test=val_df[feature_cols]
    y_train=train_df['clarity_label_id']
    y_test=val_df['clarity_label_id']
    
    clarity_labels=train_df['clarity_label'].unique()
    label_map={label: i for i, label in enumerate(clarity_labels)}
    
    return X_train, X_test, y_train, y_test, label_map


def get_data_evasion_task():
    train_df, val_df, feature_cols=get_featured_data()
    
    X_train=train_df[feature_cols]
    X_test=val_df[feature_cols]
    y_train=train_df['evasion_label_id']
    y_test=val_df['evasion_label_id']
    
    evasion_labels=train_df['evasion_label'].unique()
    label_map={label: i for i, label in enumerate(evasion_labels)}
    
    return X_train, X_test, y_train, y_test, label_map


def train_logistic_regression(X_train, X_test, y_train, y_test, label_map, task):
    model=LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    
    macro_f1=f1_score(y_test, y_pred, average='macro')
    report=classification_report(y_test, y_pred, target_names=label_map.keys(), output_dict=True)
    
    df=pd.DataFrame(report).transpose()
    df.loc["macro_f1", "f1-score"]=macro_f1
    df.to_csv(f"../csv_files/{task}_lr_fe_results.csv")
    
    print(f"{task} - Logistic Regression - Macro F1: {macro_f1:.4f}")


def train_xgboost(X_train, X_test, y_train, y_test, label_map, task):
    model=xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        #GPU PARAMETERS:
        tree_method='hist', 
        device='cuda', #use 'cuda' for GPU
    )
    
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    
    macro_f1=f1_score(y_test, y_pred, average='macro')
    report=classification_report(y_test, y_pred, target_names=label_map.keys(), output_dict=True)
    
    df=pd.DataFrame(report).transpose()
    df.loc["macro_f1", "f1-score"]=macro_f1
    df.to_csv(f"../csv_files/{task}_xgb_fe_results.csv")
    
    print(f"{task} - XGBoost - Macro F1: {macro_f1:.4f}")


def train_lightgbm(X_train, X_test, y_train, y_test, label_map, task):
    model=lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    
    macro_f1=f1_score(y_test, y_pred, average='macro')
    report=classification_report(y_test, y_pred, target_names=label_map.keys(), output_dict=True)
    
    df=pd.DataFrame(report).transpose()
    df.loc["macro_f1", "f1-score"]=macro_f1
    df.to_csv(f"../csv_files/{task}_lgbm_fe_results.csv")
    
    print(f"{task} - LightGBM - Macro F1: {macro_f1:.4f}")


def run_all_models(get_data_func, task_name):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test, label_map=get_data_func()
    
    train_logistic_regression(X_train, X_test, y_train, y_test, label_map, task_name)
    train_xgboost(X_train, X_test, y_train, y_test, label_map, task_name)
    train_lightgbm(X_train, X_test, y_train, y_test, label_map, task_name)


if __name__ == "__main__":
    run_all_models(get_data_clarity_task, "clarity_task")
    run_all_models(get_data_evasion_task, "evasion_task")