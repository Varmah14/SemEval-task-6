import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report


def get_data():
    train_df=pd.read_csv("../csv_files/training_data.csv")
    val_df=pd.read_csv("../csv_files/validation_data.csv")

    train_df['qa_text']=train_df['question']+" [SEP] "+train_df['interview_answer']
    val_df['qa_text']=val_df['question']+" [SEP] "+val_df['interview_answer']

    return train_df, val_df

def get_data_clarity_task():
    train_df,val_df=get_data()

    clarity_labels=train_df['clarity_label'].unique()
    clarity_label_map={label: i for i, label in enumerate(clarity_labels)}

    X_train=train_df['qa_text']
    X_test=val_df['qa_text']

    y_train=train_df['clarity_label_id']
    y_test=val_df['clarity_label_id']

    return X_train,X_test,y_train, y_test, clarity_label_map

def get_data_evasion_task():
    train_df,val_df=get_data()

    evasion_labels=train_df['evasion_label'].unique()
    evasion_label_map={label: i for i, label in enumerate(evasion_labels)}

    X_train=train_df['qa_text']
    X_test=val_df['qa_text']

    y_train=train_df['evasion_label_id']
    y_test=val_df['evasion_label_id']

    return X_train,X_test,y_train, y_test, evasion_label_map

def logistic_regression(get_data,task):
    
    X_train,X_test,y_train,y_test,label_map=get_data()
    
    text_clf_pipeline=Pipeline([('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  #use both 1-word and 2-word phrases (e.g., "taxes" and "low taxes")
            stop_words="english", #remove common words like 'the', 'is', 'a'
            max_df=0.7,          #ignore words that appear in > 70% of docs (too common)
            min_df=5             #ignore words that appear in < 5 docs (too rare)
        )),
        ('clf', LogisticRegression(random_state=17)),
    ])

    text_clf_pipeline.fit(X_train, y_train)
    
    y_pred=text_clf_pipeline.predict(X_test)
    
    target_names=label_map.keys()
    macro_f1=f1_score(y_test, y_pred, average='macro')

    report=classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
    df=pd.DataFrame(report).transpose()
    df.loc["macro_f1","f1-score"]=macro_f1
    
    df.to_csv(f"../csv_files/{task}_lr_results.csv")

if __name__=="__main__":
    logistic_regression(get_data_clarity_task,"clarity_task")
    logistic_regression(get_data_evasion_task,"evasion_task")