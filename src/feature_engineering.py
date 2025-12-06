import pandas as pd
import numpy as np
import re
import os
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch

warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR="../csv_files"
OUTPUT_DIR="../csv_files"

NEGATION_PATTERN=re.compile(r"\b(no|not|never|none|nothing|nowhere|neither|nor|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|without)\b",
    re.IGNORECASE)

def feature_engineering(df, embedding_model):
    df=df.copy()

    # Ensure text is string
    df['question']=df['question'].astype(str)
    df['interview_answer']=df['interview_answer'].astype(str)

    df['q_word_count']=df['question'].str.split().str.len()
    df['a_word_count']=df['interview_answer'].str.split().str.len()
    
    df['a_negation_count']=df['interview_answer'].apply(lambda x: len(NEGATION_PATTERN.findall(x)))
    
    q_embeddings=embedding_model.encode(df['question'].tolist(),show_progress_bar=True,batch_size=32,device=device)
    a_embeddings=embedding_model.encode(df['interview_answer'].tolist(),show_progress_bar=True,batch_size=32,device=device)
    
    df['qa_cosine_sim']=[cosine_similarity([q_emb], [a_emb])[0][0] for q_emb, a_emb in zip(q_embeddings, a_embeddings)]
    
    #question embeddings: 384 dimensions
    for i in range(q_embeddings.shape[1]):
        df[f'q_embedding_{i}']=q_embeddings[:, i]
    
    #answer embeddings: 384 dimensions
    for i in range(a_embeddings.shape[1]):
        df[f'a_embedding_{i}']=a_embeddings[:, i]
    
    print(f"Total features created: {4+2*q_embeddings.shape[1]}")
    print(f"4 engineered features (word counts, negation, cosine sim)")
    print(f"{q_embeddings.shape[1]} question embedding dimensions")
    print(f"{a_embeddings.shape[1]} answer embedding dimensions")
    
    return df

def main():

    train_file=os.path.join(INPUT_DIR, "training_data.csv")
    test_file=os.path.join(INPUT_DIR, "validation_data.csv")
    
    train_df=pd.read_csv(train_file)
    test_df=pd.read_csv(test_file)
    
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
    
    if torch.cuda.is_available():
        embedding_model=embedding_model.to(device)
        print(f"Model loaded on GPU")
    else:
        print(f"Model loaded on CPU")
    

    train_df=feature_engineering(train_df, embedding_model)
    test_df=feature_engineering(test_df, embedding_model)
    
    print("\n[5/5] Creating train/validation split...")
    print("  Using 85/15 split with stratification on clarity_label")
    
    train_final, val_df=train_test_split(
        train_df,
        test_size=0.15,
        random_state=SEED,
        stratify=train_df['clarity_label'])
    
    print(f"Final training set: {len(train_final)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    train_output=os.path.join(OUTPUT_DIR, "FE_training_data.csv")
    val_output=os.path.join(OUTPUT_DIR, "FE_validation_data.csv")
    test_output=os.path.join(OUTPUT_DIR, "FE_test_data.csv")
    
    train_final.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)

if __name__ == "__main__":
    main()