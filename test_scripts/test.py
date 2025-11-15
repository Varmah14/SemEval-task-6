import pandas as pd

splits={'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df=pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["train"])

print(df.head())