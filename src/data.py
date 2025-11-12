from datasets import load_dataset
from typing import Tuple, Dict

CLARITY_CLASSES = ["Ambiguous", "Clear Non-Reply", "Clear Reply"]  # sorted for consistency

def load_qevasion(splits=("train","test")):
    ds = load_dataset("ailsntua/QEvasion")
    out = {}
    for s in splits:
        out[s] = ds[s]
    return out

def get_text_pair(batch):
    # Some rows use 'question'; others 'interview_question'. Prefer 'question' if present.
    q = batch.get("question")
    iq = batch.get("interview_question")
    question = q if q and q.strip() else (iq if iq else "")
    answer = batch["interview_answer"]
    return question, answer

def label_maps_clarity(train_split) -> Tuple[Dict[str,int], Dict[int,str]]:
    # Ensure consistent ordering (macro F1 depends only on correctness, not ids).
    labels = sorted(set(train_split["clarity_label"]))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label

def label_maps_evasion(train_split) -> Tuple[Dict[str,int], Dict[int,str]]:
    labels = sorted(set([l for l in train_split["evasion_label"] if l]))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label
