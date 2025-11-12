from typing import Tuple, Dict
from pathlib import Path

from datasets import load_dataset, load_from_disk, DatasetDict

CLARITY_CLASSES = [
    "Ambiguous",
    "Clear Non-Reply",
    "Clear Reply",
]  # sorted for consistency


# ---------- PRIMARY ENTRYPOINT (use this in train/eval scripts) ----------
def prepare_qevasion(
    valid_size: float = 0.1,
    seed: int = 42,
    revision: str = "main",
    data_root: str = "data",
    force_refresh: bool = False,
):
    """
    Load QEvasion dataset with stable local caching.
    - If saved splits exist under data_root, load from disk.
    - Otherwise: download from Hub (pinned to `revision`), split, and save.

    Returns:
        train_ds, valid_ds  (datasets.Dataset, datasets.Dataset)
    """
    path = _dataset_dir(valid_size, seed, revision, data_root)

    if Path(path).exists() and not force_refresh:
        print(f"Loading dataset from {path}")
        dd = load_from_disk(path)
        return dd["train"], dd["valid"]

    print("Downloading dataset from Hugging Face…")
    ds_all = load_dataset("ailsntua/QEvasion", revision=revision)

    # Create deterministic train/valid
    split = ds_all["train"].train_test_split(test_size=valid_size, seed=seed)
    dd = DatasetDict(train=split["train"], valid=split["test"])

    # Save to disk
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(path)
    print(f"Saved splits to {path}")
    return dd["train"], dd["valid"]


# ---------- OPTIONAL: direct Hub loader (kept for convenience) ----------
def load_qevasion(splits=("train", "test")):
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


def label_maps_clarity(train_split) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Ensure consistent ordering (macro F1 depends only on correctness, not ids).
    labels = sorted(set(train_split["clarity_label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def label_maps_evasion(train_split) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(set([l for l in train_split["evasion_label"] if l]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# ---------- internal ----------
def _dataset_dir(valid_size: float, seed: int, revision: str, data_root: str) -> str:
    v = int(valid_size * 100)
    return str(Path(data_root) / f"qevasion_s{seed}_v{v}_rev-{revision}")
