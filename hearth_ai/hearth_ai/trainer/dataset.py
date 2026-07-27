import json
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset


class HearthDataset(Dataset):
    """Generic dataset for any Hearth task. Reads a JSONL file where each
    line is one example, e.g.:

        {"text": "I got promoted!", "label": "celebration"}
        {"text": "I'm exhausted.", "label": "emotional_support"}

    or, for multi-label / multi-field tasks (emotion, memory, relationship):

        {"text": "I got promoted!", "labels": [0,1,0,0,...]}
        {"text": "I'm planning to move to Japan.", "store": 1, "type": 3, "importance": 0.91}

    `label_fn` converts the raw JSON dict for one example into whatever
    tensor(s) your loss function expects - keeps this class task-agnostic
    instead of writing five near-identical Dataset classes.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        label_fn: Callable[[dict], object],
        text_field: str = "text",
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.label_fn = label_fn
        self.text_field = text_field
        self.examples: List[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
                if max_examples is not None and len(self.examples) >= max_examples:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, attention_mask = self.tokenizer.encode(ex[self.text_field])
        label = self.label_fn(ex)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": label,
        }


def make_collate_fn(pad_id: int = 0):
    """Pads a batch of variable-length sequences to the same length.
    (If your tokenizer already pads to a fixed length via enable_padding,
    this just handles the label stacking.)"""

    def collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = item["input_ids"].size(0)
            input_ids[i, :seq_len] = item["input_ids"]
            attention_mask[i, :seq_len] = item["attention_mask"]

        first_label = batch[0]["label"]
        if isinstance(first_label, dict):
            label = {
                k: torch.stack([item["label"][k] for item in batch])
                for k in first_label
            }
        else:
            label = torch.stack([item["label"] for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}

    return collate
