import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

EMOTIONS = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'pleasant_surprise',
    'sad'
]

emotion_to_idx = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

class TextEmotionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        emotion = self.data.iloc[idx]["emotion"]

        if emotion == "ps":
            emotion = "pleasant_surprise"

        label = emotion_to_idx[emotion]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=10,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
