import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

emotion_to_idx = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}

class FusionDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(data_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.data_path, file)

        # ---- Speech Feature ----
        signal, sr = librosa.load(path, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        # ---- Dummy Text Feature (placeholder) ----
        # Since TESS has no meaningful text info
        text_feat = np.zeros(10)

        emotion = file.replace(".wav", "").split("_")[2]
        label = emotion_to_idx[emotion]

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(text_feat, dtype=torch.float32),
            torch.tensor(label)
        )
