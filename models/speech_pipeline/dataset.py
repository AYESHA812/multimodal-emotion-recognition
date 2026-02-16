import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

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


class TESSDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def extract_features(self, file_path):
        signal, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.data_path, file)

        features = self.extract_features(file_path)

        emotion = file.replace(".wav", "").split("_")[2]

        # Normalize short label
        if emotion == "ps":
            emotion = "pleasant_surprise"

        label = emotion_to_idx[emotion]

        return features, label
