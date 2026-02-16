import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from dataset import TESSDataset
from model import SpeechEmotionModel
import numpy as np

DATA_PATH = "../../data/speech"
MODEL_PATH = "../../results/speech_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TESSDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = SpeechEmotionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in loader:
        features = features.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))
