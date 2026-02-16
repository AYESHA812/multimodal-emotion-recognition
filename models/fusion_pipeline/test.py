import torch
from torch.utils.data import DataLoader
from dataset import FusionDataset
from model import FusionModel

data_path = "../../data/speech"

dataset = FusionDataset(data_path)
test_loader = DataLoader(dataset, batch_size=16)

model = FusionModel()
model.load_state_dict(torch.load("fusion_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for speech, text, labels in test_loader:
        outputs = model(speech, text)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print("Fusion Test Accuracy:", accuracy)
