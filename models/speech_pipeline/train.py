import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TESSDataset
from model import SpeechEmotionModel
import os

DATA_PATH = "../../data/speech"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TESSDataset(DATA_PATH)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = SpeechEmotionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

from sklearn.metrics import classification_report, accuracy_score

model.eval()
all_preds = []
all_labels = []

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)

print("\nTest Results:")
print(classification_report(all_labels, all_preds))
print(f"\nTest Accuracy: {acc:.4f}")


os.makedirs("../../results", exist_ok=True)
torch.save(model.state_dict(), "../../results/speech_model.pth")

print("Training complete. Model saved.")
