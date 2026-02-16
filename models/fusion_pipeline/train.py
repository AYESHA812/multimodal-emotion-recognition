import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import FusionDataset
from model import FusionModel

data_path = "../../data/speech"

dataset = FusionDataset(data_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = FusionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    total_loss = 0
    for speech, text, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(speech, text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fusion_model.pth")
print("Fusion model trained.")
