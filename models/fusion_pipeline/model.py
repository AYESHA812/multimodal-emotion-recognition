import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.speech_branch = nn.Linear(40, 64)
        self.text_branch = nn.Linear(10, 16)

        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, speech, text):
        s = torch.relu(self.speech_branch(speech))
        t = torch.relu(self.text_branch(text))

        combined = torch.cat((s, t), dim=1)
        output = self.classifier(combined)
        return output
