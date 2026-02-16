import torch.nn as nn
from transformers import BertModel

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(TextEmotionModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
