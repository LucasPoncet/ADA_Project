import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, p_dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.classifier(x)
