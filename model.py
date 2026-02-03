import torch
import torch.nn as nn
import torchvision.models as models

class FusionModel(nn.Module):
    def __init__(self, num_meta_features=19, num_classes=7):
        super(FusionModel, self).__init__()

        # 🔹 Image branch — ResNet50
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity()   # Output: 2048

        # 🔹 Metadata branch: 19 → 16 → 8
        self.meta_net = nn.Sequential(
            nn.Linear(num_meta_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # 🔹 Fusion classifier: (2048 + 8 = 2056)
        self.classifier = nn.Sequential(
            nn.Linear(2056, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, meta):
        img_feat = self.cnn(image)
        meta_feat = self.meta_net(meta)
        fused = torch.cat((img_feat, meta_feat), dim=1)
        return self.classifier(fused)
