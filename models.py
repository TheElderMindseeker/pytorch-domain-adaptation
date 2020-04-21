from torch import nn
from torchvision import models


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits


class GTANet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5, stride=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(5, 7, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4375, 9),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits


def GTARes18Net(num_classes: int):
    """Create a model with Resnet18 as pretrained feature extractor

    Args:
        num_classes: Number of classes in dataset.

    """
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def GTAVGG11Net(num_classes: int):
    """Create a model with VGG-11 as pretrained feature extractor

    Args:
        num_classes: Number of classes in dataset.

    """
    model = models.vgg11_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model
