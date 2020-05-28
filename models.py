import torch
from torch import nn
from torchvision.models.vgg import VGG, make_layers, cfgs
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.utils import load_state_dict_from_url


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


class GTARes18Net(ResNet):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            self.load_state_dict(
                load_state_dict_from_url(
                    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    progress=True))

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GTAVGG11Net(VGG):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__(make_layers(cfgs['A'], batch_norm=True, **kwargs))
        if pretrained:
            self.load_state_dict(
                load_state_dict_from_url(
                    'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
                    progress=True))

        self.feature_extractor = nn.Sequential(
            self.features,
            self.avgpool,
        )
        num_ftrs = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
