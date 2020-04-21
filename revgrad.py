"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from tqdm import tqdm

import config
from data import MNISTM
from models import GTANet, GTARes18Net, GTAVGG11Net
from utils import GradientReversal, GrayscaleToRgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.model == 'gta':
        model = GTANet().to(device)
        feature_extractor = model.feature_extractor
        clf = model.classifier
        model_file = './trained_models/gta_source.pt'
        out_file = './trained_models/gta.pt'
        out_ftrs = 4375

    elif args.model == 'gta-res':
        model = GTARes18Net(9).to(device)

        def feature_extractor(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        clf = model.fc
        model_file = './trained_models/gta_res_source.pt'
        out_file = './trained_models/gta_res.pt'
        out_ftrs = model.fc.in_features

    elif args.model == 'gta-vgg':
        model = GTAVGG11Net(9).to(device)

        def feature_extractor(x):
            x = model.features(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        clf = model.classifier
        model_file = './trained_models/gta_vgg_source.pt'
        out_file = './trained_models/gta_vgg.pt'
        out_ftrs = model.classifier[0].in_features  # should be 512 * 7 * 7

    else:
        raise ValueError(f'Unknown model type {args.model}')

    model.load_state_dict(torch.load(model_file))

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(out_ftrs, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    half_batch = args.batch_size // 2

    source_dataset = ImageFolder('./data',
                                 transform=Compose([
                                     Resize((398, 224)),
                                     RandomCrop(224),
                                     RandomHorizontalFlip(),
                                     ToTensor(),
                                     Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225]),
                                 ]))
    source_loader = DataLoader(source_dataset,
                               batch_size=half_batch,
                               shuffle=True,
                               num_workers=1,
                               pin_memory=True)

    target_dataset = ImageFolder('./t_data',
                                 transform=Compose([
                                     RandomCrop(224,
                                                pad_if_needed=True,
                                                padding_mode='reflect'),
                                     RandomHorizontalFlip(),
                                     ToTensor(),
                                     Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225]),
                                 ]))
    target_loader = DataLoader(target_dataset,
                               batch_size=half_batch,
                               shuffle=True,
                               num_workers=1,
                               pin_memory=True)

    optim = torch.optim.Adam(
        list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs + 1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches,
                                                             leave=False,
                                                             total=n_batches):
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat(
                [torch.ones(source_x.shape[0]),
                 torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)

            features = feature_extractor(x).view(x.shape[0], -1)
            domain_preds = discriminator(features).squeeze()
            label_preds = clf(features[:source_x.shape[0]])

            domain_loss = F.binary_cross_entropy_with_logits(
                domain_preds, domain_y)
            label_loss = F.cross_entropy(label_preds, label_y)
            loss = domain_loss + label_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += domain_loss.item()
            total_label_accuracy += (
                label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), out_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Domain adaptation using RevGrad')
    arg_parser.add_argument('--model',
                            type=str,
                            help='A model in trained_models',
                            default='gta')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=16)
    args = arg_parser.parse_args()
    main(args)
