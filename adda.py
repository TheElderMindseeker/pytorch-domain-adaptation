"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange

import config
from utils import GrayscaleToRgb, loop_iterable, set_requires_grad
from models import GTANet, GTARes18Net, GTAVGG11Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.model == 'gta':
        model_file = './trained_models/gta_source.pt'
        out_file = './trained_models/gta_adda.pt'
        out_ftrs = 4375

        model = GTANet().to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        set_requires_grad(model, False)
        source_model = model.feature_extractor
        clf = model

        model_2 = GTANet().to(device)
        model_2.load_state_dict(torch.load(model_file))
        target_model = model_2.feature_extractor

    elif args.model == 'gta-res':
        model_file = './trained_models/gta_res_source.pt'
        out_file = './trained_models/gta_res_adda.pt'

        model = GTARes18Net(9, pretrained=False).to(device)
        out_ftrs = model.fc.in_features
        model.load_state_dict(torch.load(model_file))
        model.eval()
        set_requires_grad(model, False)

        source_model = model.feature_extractor
        clf = model

        model_2 = GTARes18Net(9, pretrained=False).to(device)
        model_2.load_state_dict(torch.load(model_file))
        target_model = model_2.feature_extractor

    elif args.model == 'gta-vgg':
        model_file = './trained_models/gta_vgg_source.pt'
        out_file = './trained_models/gta_vgg_adda.pt'

        model = GTAVGG11Net(9, pretrained=False).to(device)
        out_ftrs = model.classifier[0].in_features  # should be 512 * 7 * 7
        model.load_state_dict(torch.load(model_file))
        model.eval()
        set_requires_grad(model, False)

        def source_model(x):
            x = model.features(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        clf = model

        model_2 = GTAVGG11Net(9, pretrained=False).to(device)
        model_2.load_state_dict(torch.load(model_file))

        def target_model(x):
            x = model_2.features(x)
            x = model_2.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    else:
        raise ValueError(f'Unknown model type {args.model}')

    discriminator = nn.Sequential(
        nn.Linear(out_ftrs, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    ).to(device)

    half_batch = args.batch_size // 2
    target_dataset = ImageFolder('./data',
                                 transform=Compose([
                                     Resize((398, 224)),
                                     RandomCrop(224),
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

    source_dataset = ImageFolder('./t_data',
                                 transform=Compose([
                                     RandomCrop(224,
                                                pad_if_needed=True,
                                                padding_mode='reflect'),
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

    discriminator_optim = torch.optim.Adam(discriminator.parameters())
    target_optim = torch.optim.Adam(model_2.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        batch_iterator = zip(loop_iterable(source_loader),
                             loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(model_2, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(
                    source_x.shape[0], -1)
                target_features = target_model(target_x).view(
                    target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([
                    torch.ones(source_x.shape[0], device=device),
                    torch.zeros(target_x.shape[0], device=device)
                ])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()
                                  ).float().mean().item()

            # Train classifier
            set_requires_grad(model_2, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(
                    target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations * args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        if args.model == 'gta':
            clf.feature_extractor = target_model
        elif args.model == 'gta-res':
            clf.conv1 = model_2.conv1
            clf.bn1 = model_2.bn1
            clf.relu = model_2.relu
            clf.maxpool = model_2.maxpool

            clf.layer1 = model_2.layer1
            clf.layer2 = model_2.layer2
            clf.layer3 = model_2.layer3
            clf.layer4 = model_2.layer4

            clf.avgpool = model_2.avgpool

        torch.save(clf.state_dict(), out_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Domain adaptation using ADDA')
    arg_parser.add_argument('--model',
                            type=str,
                            default='gta',
                            help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--iterations', type=int, default=500)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=10)
    args = arg_parser.parse_args()
    main(args)
