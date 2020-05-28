"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import argparse

import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange

import config
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from models import GTANet, GTARes18Net, GTAVGG11Net
from utils import GrayscaleToRgb, loop_iterable, set_requires_grad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds,
                     interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True,
                     create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def main(args):
    if args.model == 'gta':
        model_file = './trained_models/gta_source.pt'
        out_file = './trained_models/gta_wdgrl.pt'
        out_ftrs = 4375

        clf_model = GTANet().to(device)
        clf_model.load_state_dict(torch.load(model_file))
        feature_extractor = clf_model.feature_extractor
        discriminator = clf_model.classifier

    elif args.model == 'gta-res':
        model_file = './trained_models/gta_res_source.pt'
        out_file = './trained_models/gta_res_wdgrl.pt'

        clf_model = GTARes18Net(9, pretrained=False).to(device)
        out_ftrs = clf_model.fc.in_features
        clf_model.load_state_dict(torch.load(model_file))

        feature_extractor = clf_model.feature_extractor
        discriminator = clf_model.fc

    elif args.model == 'gta-vgg':
        model_file = './trained_models/gta_vgg_source.pt'
        out_file = './trained_models/gta_vgg_wdgrl.pt'

        clf_model = GTAVGG11Net(9, pretrained=False).to(device)
        out_ftrs = clf_model.classifier[0].in_features  # should be 512 * 7 * 7
        clf_model.load_state_dict(torch.load(model_file))
        set_requires_grad(clf_model, False)

        feature_extractor = clf_model.feature_extractor
        discriminator = clf_model.classifier

    else:
        raise ValueError(f'Unknown model type {args.model}')

    critic = nn.Sequential(
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

    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    clf_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        batch_iterator = zip(loop_iterable(source_loader),
                             loop_iterable(target_loader))

        total_loss = 0
        for _ in trange(args.iterations, leave=False):
            (source_x, source_y), (target_x, _) = next(batch_iterator)
            # Train critic
            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            source_y = source_y.to(device)

            with torch.no_grad():
                h_s = feature_extractor(source_x).data.view(
                    source_x.shape[0], -1)
                h_t = feature_extractor(target_x).data.view(
                    target_x.shape[0], -1)
            for _ in range(args.k_critic):
                gp = gradient_penalty(critic, h_s, h_t)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma * gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()

            # Train classifier
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                source_features = feature_extractor(source_x).view(
                    source_x.shape[0], -1)
                target_features = feature_extractor(target_x).view(
                    target_x.shape[0], -1)

                source_preds = discriminator(source_features)
                clf_loss = clf_criterion(source_preds, source_y)
                wasserstein_distance = critic(source_features).mean() - critic(
                    target_features).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')
        torch.save(clf_model.state_dict(), out_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Domain adaptation using WDGRL')
    arg_parser.add_argument('--model',
                            type=str,
                            default='gta',
                            help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--iterations', type=int, default=10)
    arg_parser.add_argument('--epochs', type=int, default=16)
    arg_parser.add_argument('--k-critic', type=int, default=5)
    arg_parser.add_argument('--k-clf', type=int, default=1)
    arg_parser.add_argument('--gamma', type=float, default=10)
    arg_parser.add_argument('--wd-clf', type=float, default=1)
    args = arg_parser.parse_args()
    main(args)
