import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from tqdm import tqdm

import config
from gta import create_gta_dataloaders
from models import GTANet, GTARes18Net, GTAVGG11Net
from utils import GrayscaleToRgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
    dataset = MNIST(config.DATA_DIR / 'mnist',
                    train=True,
                    download=True,
                    transform=Compose([GrayscaleToRgb(),
                                       ToTensor()]))
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8 * len(dataset))]
    val_idx = shuffled_indices[int(0.8 * len(dataset)):]

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1,
                              pin_memory=True)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1,
                            pin_memory=True)
    return train_loader, val_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):

    if args.model == 'gta':
        train_loader, val_loader = create_gta_dataloaders(
            './data',
            transform=Compose([
                Resize((398, 224)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
        model = GTANet().to(device)
        model_path = './trained_models/gta_source.pt'
        params_to_update = model.parameters()
    elif args.model == 'gta-res':
        train_loader, val_loader = create_gta_dataloaders(
            './data',
            transform=Compose([
                Resize((398, 224)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
        model = GTARes18Net(9).to(device)
        model_path = './trained_models/gta_res_source.pt'
        params_to_update = list()
        for param in model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
    elif args.model == 'gta-vgg':
        train_loader, val_loader = create_gta_dataloaders(
            './data',
            transform=Compose([
                Resize((398, 224)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
        model = GTAVGG11Net(9).to(device)
        model_path = './trained_models/gta_vgg_source.pt'
        params_to_update = list()
        for param in model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
    else:
        raise ValueError(f'Unknown model type {args.model}')

    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))

    optim = torch.optim.Adam(params_to_update)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optim,
                                                  step_size=10,
                                                  gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_accuracy = do_epoch(model,
                                              train_loader,
                                              criterion,
                                              optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model,
                                              val_loader,
                                              criterion,
                                              optim=None)

        tqdm.write(
            f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
            f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Train a network on source dataset')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=64)
    arg_parser.add_argument('--model', type=str, default='gta')
    args = arg_parser.parse_args()
    main(args)
