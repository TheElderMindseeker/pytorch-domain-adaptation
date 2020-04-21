import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, RandomCrop, ToTensor
from tqdm import tqdm

from models import GTANet, GTARes18Net, GTAVGG11Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.model == 'gta':
        model = GTANet().to(device)
        model_file = './trained_models/gta.pt'
    elif args.model == 'gta-res':
        model = GTARes18Net(9).to(device)
        model_file = './trained_models/gta_res.pt'
    elif args.model == 'gta-vgg':
        model = GTAVGG11Net(9).to(device)
        model_file = './trained_models/gta_vgg.pt'
    else:
        raise ValueError(f'Unknown model type {args.model}')

    dataset = ImageFolder('./t_data',
                          transform=Compose([
                              RandomCrop(224,
                                         pad_if_needed=True,
                                         padding_mode='reflect'),
                              ToTensor(),
                              Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]),
                          ]))
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=1,
                            pin_memory=True)

    model.load_state_dict(torch.load(model_file), strict=False)
    model.eval()

    total_accuracy = 0
    total_anomaly_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(dataloader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            predictions = y_pred.max(1)[1]
            total_accuracy += (predictions == y_true).float().mean().item()
            anml_true = y_true.cpu().apply_(lambda x: 0 if x == 5 else 1)
            anml_pred = predictions.cpu().apply_(lambda x: 0 if x == 5 else 1)
            total_anomaly_accuracy += (
                anml_pred == anml_true).float().mean().item()

    mean_accuracy = total_accuracy / len(dataloader)
    mean_anomaly_accuracy = total_anomaly_accuracy / len(dataloader)
    print(f'Accuracy on target data: {mean_accuracy:.4f}')
    print(f'Accuracy on target data (anomaly): {mean_anomaly_accuracy:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Test a model on target data')
    arg_parser.add_argument('--model',
                            type=str,
                            default='gta',
                            help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=32)
    args = arg_parser.parse_args()
    main(args)
