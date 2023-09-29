import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import torch
from torch import nn
import random

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Prepare the embeddings of VisA')
parser.add_argument('--data-folder',default='./VisA_pytorch/', type=str,help='the path of the processed VisA dataset')
parser.add_argument('--model',default='resnet34', type=str,help='the model to take the embeddings: options are resnet34 and vicreg')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

config = parser.parse_args()
from torchvision.models import resnet34

split_type = "1cls"
data_folder = config.data_folder
save_folder = os.path.join(data_folder, split_type)

data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if config.model == "resnet34":
    encoder = resnet34(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(encoder.children())[:-1])
    encoder.to(device)
elif config.model == "vicreg":
    encoder = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    encoder.to(device)
else:
    raise ValueError("unrecognized model")
    
for data in data_list:
    train_folder = os.path.join(save_folder, data, 'train')
    test_folder = os.path.join(save_folder, data, 'test')
                   
    transform = transforms.Compose([transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(300),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])

    dataset = ImageFolder(root=train_folder, transform=transform)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)

    # encoder for extracting embedding
    encoder.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for i, d in enumerate(tqdm(train_loader)):
            X, y = d
            X = X.to(device)

            embeddings.append(encoder(X).squeeze().cpu().numpy())
            labels.append(y)

    dataset = ImageFolder(root=test_folder, transform=transform)

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)

    # encoder for extracting embedding
    encoder.eval()

    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader)):
            X, y = d
            X = X.to(device)

            embeddings.append(encoder(X).squeeze().cpu().numpy())
            labels.append(1-y)

    X = np.vstack(embeddings)
    y = np.hstack(labels)             
    
    path = "vision/data/" + config.model + "/"
    
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez_compressed(path + data +'.npz', X=X, y=y)