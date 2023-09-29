import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import argparse

import torch
from torch import nn
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

parser = argparse.ArgumentParser(description='Prepare the embeddings of CIFAR10 and MNIST')
parser.add_argument('--model',default='resnet34', type=str,help='the model to take the embeddings: options are resnet34 and vicreg. If using vicreg, it will only generate CIFAR10 embeddings')

config = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""VicReg code found in https://github.com/augustwester/vicreg"""
class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )

    def forward(self, x):
        return self.network(x)

class VICReg(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()

        # the default ResNet has a 7x7 kernel with stride 2 as its initial
        # convolutional layer. this works for ImageNet but is too reductive for
        # CIFAR-10. we follow the SimCLR paper and replace it with a 3x3 kernel
        # with stride 1 and remove the max pooling layer.

        self.encoder = resnet18(num_classes=encoder_dim)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1)
        self.encoder.maxpool = nn.Identity()

        self.projector = Projector(encoder_dim, projector_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        y = self.encoder(x)
        return self.projector(y).chunk(2)
    
def embedding(img_size = 32, encoder=None, batch_size=64, dataset_name=None):
  transformation = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

  # FashionMNIST, CIFAR10, SVHN
  if dataset_name == 'MNIST':
    train_tensor = datasets.MNIST('data/', train=True, transform=transformation, download=True)

  elif dataset_name == 'CIFAR10':
    train_tensor = datasets.CIFAR10('data/', train=True, transform=transformation, download=True)

  elif dataset_name == 'SVHN':
    train_tensor = datasets.SVHN('data/', split='train', transform=transformation, download=True)

  else:
    raise NotImplementedError

  train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

  # encoder for extracting embedding
  encoder.eval()

  embeddings = []
  labels = []
  with torch.no_grad():
    for i, data in enumerate(tqdm(train_loader)):
      X, y = data
      if dataset_name == 'MNIST':
        X = X.repeat(1,3,1,1) # extent the channel if the picture is not colorful
      X = X.to(device)

      embeddings.append(encoder(X).squeeze().cpu().numpy())
      labels.append(y.numpy())

  fig = plt.figure(figsize=(14, 5))
  for i, target_label in enumerate(range(10)):
    X = np.vstack(embeddings)
    y = np.hstack(labels)

    print(f'Target Label: {target_label}')

    # have one class as anomaly
    idx_n = np.where(y != target_label)[0]
    idx_a = np.where(y == target_label)[0]

    y[idx_n] = 0
    y[idx_a] = 1

    path = "vision/data/" + config.model + "/"
    
    if not os.path.exists(path):
        os.makedirs(path)
    print(f'Normal samples: {sum(y==0)}, Anomalies: {sum(y==1)}')
    np.savez_compressed(os.path.join(path, dataset_name + '_' + str(target_label) + '.npz'), X=X, y=y)

set_seed(42)

encoder_name = config.model
img_size = 32

if encoder_name == 'resnet34':
  # resnet34 pretrained on the ImageNet (embedding dimension: 512)
  encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
  encoder = nn.Sequential(*list(encoder.children())[:-1])
  encoder.to(device)
  
  for dataset_name in ['MNIST', 'CIFAR10']:
    embedding(img_size=img_size, encoder=encoder, dataset_name=dataset_name)
    
elif encoder_name == 'vicreg':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  encoder_dim, projector_dim = 512, 1024
  model = VICReg(encoder_dim, projector_dim)
  cp = torch.load('checkpoint.pt')

  model.load_state_dict(cp["model_state_dict"])
  encoder = model.encoder.to(device)
  
  for dataset_name in ['CIFAR10']:
    embedding(img_size=img_size, encoder=encoder, dataset_name=dataset_name)


