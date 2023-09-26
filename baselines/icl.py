"""
Implementation of ICL,a method proposed in the paper (https://openreview.net/forum?id=_hszZbt46bT)
The code is adapted from the official implementation in the supplemental material of the openreview forum.
"""

import torch
import numpy as np
import random
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def scores_calc_internal(query, positive,no_negatives,tau):
    pos_multiplication = (query * positive).sum(dim=2).unsqueeze(2).to(device)
    if no_negatives <= query.shape[1]:
        negative_index = random.sample(range(0, query.shape[1]), no_negatives)
    else:
        negative_index = random.sample(range(0, query.shape[1]), query.shape[1])
    neg_multiplication = torch.matmul(query, positive.permute(0, 2, 1)[:, :,negative_index])
    # Removal of the diagonals
    identity_matrix = torch.eye(np.shape(query)[1]).unsqueeze(0).repeat(np.shape(query)[0], 1,
                                                                        1)[:, :, negative_index].to(device)
    neg_multiplication.masked_fill_(identity_matrix == 1, -float('inf'))  # exp of -inf=0
    logits = torch.cat((pos_multiplication, neg_multiplication), dim=2).to(device)
    logits=logits/tau
    return (logits)


def take_per_row_complement(A, indx, num_elem=3):
    all_indx = indx[:,None] + np.arange(num_elem)

    all_indx_complement=[]
    for row in all_indx:
        complement=a_minus_b(np.arange(A.shape[2]),row)
        all_indx_complement.append(complement)
    all_indx_complement=np.array(all_indx_complement)
    return (A[:,np.arange(all_indx.shape[0])[:,None],all_indx],A[:,np.arange(all_indx.shape[0])[:,None],all_indx_complement])

def positive_matrice_builder(dataset, kernel_size):
    dataset = torch.squeeze(dataset, 2)
    if kernel_size != 1:
        indices = np.array((range(dataset.shape[1])))[:-kernel_size + 1]
    else:
        indices = np.array((range(dataset.shape[1])))
    dataset = torch.unsqueeze(dataset, 1)
    dataset = dataset.repeat(1, dataset.shape[2], 1)

    matrice,complement_matrice = take_per_row_complement(dataset, indices, num_elem=kernel_size)
    return (matrice,complement_matrice)


def take_per_row(A, indx, num_elem=2):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[:, np.arange(all_indx.shape[0])[:, None], all_indx]


def f1_calculator(classes, losses):
    df_version_classes = pd.DataFrame(data=classes)
    df_version_losses = pd.DataFrame(losses).astype(np.float64)
    Na = df_version_classes[df_version_classes.iloc[:, 0] == 1].shape[0]
    anomaly_indices = df_version_losses.nlargest(Na, 0).index.values
    picked_anomalies = df_version_classes.iloc[anomaly_indices]
    true_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 1].shape[0]
    false_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 0].shape[0]
    f1 = true_pos / (true_pos + false_pos)
    return (f1)


def a_minus_b (a,b):
    sidx = b.argsort()
    idx = np.searchsorted(b, a, sorter=sidx)
    idx[idx == len(b)] = 0
    out = a[b[sidx[idx]] != a]
    return out

class DatasetBuilder(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': self.data[idx], 'index': idx}
        return sample

class encoder_a(nn.Module):
    def __init__(self, kernel_size,hdn_size,d):
        super(encoder_a, self).__init__()
        self.fc1 = nn.Linear(d-kernel_size, hdn_size) #F network
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(hdn_size, hdn_size*2)
        self.activation2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hdn_size*2, hdn_size)
        self.activation3 = nn.LeakyReLU(0.2)
        self.batchnorm_1 = nn.BatchNorm1d(d-kernel_size+1)
        self.batchnorm_2 = nn.BatchNorm1d(d-kernel_size+1)
        self.fc1_y = nn.Linear(kernel_size, int(hdn_size/4)) #G network
        self.activation1_y = nn.LeakyReLU(0.2)
        self.fc2_y = nn.Linear(int(hdn_size/4), int(hdn_size/2))
        self.activation2_y = nn.LeakyReLU(0.2)
        self.fc3_y = nn.Linear(int(hdn_size/2), hdn_size)
        self.activation3_y = nn.LeakyReLU(0.2)
        self.kernel_size = kernel_size
        self.batchnorm1_y=nn.BatchNorm1d(d-kernel_size+1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y,x = positive_matrice_builder(x, self.kernel_size)
        x = self.activation1(self.fc1(x))
        x=self.batchnorm_1(x)
        x = self.activation2(self.fc2(x))
        x=self.batchnorm_2(x)
        x = self.activation3(self.fc3(x))
        y = self.activation1_y(self.fc1_y(y))
        y=self.batchnorm1_y(y)
        y = self.activation2_y(self.fc2_y(y))
        y = self.activation3_y(self.fc3_y(y))
        x=nn.functional.normalize(x,dim=1)
        y=nn.functional.normalize(y,dim=1)
        x=nn.functional.normalize(x,dim=2)
        y=nn.functional.normalize(y,dim=2)
        return (x, y)

class ICL():
    def __init__(self, seed=0, model_name="ICL", num_epochs = 2000, no_batchs = 3000, no_negatives=1000, temperature=0.1, lr=0.001, device=None):
        self.seed = seed
        
        if device is None:       
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = device
        
        self.num_epochs = num_epochs
        self.no_btchs = no_batchs
        self.no_negatives=no_negatives
        self.temperature=temperature
        self.lr=lr
        self.faster_version="no"
        
        self.models = []
        
        self.perms = []

    def fit(self,X_train, y_train=None):
        train = X_train
        train = torch.as_tensor(train, dtype=torch.float)
        d = train.shape[1]
        n = train.shape[0]
        if self.faster_version=='yes':
            num_permutations = min(int(np.floor(100 / (np.log(n) + d)) + 1),2)
        else:
            num_permutations=int(np.floor(100/(np.log(n)+d))+1)
        num_permutations = 1
        print("going to run for: ", num_permutations, ' permutations')
        hiddensize = 200
        if d <= 40:
            kernel_size = 2
            stop_crteria = 0.001
        if 40 < d and d <= 160:
            kernel_size = 10
            stop_crteria = 0.01
        if 160 < d:
            kernel_size = d - 150
            stop_crteria = 0.01
        for permutations in range(num_permutations):
            if num_permutations > 1:
                random_idx = torch.randperm(train.shape[1])
                self.perms.append(random_idx)
                train = train[:, random_idx]               
       
            dataset_train = DatasetBuilder(train)
            model_a = encoder_a(kernel_size, hiddensize, d).to(device)
            self.models.append(model_a)
            criterion = nn.CrossEntropyLoss()
            optimizer_a = torch.optim.Adam(model_a.parameters(), lr=self.lr)
            trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                        shuffle=True, num_workers=0, pin_memory=True)
            ### training
            for epoch in range(self.num_epochs):
                model_a.train()
                running_loss = 0
                for i, sample in enumerate(trainloader, 0):
                    model_a.zero_grad()
                    pre_query = sample['data'].to(device)
                    pre_query = torch.unsqueeze(pre_query, 1)
                    pre_query, positives_matrice = model_a(pre_query)
                    scores_internal = scores_calc_internal(pre_query, positives_matrice,self.no_negatives,self.temperature).to(device)
                    scores_internal = scores_internal.permute(0, 2, 1)
                    correct_class = torch.zeros((np.shape(scores_internal)[0], np.shape(scores_internal)[2]),
                                                dtype=torch.long).to(device)
                    loss = criterion(scores_internal, correct_class).to(device)
                    loss.backward()
                    optimizer_a.step()
                    running_loss += loss.item()
                if (running_loss / (i + 1) < stop_crteria):
                    break
                if n<2000:
                    if (epoch + 1) % 100 == 0:
                        print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                else:
                    if (epoch + 1) % 10 == 0:
                        print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
        return self
        
    def predict_score(self, test_X):
        test = torch.as_tensor(test_X, dtype=torch.float)
        
        test_losses_contrastloss = torch.zeros(test.shape[0],dtype=torch.float).to(device)
        
        
        for i, model in enumerate(self.models):
            if len(self.perms) > 0:
                test = test[:, self.perms[i]]
            dataset_test = DatasetBuilder(test)
            testloader = DataLoader(dataset_test, batch_size=self.no_btchs,
                            shuffle=True, num_workers=0, pin_memory=True)
            
            model.eval()
            criterion_test = nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                for i, sample in enumerate(testloader, 0):
                    pre_query = sample['data'].to(device)
                    indexes = sample['index'].to(device)
                    pre_query_test = torch.unsqueeze(pre_query, 1)  # batch X feature X 1
                    pre_query_test, positives_matrice_test = model(pre_query_test)
                    scores_internal_test = scores_calc_internal(pre_query_test, positives_matrice_test,self.no_negatives,self.temperature).to(device)
                    scores_internal_test = scores_internal_test.permute(0, 2, 1)
                    correct_class = torch.zeros((np.shape(scores_internal_test)[0], np.shape(scores_internal_test)[2]),
                                                dtype=torch.long).to(device)
                    loss_test = criterion_test(scores_internal_test, correct_class).to(device)
                    test_losses_contrastloss[indexes] += loss_test.mean(dim=1).to(device)
        return test_losses_contrastloss.cpu().detach().numpy()
        
    
    