"""
Code adapted from the official implementation of GOAD (https://arxiv.org/abs/2005.02359) 
provided at https://github.com/lironber/GOAD
"""

import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)

class netC1(nn.Module):
    def __init__(self, d, ndf, nc):
        super(netC1, self).__init__()
        self.trunk = nn.Sequential(
        nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.trunk(input)
        ce = self.head(tc)
        return tc, ce

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss

class TransClassifierTabular():
    def __init__(self, d_out = 32, m = 1, n_rots = 256, n_epoch = 1, ndf = 8, batch_size = 64, lmbda = 0.1, eps=0, lr = 0.001):
        self.m = m
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.ndf = ndf
        self.n_rots = n_rots
        self.d_out = d_out
        self.eps = eps

        self.n_epoch = n_epoch
        self.netC = netC1(self.d_out, self.ndf, self.n_rots).cuda()

        weights_init(self.netC)
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=lr, betas=(0.5, 0.999))


    def fit_trans_classifier(self, train_xs):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        celoss = nn.CrossEntropyLoss()

        for epoch in range(self.n_epoch):
            self.netC.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_xs), self.batch_size):
                self.netC.zero_grad()
                batch_range = min(self.batch_size, len(train_xs) - i)
                train_labels = labels
                if batch_range == len(train_xs) - i:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_xs) - i, self.n_rots)).long().cuda()
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(train_xs[rp[idx]]).float().cuda()
                tc_zs, ce_zs = self.netC(xs)
                sum_zs = sum_zs + tc_zs.mean(0)
                tc_zs = tc_zs.permute(0, 2, 1)

                loss_ce = celoss(ce_zs, train_labels)
                er = self.lmbda * tc_loss(tc_zs, self.m) + loss_ce
                er.backward()
                self.optimizerC.step()
                n_batch += 1

            means = sum_zs.t() / n_batch
            self.means = means.unsqueeze(0)
            
    def evaluation(self, x_test):
        self.netC.eval()
        with torch.no_grad():
            val_probs_rots = np.zeros((len(x_test), self.n_rots))
            for i in range(0, len(x_test), self.batch_size):
                batch_range = min(self.batch_size, len(x_test) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_test[idx]).float().cuda()
                zs, fs = self.netC(xs)
                zs = zs.permute(0, 2, 1)
                diffs = ((zs.unsqueeze(2) - self.means) ** 2).sum(-1)

                diffs_eps = self.eps * torch.ones_like(diffs)
                diffs = torch.max(diffs, diffs_eps)
                logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)

                val_probs_rots[idx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

            val_probs_rots = val_probs_rots.sum(1)
        return val_probs_rots
    

class GOAD():
    def __init__(self, seed=0, model_name = "GOAD", d_out = 32, m = 1, n_rots = 256, n_epoch = 1, ndf = 8, batch_size = 64, lmbda = 0.1, eps=0, lr = 0.001, device=None):
        self.m = m
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.ndf = ndf
        self.n_rots = n_rots
        self.d_out = d_out
        self.eps = eps
        self.lr = lr

        self.n_epoch = n_epoch

        if device is None:       
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.seed = seed
        
        self.model = None
    
    def fit(self, X_train, y_train = None):
        if self.model is None:
            self.model = TransClassifierTabular(d_out = self.d_out, m = self.m, n_rots = self.n_rots, n_epoch = self.n_epoch, ndf = self.ndf, batch_size = self.batch_size, lmbda = self.lmbda, eps=self.eps, lr = self.lr)
        
        n_train, n_dims = X_train.shape
        self.rots_trans = np.random.randn(self.n_rots, n_dims, self.d_out)
        
        X_train = np.stack([X_train.dot(rot) for rot in self.rots_trans], 2)
        
        self.model.fit_trans_classifier(X_train)

        return self

    def predict_score(self, X):
        X = np.stack([X.dot(rot) for rot in self.rots_trans], 2)
         
        return self.model.evaluation(X)