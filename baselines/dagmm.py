"""
Implementation of DAGMM (https://openreview.net/forum?id=BJJLHbb0-)
modified from the implementation from ADBench https://github.com/Minqi824/ADBench
"""
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from torch.utils.data import DataLoader

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s



class DAGMMModule(nn.Module):
    def __init__(self, input_size, n_gmm=2, z_dim=1):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMMModule, self).__init__()
        #Encoder network
        self.fc1 = nn.Linear(input_size, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        #Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, input_size)

        #Estimation network
        self.fc9 = nn.Linear(z_dim+2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma
      
class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.device = device

        # input data
        X_train = data
        self.input_size = X_train.shape[1]

        # dataloader
        self.train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                       batch_size=self.args.batch_size, shuffle=False, drop_last=True)

    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMMModule(self.input_size, self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.device, self.args.n_gmm)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                _, x_hat, z, gamma = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()

            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))

def eval(model, data, device, n_gmm, batch_size):
    """Testing the DAGMM model"""
    X_train = data['X_train']
    X_test = data['X_test']

    dataloader_train = DataLoader(torch.from_numpy(X_train).float(),
                                  batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(torch.from_numpy(X_test).float(),
                                 batch_size=batch_size, shuffle=False, drop_last=False)

    # evaluation mode
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, device, n_gmm)

    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0

        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
        for x in dataloader_train:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

            N_samples += x.size(0)

        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # Obtaining Labels and energy scores for test data
        energy_test = []
        for x in dataloader_test:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            sample_energy, cov_diag = compute.compute_energy(z, gamma, train_phi,
                                                             train_mu, train_cov,
                                                             sample_mean=False)

            energy_test.append(sample_energy.detach().cpu())

        energy_test = torch.cat(energy_test).numpy() # the output score

    return energy_test          

class DAGMM():
    '''
    PyTorch implementation of DAGMM from "https://github.com/mperezcarrasco/PyTorch-DAGMM"
    '''
    def __init__(self, seed, model_name='DAGMM', tune=False,
                 num_epochs=200, patience=50, lr=1e-4, lr_milestones=[50], batch_size=256,
                 latent_dim=1, n_gmm=4, lambda_energy=0.1, lambda_cov=0.005, device = None):
        '''
        The default batch_size is 1024
        The default latent_dim is 1
        The default lambda_cov is 0.005
        '''
        if device is None:       
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.seed = seed
        self.tune = tune

        # hyper-parameter
        class Args:
            pass

        self.args = Args()
        self.args.num_epochs = num_epochs
        self.args.patience = patience
        self.args.lr = lr
        self.args.lr_milestones = lr_milestones
        self.args.batch_size = batch_size
        self.args.latent_dim = latent_dim
        self.args.n_gmm = n_gmm
        self.args.lambda_energy = lambda_energy
        self.args.lambda_cov = lambda_cov

    def grid_search(self, X_train, y_train, ratio):
        '''
        implement the grid search for unsupervised models and return the best hyper-parameters
        the ratio could be the ground truth anomaly ratio of input dataset
        '''

        # set seed
        self.utils.set_seed(self.seed)
        # get the hyper-parameter grid (n_gmm, default=4)
        param_grid = [4, 6, 8, 10]

        # index of normal ana abnormal samples
        idx_a = np.where(y_train==1)[0]
        idx_n = np.where(y_train==0)[0]
        idx_n = np.random.choice(idx_n, int((len(idx_a) * (1-ratio)) / ratio), replace=True)

        idx = np.append(idx_n, idx_a) #combine
        np.random.shuffle(idx) #shuffle

        # valiation set (and the same anomaly ratio as in the original dataset)
        X_val = X_train[idx]
        y_val = y_train[idx]

        # fitting
        metric_list = []
        for param in param_grid:
            try:
                self.args.n_gmm = param
                model = TrainerDAGMM(self.args, X_train, self.device)
                model.train()

            except:
                metric_list.append(0.0)
                continue

            try:
                # model performance on the validation set
                data = {'X_train': X_train, 'X_test':X_val}

                score_val = eval(model.model, data, self.device, self.args.n_gmm, self.args.batch_size)
                metric = self.utils.metric(y_true=y_val, y_score=score_val, pos_label=1)
                metric_list.append(metric['aucpr'])

            except:
                metric_list.append(0.0)
                continue

        self.args.n_gmm = param_grid[np.argmax(metric_list)]

        print(f'The candidate hyper-parameter: {param_grid},',
              f' corresponding metric: {metric_list}',
              f' the best candidate: {self.args.n_gmm}')

        return self

    def fit(self, X_train, y_train=None, ratio=None):
        if sum(y_train) > 0 and self.tune:
            self.grid_search(X_train, y_train, ratio)
        else:
            pass

        print(f'using the params: {self.args.n_gmm}')

        # initialization
        self.model = TrainerDAGMM(self.args, X_train, self.device)
        # fitting
        self.model.train()

        return self

    def predict_score(self, X_train, X_test):
        data = {'X_train': X_train, 'X_test': X_test}

        # predicting
        score = eval(self.model.model, data, self.device, self.args.n_gmm, self.args.batch_size)
        return score