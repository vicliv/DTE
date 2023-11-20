# %%
import numpy as np
import torch
import torch.distributions as dist

import matplotlib
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

matplotlib.rcParams.update({'font.size': 12})

from scipy.stats import invgamma

"""
We do not recommand using this code, it is not efficient and provides the same results
as kNN for anomaly detection, this is just to showcase how it works. Instead, use kNN
from the PyOD library with a parameter method="mean".
"""

# semi supervised train test split
def binning(t, T,  num_bins=30):
    return torch.maximum(torch.minimum(torch.floor(t*num_bins/T), torch.tensor(num_bins-1)), torch.tensor(0)).long()

def create_noisy_data(X, noise_std):
    noise = torch.randn_like(X) * noise_std
    return X + noise

def compute_pairwise_diff(X1, X2):
    #return torch.sqrt(torch.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    return X1[:, None, :] - X2[None, :, :]

def train_test_split_anomaly(X, y, train_split=0.5):
    indices = np.arange(len(X))
    normal_indices = indices[y == 0]
    anomaly_indices = indices[y == 1]

    train_size = round(train_split * normal_indices.size)
    train_indices, test_indices = normal_indices[:train_size], normal_indices[train_size:]
    test_indices = np.append(test_indices, anomaly_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

class DTENonParametric(object):
    def __init__(self, seed = 0, model_name = "DTE-NP", batch_size = 64, K=5, T=500):
        beta_0 = 0.0001
        beta_T = 0.01
        self.T = T
        self.K = K
        self.seed = seed
        self.T_range = np.arange(0, self.T)
        betas = torch.linspace(beta_0, beta_T, self.T)
        
        self.batch_size = batch_size

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) #std deviations
        self.model_name = model_name

    def compute_log_likelihood(self, X):
        N1, N2, dim = X.shape
        log_likelihood = torch.zeros((self.T, N1, N2))
        # loop because one shotting causes memory issues
        for t in range(self.T):
            loc = torch.zeros((dim))
            scale = torch.ones((dim)) * self.sqrt_one_minus_alphas_cumprod[t]
            dist_t = dist.Independent(dist.Normal(loc=loc, scale=scale), 1)
            #dist_t = dist.Normal(loc=0., scale=sqrt_one_minus_alphas_cumprod[t])
            log_likelihood[t, ...] = dist_t.log_prob(X)
        return log_likelihood

    def kernel_estimator(self, X_test, X_train, timestep=0, eval=False, verbose=False):
        _, dim = X_train.shape
        X_test = torch.from_numpy(X_test).float()
        X_train = torch.from_numpy(X_train).float()
        if eval:
            X_noisy = X_test.clone()
        else:
            X_noisy = create_noisy_data(X_test, self.sqrt_one_minus_alphas_cumprod[timestep])
            
        pairwise_diff = compute_pairwise_diff(X_noisy, X_train)

        log_p_t_given_y = torch.zeros((self.T, X_test.shape[0]))
    
        # non-parametric solution
        pairwise_norm_2 = torch.sum(pairwise_diff**2, axis=-1)

        min_norm_2 = (torch.topk(pairwise_norm_2, self.K, largest=False, axis=-1).values).mean(-1)

        density = torch.zeros((self.T, X_test.shape[0]))
        for i in range(min_norm_2.shape[0]):
            density[:,i] = torch.from_numpy(invgamma.logpdf((1. - self.alphas_cumprod), a=0.5*dim-1, \
                                loc=0, scale=(min_norm_2[i]/2))).float()
        
        density = density - density.logsumexp(0, keepdim=True)

        return log_p_t_given_y.exp().t(), density.exp().t()
    

    def nonparametric(self, X_test, X_train, batch_size=64, timestep=0, eval=False, verbose=False):
        num_batches = int(np.ceil(X_test.shape[0] / batch_size))

        p_t_given_y = torch.zeros((X_test.shape[0], self.T))
        density = torch.zeros((X_test.shape[0], self.T))
        for i in range(num_batches):
            if verbose:
                print('Batch {}/{}'.format(i+1, num_batches), end='\r')
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, X_test.shape[0])
            p_t_given_y[start_idx:end_idx, :], density[start_idx:end_idx, :] = \
                self.kernel_estimator(X_test[start_idx:end_idx, :], X_train, timestep=timestep, eval=eval, verbose=verbose)
        
        return p_t_given_y, density

    def plot_timestep_colorplot(self, p_t=None, dataset_name=""):
        self.dataset_name = dataset_name
        if p_t is None:
            p_t = torch.from_numpy(np.load('{}_p_t.npy'.format(self.dataset_name))).float()

        invgamma_p_t = torch.from_numpy(np.load('{}_invgamma_p_t.npy'.format(self.dataset_name))).float()

        colors = np.linspace(0, 1, 100)
        cmap = ListedColormap(plt.cm.get_cmap('viridis')(colors))
        
        p_t_mean = p_t.mean(1)

        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        ax = plt.axes(projection='3d')
        for t in range(30, self.T, 1):
            ax.plot((1. - self.alphas_cumprod).numpy(), np.repeat(t, p_t_mean.shape[1]), \
                    p_t_mean[t, :], color=cmap(float(t/self.T)), alpha=0.5)

        ax.set_xlabel(r'$\sigma^2_t$', fontsize=16, labelpad=10)
        ax.set_ylabel(r'$x_s, s \in \{0,1,\ldots,T\}$', fontsize=16, labelpad=10)
        #ax.set_zlabel('Density Value', fontsize=16, labelpad=10)
        ax.set_title(r'Analytical posterior distribution $p(\sigma^2_t | x_s)$', fontsize=18)
        ax.view_init(elev=30, azim=-60)

        plt.savefig('./{}_timestep_dist_analytical.pdf'.format(self.dataset_name), bbox_inches='tight')
        plt.close()

        #invgamma_p_t = invgamma_p_t / invgamma_p_t.sum(axis=-1, keepdim=True)
        invgamma_p_t = invgamma_p_t.mean(1)

        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        ax = plt.axes(projection='3d')
        for t in range(30, self.T, 1):
            ax.plot((1. - self.alphas_cumprod).numpy(), np.repeat(t, invgamma_p_t.shape[1]), \
                    invgamma_p_t[t,:], color=cmap(float(t/self.T)), alpha=0.5)

        ax.set_xlabel(r'$\sigma^2_t$', fontsize=16, labelpad=10)
        ax.set_ylabel(r'$x_s, s \in \{0,1,\ldots,T\}$', fontsize=16, labelpad=10)
        #ax.set_zlabel('Density Value', fontsize=16, labelpad=10)
        ax.zaxis.set_label_position("bottom") 

        ax.set_title(r'Non-parametric estimate of posterior distribution $p(\sigma^2_t | x_s)$', fontsize=18)
        ax.view_init(elev=30, azim=-60)
        #plt.tight_layout()
        plt.savefig('./{}_timestep_dist_invgamma_min.pdf'.format(self.dataset_name), bbox_inches='tight')
        plt.close()


    def compute_timestep_prediction(self, X_test, X_train):
        p_t = torch.zeros((self.T, X_test.shape[0], self.T))
        invgamma_p_t = torch.zeros((self.T, X_test.shape[0], self.T))
        for t in range(self.T):
            p_t[t, ...], invgamma_p_t[t, ...] = self.kernel_estimator(X_test, X_train, timestep=t)
            print('Completed t = {}/{}'.format(t, self.T), end='\r')
        print('\n')

        np.save('./{}_p_t.npy'.format(self.dataset_name), p_t.numpy())
        np.save('./{}_invgamma_p_t.npy'.format(self.dataset_name), invgamma_p_t.numpy())
        return

    def plot_timestep_prediction(self, p_t=None, invgamma_p_t=None):
        if p_t is None:
            p_t = torch.from_numpy(np.load('{}_p_t.npy'.format(self.dataset_name))).float()

        if invgamma_p_t is None:
            invgamma_p_t = torch.from_numpy(np.load('{}_invgamma_p_t.npy'.format(self.dataset_name))).float()

        #breakpoint()
        # timestep prediction
        pred = torch.argmax(p_t, axis=-1).float()
        pred_mean = torch.mean(pred, axis=1)
        pred_std = torch.std(pred, axis=1)

        invgamma_pred = torch.argmax(invgamma_p_t, axis=-1).float()
        invgamma_pred_mean = torch.mean(invgamma_pred, axis=1)
        invgamma_pred_std = torch.std(invgamma_pred, axis=1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.T_range, pred_mean,  color = "C0", label="Analytical", alpha=0.8)
        ax.fill_between(self.T_range, pred_mean-pred_std, pred_mean+pred_std, color="C0", alpha=0.2)
        ax.plot(self.T_range, invgamma_pred_mean,  color = "C3", label="Non-parametric", alpha=0.8)
        ax.fill_between(self.T_range, invgamma_pred_mean-invgamma_pred_std, invgamma_pred_mean+invgamma_pred_std, color="C3", alpha=0.2)

        ax.plot(self.T_range,self.T_range, "k--")
        #plt.xlabel("Variance")
        ax.set_xlabel("Ground truth timestep") 
        ax.set_ylabel("Average prediction")
        ax.set_title(r"Diffusion time prediction on vertebral")
        ax.legend()
        plt.savefig('./{}_timestep_pred_min.pdf'.format(self.dataset_name))
        plt.close()

        return

    def fit(self, X_train, y_train=None):
        self.X_train = X_train
        
        return self

    def predict_score(self, X_test):
        p_t, invgamma_p_t = self.nonparametric(X_test, self.X_train, batch_size=self.batch_size, timestep=0, eval=True)
        
        #preds = torch.argmax(invgamma_p_t,axis=-1).float().numpy()
        preds = np.matmul(invgamma_p_t, np.arange(0, self.T))

        return preds