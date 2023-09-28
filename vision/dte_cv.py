import torch.nn.functional as F
from torch import nn
import torch
import sklearn.metrics as skm
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50
  
def binning(t, T= 300, num_bins = 30, device = 'cpu'):
    """ 
    Gives the bin number for a given t based on T (maximum) and the number of bins
    This is floor(t*num_bins/T) bounded by 0 and T-1
    """
    return torch.maximum(torch.minimum(torch.floor(t*num_bins/T).to(device), torch.tensor(num_bins-1).to(device)), torch.tensor(0).to(device)).long()

class DTE():
    def __init__(self, seed = 0, model_name = "DTE", hidden_size = [256, 512, 256], epochs = 400, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=400, num_bins=7, device = None):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.T = T
        self.num_bins = num_bins
        
        if device is None:       
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.seed = seed
        
        betas = torch.linspace(0.0001, 0.01, T) # linear beta scheduling

        # Pre-calculate different terms for closed form of diffusion process
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        def forward_noise(x_0, t, drift = False):
            """ 
            Takes data point and a timestep as input and 
            returns the noisy version of it
            """
            noise = torch.randn_like(x_0) # epsilon

            noise.requires_grad_() # for the backward propagation of the NN
            sqrt_alphas_cumprod_t = torch.take(sqrt_alphas_cumprod, t.cpu()).to(self.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            sqrt_one_minus_alphas_cumprod_t = torch.take(sqrt_one_minus_alphas_cumprod, t.cpu()).to(self.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            # mean + variance
            if drift:
                return (sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)).to(torch.float32)
            else: # variance only
                return (x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)).to(torch.float32)
        
        self.forward_noise = forward_noise
        self.model = None
    
    def compute_loss(self, x, t):
        pass

    def fit(self, X_train, y_train = None, X_test = None, y_test = None, verbose=False):
        if self.model is None: # allows retraining
                self.model = resnet50()
                self.model.fc = nn.Linear(self.model.fc.in_features, self.num_bins)
                
                self.model.fc = nn.Sequential(
                    self.model.fc,
                    nn.Softmax(1),
                )
                
                self.model = self.model.to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = DataLoader(torch.from_numpy(X_train).float(), batch_size=self.batch_size, shuffle=True, drop_last=False)
        
        train_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            loss_ = []
            
            for x in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()

                # sample t uniformly
                t = torch.randint(0, self.T, (x.shape[0],), device=self.device).long()

                # compute the loss
                loss = self.compute_loss(x, t)
                
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
                
            train_losses.append(np.mean(np.array(loss_)))

            if epoch % 1 == 0 and verbose:
                if X_test is not None and y_test is not None:
                    print(roc_auc_score(y_true=y_test, y_score=self.predict_score(X_test)))
                print(f"Epoch {epoch} Train Loss: {train_losses[len(train_losses)-1]}")
        
        return self

    def predict_score(self, X):
        test_loader = DataLoader(torch.from_numpy(X).float(), batch_size=100, shuffle=False, drop_last=False)
        preds = []
        self.model.eval()
        for x in test_loader:
            # predict the timestep based on x, or the probability of each class for the classification
            pred_t = self.model(x.to(self.device).to(torch.float32))
            preds.append(pred_t.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        
        if self.num_bins > 1:
            #preds = np.argmax(preds, axis=1)
            
            # compute mean prediction over all bins
            preds = np.matmul(preds, np.arange(0, preds.shape[-1]))
        else:
            preds = preds.squeeze()
        
        return preds
  
class DTECategorical(DTE):
    def __init__(self, seed = 0, model_name = "DTE_categorical", hidden_size = [256, 512, 256], epochs = 400, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=400, num_bins=7):
        if num_bins < 2:
            raise ValueError("num_bins must be greater than or equal to 2")
        
        super().__init__(seed, model_name, hidden_size, epochs, batch_size, lr, weight_decay, T, num_bins)
        
        
    def compute_loss(self, x_0, t):
        # get the loss based on the input and timestep
        
        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the timestep
        t_pred = self.model(x_noisy)
        
        # For the categorical model, the target is the binned t with cross entropy loss
        target = binning(t, T = self.T, device = self.device,  num_bins = self.num_bins)

        loss = nn.CrossEntropyLoss()(t_pred, target)

        return loss

class DTEInverseGamma(DTE):
    def __init__(self, seed = 0, model_name = "DTE_inverse_gamma", hidden_size = [256, 512, 256], epochs = 400, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=400):        
        super().__init__(seed, model_name, hidden_size, epochs, batch_size, lr, weight_decay, T, 1)
        
        
    def compute_loss(self, x_0, t):
        # get the loss based on the input and timestep
        _, dim = x_0.shape
        eps = 1e-5
        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the inv gamma parameter
        sqrt_beta_pred = self.model(x_noisy)
        beta_pred = torch.pow(sqrt_beta_pred, 2).squeeze()

        var_target = (1. - self.alphas_cumprod[t.cpu()]).to(self.device)
        log_likelihood = (0.5 * dim - 1) * torch.log(beta_pred + eps) - beta_pred / (var_target)
        loss = -log_likelihood.mean()
        
        return loss

class DTEGaussian(DTE):
    def __init__(self, seed = 0, model_name = "DTE_gaussian", hidden_size = [256, 512, 256], epochs = 400, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=400):        
        super().__init__(seed, model_name, hidden_size, epochs, batch_size, lr, weight_decay, T, 0)
        
    def compute_loss(self, x_0, t):
        # get the loss based on the input and timestep
        
        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the timestep
        t_pred = self.model(x_noisy)
        
        t_pred = t_pred.squeeze()
        target = t.float()
        
        loss = nn.MSELoss()(t_pred, target)
        
        return loss
    

class DTEBagging():
    def __init__(self, num_bags = 5, hidden_size = [256, 512, 256], epochs = 200, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=300, num_bins=7):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.T = T
        self.num_bins = num_bins

        self.num_bags = num_bags
        
        self.models = []
    
    def fit(self, X_train, y_train, X_test, Y_test):
        for _ in range(self.num_bags):
            if self.num_bags > 1:
                indices = np.arange(len(X_train))
                random_idx = np.random.choice(indices, size = len(indices))
                X_train = X_train[random_idx, :]           
       
            model = DTE(hidden_size = self.hidden_size, epochs = self.epochs, batch_size = self.batch_size, lr = self.lr, weight_decay = self.weight_decay, T=self.T, num_bins=self.num_bins)
            self.models.append(model)
            
            model.fit(X_train=X_train, y_train=y_train, X_test = X_test, Y_test = Y_test)
        
        return self
      
    def predict_score(self, X):
        total = []
        
        # compute prediction for all models
        for model in self.models:
            total.append(model.predict_score(X))
        
        # sum the predictions
        pred = np.stack(total)
        preds = np.sum(pred, axis=0)

        return preds