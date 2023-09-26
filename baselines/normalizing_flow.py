import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.bias = nn.Parameter(torch.randn(1) * 0.01)
        self.scale = nn.Parameter(torch.randn(1, dim) * 0.01)

    def forward(self, z):
        activation = torch.tanh(torch.mm(z, self.weight.t()) + self.bias)
        return z + activation * self.scale

    def log_abs_det_jacobian(self, z):
        z = torch.mm(z, self.weight.t()) + self.bias
        psi = (1 - torch.tanh(z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-6)

class NormalizingFlow(nn.Module):
    def __init__(self, dim, K):
        super(NormalizingFlow, self).__init__()
        self.transforms = nn.Sequential(*[PlanarFlow(dim) for _ in range(K)])

    def forward(self, z):
        log_det_jacobians = []
        for transform in self.transforms:
            log_det_jacobian = transform.log_abs_det_jacobian(z)
            z = transform(z)
            log_det_jacobians.append(log_det_jacobian)
        return z, torch.stack(log_det_jacobians, dim=1).sum(1)

class FlowModel:
    def __init__(self, seed = 0, model_name = "Flow", K = 10, device = None):        
        self.K = K
        
        if device is None:       
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.flow = None

    def fit(self, X_train, y_train = None, epochs = 200, batch_size = 64, lr=2e-3):
        if self.flow is None:
            self.flow = NormalizingFlow(X_train.shape[-1], self.K).to(self.device)
        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=1e-4)
        train_loader = DataLoader(torch.from_numpy(X_train).float().to(self.device),
                                          batch_size=batch_size, shuffle=True, drop_last=False)

        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)

        for epoch in range(epochs):
            train_loss = []
            for x in train_loader:
                z, log_det_jacobian = self.flow(x)
                base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
                log_prob = base_log_prob + log_det_jacobian
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            if epoch % 5 == 0:
                print("Epochs " + str(epoch) + "\tTrain loss:" + str(np.array(train_loss).mean()))
        return self
        

    def predict_score(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            z, log_det_jacobian = self.flow(X)
            base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
            log_prob = base_log_prob + log_det_jacobian
        return -log_prob.cpu().numpy().mean(1)