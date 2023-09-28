import torch.nn.functional as F
from torch import nn
import torch
import sklearn.metrics as skm
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import scipy
import math

import random
import numpy as np

from inspect import isfunction
from functools import partial
import math
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn
from torch import einsum
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch

def exists(x):
  return x is not None

def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d

class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, *args, **kwargs):
    return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
  return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
  return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings

class Block(nn.Module):
  def __init__(self, dim, dim_out, groups = 8):
    super().__init__()
    self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
    self.norm = nn.GroupNorm(groups, dim_out)
    self.act = nn.SiLU()

  def forward(self, x, scale_shift = None):
    x = self.proj(x)
    x = self.norm(x)

    if exists(scale_shift):
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    x = self.act(x)
    return x

class ResnetBlock(nn.Module):
  """https://arxiv.org/abs/1512.03385"""
  
  def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
    super().__init__()
    self.mlp = (
      nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
      if exists(time_emb_dim)
      else None
    )

    self.block1 = Block(dim, dim_out, groups=groups)
    self.block2 = Block(dim_out, dim_out, groups=groups)
    self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

  def forward(self, x, time_emb=None):
    h = self.block1(x)

    if exists(self.mlp) and exists(time_emb):
      time_emb = self.mlp(time_emb)
      h = rearrange(time_emb, "b c -> b c 1 1") + h

    h = self.block2(h)
    return h + self.res_conv(x)
  
class Attention(nn.Module):
  def __init__(self, dim, heads=4, dim_head=32):
    super().__init__()
    self.scale = dim_head**-0.5
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
    self.to_out = nn.Conv2d(hidden_dim, dim, 1)

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim=1)
    q, k, v = map(
        lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
    )
    q = q * self.scale

    sim = einsum("b h d i, b h d j -> b h i j", q, k)
    sim = sim - sim.amax(dim=-1, keepdim=True).detach()
    attn = sim.softmax(dim=-1)

    out = einsum("b h i j, b h d j -> b h i d", attn, v)
    out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
    return self.to_out(out)

class LinearAttention(nn.Module):
  def __init__(self, dim, heads=4, dim_head=32):
    super().__init__()
    self.scale = dim_head**-0.5
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

    self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                nn.GroupNorm(1, dim))

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim=1)
    q, k, v = map(
      lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
    )

    q = q.softmax(dim=-2)
    k = k.softmax(dim=-1)

    q = q * self.scale
    context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

    out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
    out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
    return self.to_out(out)

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(1, dim)

  def forward(self, x):
    x = self.norm(x)
    return self.fn(x)

class Unet(nn.Module):
  def __init__(
      self,
      dim,
      init_dim=None,
      out_dim=None,
      dim_mults=(1, 2, 4, 8),
      channels=3,
      with_time_emb=True,
      resnet_block_groups=8,
  ):
    super().__init__()

    # determine dimensions
    self.channels = channels

    init_dim = default(init_dim, dim // 3 * 2)
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))
    
    block_klass = partial(ResnetBlock, groups=resnet_block_groups)

    # time embeddings
    if with_time_emb:
      time_dim = dim * 4
      self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(dim),
        nn.Linear(dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim),
      )
    else:
      time_dim = None
      self.time_mlp = None

    # layers
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    num_resolutions = len(in_out)

    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)

      self.downs.append(
        nn.ModuleList(
          [
            block_klass(dim_in, dim_out, time_emb_dim=time_dim),
            block_klass(dim_out, dim_out, time_emb_dim=time_dim),
            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            Downsample(dim_out) if not is_last else nn.Identity(),
          ]
        )
      )

    mid_dim = dims[-1]
    self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
    self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= (num_resolutions - 1)

      self.ups.append(
        nn.ModuleList(
          [
            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            Upsample(dim_in) if not is_last else nn.Identity(),
          ]
        )
      )

    out_dim = default(out_dim, channels)
    self.final_conv = nn.Sequential(
      block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
    )

  def forward(self, x, time):
    # Returns the noise prediction from the noisy image x at time t
    # Inputs:
    #   x: noisy image tensor of size (batch_size, 3, 32, 32)
    #   t: time-step tensor of size (batch_size,)
    #   x[i] contains image i which has been added noise amount corresponding to t[i]
    # Returns:
    #   noise_pred: noise prediction made from the model, size (batch_size, 3, 32, 32)

    x = self.init_conv(x)

    t = self.time_mlp(time) if exists(self.time_mlp) else None

    h = []

    # downsample
    for block1, block2, attn, downsample in self.downs:
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      h.append(x)
      x = downsample(x)

    # bottleneck
    x = self.mid_block1(x, t)
    x = self.mid_attn(x)
    x = self.mid_block2(x, t)

    # upsample
    for block1, block2, attn, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim=1)
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      x = upsample(x)

    noise_pred = self.final_conv(x)
    return noise_pred

class DDPM():
    def __init__(self, seed=0, model_name = "diffusion", hidden_size = [256, 512, 256], epochs = 100, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, T=1000, reconstruction_t = 250, device = 'cpu', full_path=False):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.full_path = full_path
        
        self.T = T
        self.rec_t = reconstruction_t
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        betas = torch.linspace(0.0001, 0.01, T) # linear beta scheduling

        # Pre-calculate different terms for closed form of diffusion process
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)     
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)               
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        def forward_noise(x_0, t, drift = True):
            """ 
            Takes data point and a timestep as input and 
            returns the noisy version of it
            """
            noise = torch.randn_like(x_0) # epsilon

            noise.requires_grad_() # for the backward propagation of the NN
            sqrt_alphas_cumprod_t = torch.take(sqrt_alphas_cumprod, t.cpu()).to(device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            sqrt_one_minus_alphas_cumprod_t = torch.take(sqrt_one_minus_alphas_cumprod, t.cpu()).to(device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            # mean + variance
            if drift:
                return (sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)).to(torch.float32), noise.to(self.device)
            else: # variance only
                return (x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)).to(torch.float32), noise.to(self.device)
        
        def get_loss(model, x_0, t):
            # get the loss based on the input and timestep
            
            # get noisy sample
            x_noisy, noise = forward_noise(x_0, t)

            # predict the timestep
            noise_pred = model(x_noisy, t)

            # For the regression model, the target is t with mean squared error loss
            loss_fn = nn.MSELoss()
            
            loss = loss_fn(noise_pred, noise)

            return loss
        
        def p_sample(model, x, t):
            t_index = t[0]
            with torch.no_grad():
                betas_t = torch.take(betas, t.cpu()).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)
                sqrt_one_minus_alphas_cumprod_t = torch.take(sqrt_one_minus_alphas_cumprod, t.cpu()).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)
                sqrt_recip_alphas_t = torch.take(sqrt_recip_alphas, t.cpu()).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)
                
                p_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

                if t_index == 0:
                    sample = p_mean                       
                else:
                    posterior_variance_t = torch.take(posterior_variance, t.cpu()).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)  
                
                    z = torch.randn_like(x).to(self.device)
                    sample = p_mean #+ torch.sqrt(posterior_variance_t) * z

            return sample
        
        self.forward_noise = forward_noise
        self.sample = p_sample
        self.loss_fn = get_loss
        self.model = None
        
    def reconstruct(self, x, t):
        with torch.no_grad():
            b = x.shape[0]
            xs = [] 
            #x_noisy, _ = self.forward_noise(x, torch.full((b,), t, device=self.device).long())
            x_noisy = x
            for i in reversed(range(0, t)):
                x_noisy = self.sample(self.model, x_noisy, torch.full((b,), i, device=self.device).long())
                xs.append(x_noisy)
        return xs

    def fit(self, X_train, y_train = None):
        if self.model is None: # allows retraining
            self.model = Unet(
                dim=X_train.shape[-1],
                channels=3,
                dim_mults=(1, 2, 4, 8),
            ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = DataLoader(torch.from_numpy(X_train).float(), batch_size=self.batch_size, shuffle=True, drop_last=True)
        
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
                loss = self.loss_fn(self.model, x, t)
                
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
                
            train_losses.append(np.mean(np.array(loss_)))

            if epoch % 5 == 0:
                print(f"Epoch {epoch} Train Loss: {train_losses[len(train_losses)-1]}")
            if epoch > 50:
                if train_losses[len(train_losses)-1] > train_losses[len(train_losses)-40]:
                    break
        
        return self

    @torch.no_grad()
    def predict_score(self, X, reconstruction_t = None):
        test_loader = DataLoader(torch.from_numpy(X).float(), batch_size=500, shuffle=False, drop_last=False)
        preds = []
        self.model.eval()
        if reconstruction_t is not None:
            self.rec_t = reconstruction_t
        for x in test_loader:
            x = x.to(self.device)
            # predict the timestep based on x, or the probability of each class for the classification
            x_rec = self.reconstruct(x, self.rec_t)
            
            if not self.full_path:
                pred = ((x-x_rec[-1]) ** 2).mean(-1).mean(-1).mean(-1).squeeze().cpu().detach().numpy()
            else: 
                prev = x
                total = np.zeros((x.shape[0],))
                for rec in x_rec:
                    pred  = ((prev-rec) ** 2).mean(-1).mean(-1).mean(-1)
                    total += pred.squeeze().cpu().detach().numpy()
                    prev = rec
                pred = total

            
            preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        
        return preds