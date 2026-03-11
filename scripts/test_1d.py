import torch
import sys, os
from argparse import Namespace
import numpy as np

# Add CNEEP_v2 root to path so 'models' and 'utils' can be imported
sys.path.append(os.getcwd())
# Add data/AMB to path so 'generate_trajectories_1d' can be imported
sys.path.append(os.path.join(os.getcwd(), 'data', 'AMB'))

from generate_trajectories_1d import ActiveModelB1D
from utils.sampler import CartesianSeqSampler
from models.train_1d import train_1d as train
from models.validate_1d import validate_1d as validate
from models.CNEEP_1D import CNEEP
from models.UNEEP_1D import CNEEP as UNEEP

# Setup
opt = Namespace()
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
opt.alpha = -0.5
opt.lam = 0.0
opt.threshold = 0.01
opt.positional = False
opt.latent_size = 10
opt.n_iter = 2
opt.train_batch_size = 128
opt.test_batch_size = 128
opt.video_batch_size = 128
opt.n_hidden = 512
opt.lr = 1e-3
opt.wd = 1e-5
opt.n_layer = 4
opt.n_channel = 32
opt.input_shape = (256,)
opt.M = 1
opt.seq_len = 2
opt.time_step = 0.01

print(f"Device: {opt.device}")

# Generate data
amb_params = dict(Lx=256, dx=1.0, a=0.25, b=0.25, kappa=4.0, lam=1.0, D=0.1, dt=0.01, smooth=False)
model_amb = ActiveModelB1D(**amb_params)
print("Generating 100 steps of 1D AMB data...")
traj_train = model_amb.generate_trajectory(n_steps=100, burn_in=100) # shorter burn-in for test
opt.L = traj_train.shape[0]

train_video = torch.from_numpy(traj_train).float()
train_video = train_video.unsqueeze(0).unsqueeze(2)   # (1, L, 1, Lx)

mean = torch.mean(train_video)
std  = torch.std(train_video)
transform = lambda x: (x - mean) / std

print(f"Train video shape: {train_video.shape}")

# Test CNEEP_1D
opt.n_layer = 4
model = CNEEP(opt).to(opt.device)
optim = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.wd)
train_sampler = CartesianSeqSampler(opt.M, opt.L, opt.seq_len, opt.train_batch_size, device=opt.device)

print("Training CNEEP_1D...")
for i in range(opt.n_iter):
    loss, R = train(opt, model, optim, train_video, train_sampler, transform)
    print(f"Iter {i+1}: Loss {loss:.4f}, R {R:.4f}")

# Test UNEEP_1D
model_u = UNEEP(opt).to(opt.device)
optim_u = torch.optim.Adam(model_u.parameters(), opt.lr, weight_decay=opt.wd)
train_sampler_u = CartesianSeqSampler(opt.M, opt.L, opt.seq_len, opt.train_batch_size, device=opt.device)

print("Training UNEEP_1D...")
for i in range(opt.n_iter):
    loss, R = train(opt, model_u, optim_u, train_video, train_sampler_u, transform)
    print(f"Iter {i+1}: Loss {loss:.4f}, R {R:.4f}")

print("Validation pass (CNEEP_1D)...")
test_sampler = CartesianSeqSampler(opt.M, opt.L, opt.seq_len, opt.test_batch_size, device=opt.device, train=False)
ret, maps, _ = validate(opt, model, train_video, test_sampler, transform)
print(f"ret shape: {ret.shape}, maps shape: {maps.shape}")

print("Test Success!")
