from argparse import Namespace
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms.functional as TF
from datetime import datetime

###########################################################################

#
# Hyper parameters
#
opt = Namespace()
opt.device = "cpu"

# alpha-NEEP parameter
opt.alpha   = -0.5
# Masking Regularization parameters
opt.lam     = 0.0
opt.threshold = 0.01

opt.latent_size = 10

# gradient descent parameters
opt.n_iter = 1
opt.train_batch_size = 500
opt.test_batch_size = 10000
opt.video_batch_size = 400
opt.n_hidden = 512
opt.lr = 1e-3
opt.wd = 1e-5

opt.record_freq = 1000
opt.seed = 3

# dataset configurations
opt.n_layer = 4
opt.n_channel = 32
opt.input_shape = (144, 144)
opt.M = 1
opt.L = 500
opt.seq_len = 2
opt.time_step = 0.01

torch.manual_seed(opt.seed)


#
# path fot results
#
data_folder = "data"
result_folder = "results"
current_result_folder = f"{result_folder}/{datetime.now().strftime("%Y%m%d-%H%M%S")}"
os.makedirs(current_result_folder)

current_checkpoint_path = f"{current_result_folder}/model_parameter.pth.tar"


if not os.path.exists(result_folder): os.makedirs(result_folder)


###########################################################################
# After this line, you generally don't need to change anything
###########################################################################



from utils.sampler import CartesianSampler, CartesianSeqSampler
from scipy import stats
from models.CNEEP_0 import CNEEP
from models.train import train
from models.validate import validate

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib as mpl

#
# Loading MIPS data (single video version)
#
video_frames, _, _ = torchvision.io.read_video(
    data_folder + "/MIPS.mp4",
    pts_unit='sec', start_pts=10)
video_tensor = video_frames.permute(0, 3, 1, 2).float()
train_video = video_tensor[:, 0, 3:147, 3:147]
train_video = train_video.unsqueeze(0)
train_video = train_video.unsqueeze(2)
print(f"Video tensor shape: {train_video.shape}")
mean    = torch.mean(train_video)
std     = torch.std(train_video)
transform = lambda x: (x - mean) / std

#
# Building our model
#
model = CNEEP(opt)
model = model.to(opt.device)
optim = torch.optim.Adam(
    model.parameters(), opt.lr, weight_decay=opt.wd)
train_sampler = CartesianSeqSampler(
    opt.M, opt.L, opt.seq_len, opt.train_batch_size, device=opt.device
)
test_sampler = CartesianSeqSampler(
    opt.M, opt.L, opt.seq_len, opt.test_batch_size, device=opt.device,
    train=False
)

#
# Training the model
#

train_losses = []
R_values = []
valid_losses = []
best_valid_loss = float('inf')
for i in tqdm(range(1, opt.n_iter + 1)) :
    train_loss, R_value = train(
        opt, model, optim, train_video, train_sampler, transform
    )
    train_losses.append(train_loss)
    R_values.append(R_value)

    _, _, valid_loss = validate(
        opt, model, train_video, test_sampler, transform
    )
    valid_losses.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        state = {
            'epoch': i,
            'settings': opt.__dict__,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
        }
        torch.save(state, current_checkpoint_path)

model.load_state_dict(
    torch.load(current_checkpoint_path)['state_dict'])
plt.plot(train_losses)
plt.savefig(f"{current_result_folder}/train_loss.png")
plt.clf()
plt.plot(valid_losses)
plt.savefig(f"{current_result_folder}/valid_loss.png")
plt.clf()
plt.plot(R_values)
plt.savefig(f"{current_result_folder}/R_values.png")
plt.clf()

#
# PCA study
#
latent_results = []
hooks = []
def hook_latent(module, input, output):
    latent_results.append(output.cpu().detach().numpy())
hooks.append(
    model._modules.get("latent")
    .register_forward_hook(hook_latent)
)
test_sampler = CartesianSeqSampler(
    1, opt.L, opt.seq_len, opt.video_batch_size,
    device = opt.device, train=False
)
ent, ent_map, _ = validate(opt, model, train_video, test_sampler, transform)

latent_vectors = latent_results[0] - np.mean(latent_results[0], axis = 0)
latent_vectors = latent_vectors / np.std(latent_vectors, axis = 0)
U, S, V = torch.pca_lowrank(torch.tensor(latent_vectors), q=opt.latent_size)
x = U[:, 0]
y = U[:, 1]
color_data = U[:, 2]
colors = (color_data - color_data.mean()) / color_data.std()
plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.savefig(f"{current_result_folder}/PCA_sccater(0, 1, 2).png")
plt.clf()

np.set_printoptions(precision=2, suppress=True)
print(S)

#
# Visualizing results
#

# animation
ent_map_normalized = (ent_map - ent_map.min()) / (ent_map.max() - ent_map.min())
train_video_np = train_video.clone().squeeze(0).squeeze(1).cpu().numpy()

fig, ax = plt.subplots()
im = ax.imshow(train_video_np[0], cmap='gray', animated=True)
overlay = ax.imshow(ent_map_normalized[0], cmap='viridis', alpha=0.5, animated=True)

def update(frame):
    im.set_array(train_video_np[frame])
    overlay.set_array(ent_map_normalized[frame])
    return [im, overlay]

ani = FuncAnimation(fig, update, frames=len(ent_map), blit=True)
ani.save(f"{current_result_folder}/ent_map_animation.mp4", fps=10)
plt.clf()

# mean local EP density
mean_map = ent_map.mean(axis=0)
plt.imshow(mean_map, cmap='viridis')
plt.colorbar(label='Mean Local EP Density')
plt.title('Mean Local EP Density Map')
plt.savefig(f"{current_result_folder}/mean_map.png")
plt.clf()