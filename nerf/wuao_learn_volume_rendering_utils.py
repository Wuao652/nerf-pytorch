import torch
import numpy as np
from nerf.volume_rendering_utils import volume_render_radiance_field
from nerf.nerf_helpers import cumprod_exclusive
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("hello world from wuao_learn!")
radiance_field = torch.tensor(np.load('../npy/radiance_field.npy')).to(device)
rd = torch.tensor(np.load('../npy/rd.npy')).to(device)
z_vals = torch.tensor(np.load('../npy/z_vals.npy')).to(device)

# (rf_rgb_coarse,
#  rf_disp_coarse,
#  rf_acc_coarse,
#  rf_weights,
#  rf_depth_coarse
# ) = volume_render_radiance_field(
#     radiance_field,
#     z_vals,
#     rd,
#     # radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
#     # white_background=getattr(options.nerf, mode).white_background,
#     radiance_field_noise_std=0.2,
#     white_background=True
# )

noise_std = 0.2
one_e_10 = torch.tensor([1e10], dtype=rd.dtype, device=rd.device)
# [1024, 63]
delta = z_vals[..., 1:] - z_vals[..., :-1]
dists = torch.cat((delta, one_e_10.expand(z_vals[:, :1].shape)), dim=-1)
norm_rd = rd.norm(p=2, dim=-1).reshape((-1, 1))
dists = dists * norm_rd

rgb = torch.sigmoid(radiance_field[..., :3])
noise = 0.0
if noise_std > 0.0:
    noise = noise_std * torch.randn(radiance_field[..., 3].shape,
                                   dtype=radiance_field.dtype,
                                   device=radiance_field.device)
sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
alpha = 1.0 - torch.exp(-sigma_a * dists)
in_T = 1.0 - alpha + 1e-10
rf_T = cumprod_exclusive(1.0 - alpha + 1e-10)
