import torch

from .nerf_helpers import cumprod_exclusive


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
):
    """
    transform the radiance field to semantically meaningful values.
    get the color, depth, opacity and disparity
    Args:
        radiance_field: [num_rays, N_samples, 4] the prediction of the neural network. e.g.(1024, 64, 4)
        depth_values: [num_rays, N_samples along ray]. range in (near, far). e.g.(1024, 64)
        ray_directions: [num_rays, 3]. direction of each ray e.g.(1024, 3)
        radiance_field_noise_std: float
        white_background: bool. If true, assume a white background.

    Returns:
        rgb_map: [num_rays, 3]. estimated RGB color of a ray.
        disp_map: [num_rays] estimated disparity of a ray. 1 / depth.
        acc_map: [num_rays] estimated opacity.
        weights: [num_rays, N_samples]
        depth_map: [num_rays] estimated depth of a ray.
    """
    # TODO: just for testing the algorithm
    #       delete after learning the rendering
    # seed = 42
    # import numpy as np
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    # compute the delta_i for each query points.
    # the delta for the last points of each ray are assigned to 1e10.
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    # [num_rays, N_samples]
    # dists times the norm of the ray directions
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    # [num_rays, N_samples, 3] stores the color of each query point
    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
    # [num_rays, N_samples] stores the volume density of each query point, e.g.[1024, 64]
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)

    # cumprod_exclusive computes the T for each query point.
    # cumprod_exclusive(torch.exp(-sigma_a * dists))
    # weights [num_rays, N_samples]
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
