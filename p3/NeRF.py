from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import imageio
import cv2
import random
import logging
from torch.optim.lr_scheduler import StepLR

# Minimal logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("nerf_debug.log"),
        logging.StreamHandler()
    ]
)

def load_colmap_data():
    json_path = "/app/p3/data/transforms_colmap.json"
    with open(json_path, "r") as f:
        meta = json.load(f)
    focal = meta.get("focal_length")
    print("focal:", focal)
    camera_angle_x = meta.get("camera_angle_x")
    print("camera_angle_x:", camera_angle_x)
    frames = meta["frames"]
    print("frames:", len(frames))
    imgs, poses = [], []
    for frame in frames:
        img_path = frame["file_path"]
        if not os.path.splitext(img_path)[1]:
            img_path += ".png"
            img_path = os.path.join("/app/p3/data/images", img_path)
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        imgs.append(img)
        poses.append(np.array(frame["transform_matrix"]))
    imgs = np.stack(imgs, axis=0)
    poses = np.stack(poses, axis=0)
    H, W = imgs.shape[1], imgs.shape[2]
    if focal is None:
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    return imgs, poses, [H, W, focal]

def get_rays(H, W, focal, c2w, device):
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    c2w = torch.from_numpy(c2w).to(device=device, dtype=dirs.dtype) if isinstance(c2w, np.ndarray) else c2w.to(device)
    ray_directions = torch.matmul(dirs.view(-1, 3), c2w[:3, :3].T).view(H, W, 3)
    ray_origins = c2w[:3, 3].expand(H, W, 3)
    return ray_origins, ray_directions

def sample_points_from_rays(ray_origins, ray_directions, near_point, far_point, num_samples):
    H, W, _ = ray_origins.shape
    t_vals = torch.linspace(0, 1, steps=num_samples, device=ray_origins.device).view(1, 1, num_samples).expand(H, W, num_samples)
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
    lower = torch.cat([t_vals[..., :1], mids], dim=-1)
    t_rand = torch.rand_like(t_vals)
    t_vals = lower + (upper - lower) * t_rand
    depth_values = near_point + (far_point - near_point) * t_vals
    sampled_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return sampled_points, depth_values

def positional_encoding(x, num_freq=10, include_input=True):
    out = [x] if include_input else []
    freq_bands = 2.0 ** torch.arange(num_freq, dtype=torch.float32, device=x.device)
    for freq in freq_bands:
        out.append(torch.sin(x * freq))
        out.append(torch.cos(x * freq))
    return torch.cat(out, dim=-1)

def volume_rendering(radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor) -> Tuple[torch.Tensor]:
    rgb = torch.sigmoid(radiance_field[..., :3])
    sigma = F.relu(radiance_field[..., 3])
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e3, neginf=0.0)
    sigma = torch.clamp(sigma, 0.0, 1e3)
    deltas = depth_values[..., 1:] - depth_values[..., :-1]
    delta_last = 1e10 * torch.ones_like(deltas[..., :1])
    deltas = torch.cat([deltas, delta_last], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    return rgb_map,

class TinyNeRF(nn.Module):
    def __init__(self, pos_dim, fc_dim=128):
        super().__init__()
        self.nerf = nn.Sequential(
            nn.Linear(pos_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 4)
        )
    def forward(self, x):
        return self.nerf(x)

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def nerf_step_forward(height, width, focal_length, trans_matrix,
                      near_point, far_point, num_depth_samples_per_ray,
                      get_minibatches_function, model, device):
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix, device)
    sampled_points, depth_values = sample_points_from_rays(ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray)
    points_flat = sampled_points.view(-1, 3)
    positional_encoded_points = positional_encoding(points_flat, num_freq=10, include_input=True)
    batches = get_minibatches_function(positional_encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
        out = model(batch)
        predictions.append(out)
    radiance_field = torch.cat(predictions, dim=0).reshape(height, width, num_depth_samples_per_ray, 4)
    rgb_predicted = volume_rendering(radiance_field, ray_origins, depth_values)[0]
    return rgb_predicted

def plot_loss_curve(losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def render_and_save(rgb_pred, out_path):
    if rgb_pred is None:
        return
    rgb_pred = rgb_pred.detach().cpu()
    if torch.isnan(rgb_pred).any():
        return
    rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0).numpy()
    if rgb_pred.shape[-1] != 3:
        return
    plt.imsave(out_path, rgb_pred)

def train(images, poses, hwf, near_point, far_point, num_depth_samples_per_ray, num_iters, model, DEVICE="cuda"):
  H, W, focal_length = hwf
  H, W = int(H), int(W)
  lr = 5e-5
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  torch.manual_seed(9458)
  np.random.seed(9437)
  device = torch.device(DEVICE)
  model.to(device)
  losses = []
  for i in tqdm(range(num_iters)):
    train_idx = np.random.randint(images.shape[0])
    train_img_rgb = torch.from_numpy(images[train_idx, ..., :3]).to(device)
    
    train_pose = poses[train_idx]
    rgb_pred = nerf_step_forward(H, W, focal_length, train_pose, near_point, far_point, num_depth_samples_per_ray, get_minibatches, model, device)
    if rgb_pred is None or torch.isnan(rgb_pred).any():
      continue
    loss = F.mse_loss(rgb_pred, train_img_rgb)
    loss.backward()
    
    

    
    if i == 110:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 3e-5
    
    if i == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5

    if i % 100 == 0:
        plot_loss_curve(losses, "/my_vol/train_loss_curve.png")
    optimizer.step()
    optimizer.zero_grad()
    
    losses.append(loss.item())
    if i % 10 == 0:
      logging.info(f"[Iter {i}] Loss: {loss.item():.6f}")
    if i % 100 == 0 or i == num_iters - 1:
      model.eval()
      with torch.no_grad():
        render_and_save(rgb_pred, f"/my_vol/view_{i:04d}.png")
      model.train()
  plot_loss_curve(losses, "/my_vol/train_loss_curve.png")
  return model

def inference(model, poses, hwf, near_point, far_point, num_depth_samples_per_ray, DEVICE="cuda"):
    H, W, focal_length = map(int, hwf)
    device = torch.device(DEVICE)
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(random.sample(range(len(poses)), min(5, len(poses)))):
            test_pose = poses[idx]
            rgb_pred = nerf_step_forward(H, W, focal_length, test_pose, near_point, far_point, num_depth_samples_per_ray, get_minibatches, model, device)
            render_and_save(rgb_pred, f"/my_vol/test_view_{i}.png")

def visualize_nerf_predictions(model, images, poses, hwf, near, far, num_samples, device="cuda"):
    import matplotlib.pyplot as plt

    H, W, focal = map(int, hwf)
    device = torch.device(device)
    model.eval()
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    indices = random.sample(range(len(images)), 5)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            gt_img = images[idx]
            pose = poses[idx]
            gt_img_tensor = torch.from_numpy(gt_img[..., :3]).to(device)

            pred_img = nerf_step_forward(H, W, focal, pose, near, far, num_samples, get_minibatches, model, device)
            pred_img = pred_img.detach().cpu().numpy()
            pred_img = np.clip(pred_img, 0.0, 1.0)

            axes[i, 0].imshow(gt_img)
            axes[i, 0].set_title(f"Ground Truth #{idx}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f"NeRF Output #{idx}")
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/my_vol/nerf_predictions_comparison.png")
    plt.show()


if __name__ == "__main__":
    imgs, poses, hwf = load_colmap_data()
    pos_dim = 3 + 2 * 10 * 3
    model = TinyNeRF(pos_dim=pos_dim, fc_dim=128)
    train(imgs, poses, hwf, near_point=2.0, far_point=6.0, num_depth_samples_per_ray=64, num_iters=4000, model=model, DEVICE="cuda" if torch.cuda.is_available() else "cpu")
    inference(model, poses, hwf, near_point=2.0, far_point=6.0, num_depth_samples_per_ray=64, DEVICE="cuda" if torch.cuda.is_available() else "cpu")
