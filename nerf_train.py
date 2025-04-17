import modal

app = modal.App("tiny-nerf-training")

vol = modal.Volume.from_name("tiny-nerf_output")

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install([
        "torch", 
        "numpy", 
        "matplotlib", 
        "tqdm", 
        "imageio", 
        "opencv-python-headless"
    ])
    .add_local_dir("p3", remote_path="/app/p3")
)

@app.function(image=image, gpu="A100-80GB", memory=8000, volumes={"/my_vol": vol}, timeout=60000)
def modal_train():
    import os
    import sys
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    torch.cuda.empty_cache()

    sys.path.insert(0, "/app")
    from p3.NeRF import load_colmap_data, TinyNeRF, get_minibatches, nerf_step_forward, train, inference, visualize_nerf_predictions

    
    imgs, poses, hwf = load_colmap_data()
    pos_dim = 3 + 2 * 10 * 3
    model = TinyNeRF(pos_dim=pos_dim, fc_dim=64)

   
    model.load_state_dict(torch.load("/my_vol/model.pth"))

    # Move to GPU if available (optional)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model = train(imgs, poses, hwf, 2.0, 6.0, 64, 2000, model, DEVICE="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    visualize_nerf_predictions(model, imgs, poses, hwf, near=2.0, far=6.0, num_samples=64, device="cuda")
    

    inference(model, poses, hwf, 2.0, 6.0, 64, DEVICE="cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    modal_train.call()
