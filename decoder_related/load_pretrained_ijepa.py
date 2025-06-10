import sys
import yaml
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Ensure the src directory is in the Python path
sys.path.append("c:/Users/dash/Documents/learning_ai/ijepa")

from src.helper import init_model
from src.masks.multiblock import MaskCollator

import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(input_tensor, patch_size, context_masks, target_masks, save_path):
    # Denormalize the input tensor for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denormalized_tensor = input_tensor * std + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)  # Clip to [0, 1]

    # get the grid size from the patch size and input image size
    grid_size = denormalized_tensor.shape[2] // patch_size

    # prepare a figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(denormalized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax.set_title("Context and target patches", fontsize=20)
    ax.axis("off")

    # in the second image, fill the context patches with blue and target patches with red
    # with alpha=0.5
    for mask in context_masks:
        ax.add_patch(plt.Rectangle((patch_size * (mask % grid_size), patch_size * (mask // grid_size)),
                                     patch_size, patch_size,
                                     edgecolor='blue', facecolor='blue', alpha=0.5))

    for mask in target_masks:
        ax.add_patch(plt.Rectangle((patch_size * (mask % grid_size), patch_size * (mask // grid_size)),
                                     patch_size, patch_size,
                                     edgecolor='red', facecolor='red', alpha=0.5))

    # save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=60.7)
    plt.close(fig)

def main():
    cfg_path = "c:/Users/dash/Documents/learning_ai/ijepa/configs/in1k_vith14_ep300.yaml"
    weights_path = "c:/Users/dash/Documents/learning_ai/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar"
    cfg = yaml.safe_load(open(cfg_path))

    device = torch.device('cpu')
    model, predictor = init_model(device=device,
                                  patch_size=cfg["mask"]["patch_size"],
                                  model_name=cfg["meta"]["model_name"],
                                  crop_size=cfg["data"]["crop_size"],
                                  pred_depth=cfg["meta"]["pred_depth"],
                                  pred_emb_dim=cfg["meta"]["pred_emb_dim"])
    model.eval()
    predictor.eval()

    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["encoder"], strict=False)
    predictor.load_state_dict(ckpt["predictor"], strict=False)

    img = Image.open("c:/Users/dash/Documents/learning_ai/ijepa/images/my_image.jpg").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # create a MaskCollator instance with the desired parameters
    mask_collator = MaskCollator(
        input_size=(224, 224),
        patch_size=cfg["mask"]["patch_size"],
        enc_mask_scale=(0.85, 0.85),  # Ensure context patches cover at least 85% of the image
        pred_mask_scale=(0.05, 0.15),  # Target patches scale
        aspect_ratio=(0.3, 3.0),
        nenc=1,  # Number of context masks
        npred=1,  # Number of target masks
        min_keep=4,  # Minimum number of patches to keep
        allow_overlap=False  # Ensure no overlap between context and target patches
    )

    # Wrap input_tensor in a list to simulate a batch
    # Unpack the three values returned by MaskCollator
    _, context_masks, target_masks = mask_collator([input_tensor])
    # prepare an image visualizing the context and target masks
    visualize_masks(input_tensor,
                    patch_size=cfg["mask"]["patch_size"],
                    context_masks=context_masks[0].squeeze(0),
                    target_masks=target_masks[0].squeeze(0),
                    save_path="c:/Users/dash/Documents/learning_ai/ijepa/images/mask_visualization.png")

    with torch.no_grad():
        z = model(input_tensor, context_masks)
        p = predictor(z, context_masks, target_masks)  # Use target_masks instead of None

    

if __name__ == "__main__":
    main()
