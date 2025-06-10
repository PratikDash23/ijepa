import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.append("c:/Users/dash/Documents/learning_ai/ijepa")
from src.helper import init_model
from src.masks.multiblock import MaskCollator

class SketchFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.rglob("*.png")) + list(self.root_dir.rglob("*.jpg")) + list(self.root_dir.rglob("*.jpeg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def prepare_dataloader(data_root, batch_size=32, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SketchFolderDataset(data_root, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

class CNNDecoder(nn.Module):
    def __init__(self, input_dim=384, patch_size=14):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128 * 3 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),  # 3x3 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),   # 7x7 -> 15x15
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),                        # 15x15 -> 15x15 RGB
            nn.Sigmoid()
        )
        self.patch_size = patch_size

    def forward(self, x):
        x = self.fc(x)  # [B, 128 * 3 * 3]
        x = x.view(-1, 128, 3, 3)
        x = self.decoder(x)
        return nn.functional.interpolate(x, size=(self.patch_size, self.patch_size), mode='bilinear')

class ShallowDecoder(nn.Module):
    def __init__(self, input_dim=384, patch_size=14):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * patch_size * patch_size),
            nn.Sigmoid()
        )
        self.patch_size = patch_size

    def forward(self, x):
        out = self.decoder(x)
        return out.view(-1, 3, self.patch_size, self.patch_size)

def extract_patches(img_tensor, indices, patch_size, grid_w):
    indices = indices.flatten()  # Ensure indices is 1D
    patches = []
    for idx in indices:
        idx = int(idx)  # Now idx is a Python int
        row = idx // grid_w
        col = idx % grid_w
        y, x = row * patch_size, col * patch_size
        patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    return torch.cat(patches, dim=0)

def show_comparison(input_tensor, recon_canvas, context_masks, target_masks, patch_size, grid_size, save_path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    input_img = input_tensor * std + mean
    recon_img = recon_canvas * std + mean
    input_img = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_img = recon_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    context_indices = context_masks[0].squeeze(0)
    target_indices = target_masks[0].squeeze(0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img, title in zip(axs, [input_img, recon_img], ["Original", "Reconstructed"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        for idx in context_indices:
            row, col = idx // grid_size, idx % grid_size
            ax.add_patch(plt.Rectangle((col * patch_size, row * patch_size), patch_size, patch_size,
                                       color='blue', alpha=0.3))
        for idx in target_indices:
            row, col = idx // grid_size, idx % grid_size
            ax.add_patch(plt.Rectangle((col * patch_size, row * patch_size), patch_size, patch_size,
                                       edgecolor='red', facecolor='none', linewidth=2))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def train_decoder(model, predictor, decoder, dataloader, mask_collator, device, patch_size, grid_w, num_batches=5):
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    decoder.train()

    print("Starting efficient decoder training")

    for batch_count, batch_imgs in enumerate(dataloader):
        if batch_count >= num_batches:
            break

        batch_imgs = batch_imgs.to(device)
        batch_img_list = [img.unsqueeze(0) for img in batch_imgs]
        
        # Batch masking
        try:
            _, context_masks, target_masks = mask_collator(tuple(batch_img_list))
            print(f"context_masks shape: {context_masks.shape}, target_masks shape: {target_masks.shape}")
        except Exception as e:
            print(f"Skipping batch {batch_count+1} due to mask error: {e}")
            continue

        with torch.no_grad():
            z = model(batch_imgs, context_masks)
            p = predictor(z, context_masks, target_masks)

        loss_total = 0
        optimizer.zero_grad()

        for i in range(batch_imgs.size(0)):
            input_img = batch_imgs[i].unsqueeze(0)  # [1, 3, 224, 224]
            target_indices = target_masks[i][0].squeeze(0)
            pred_tokens = p[i]

            gt_patches = extract_patches(input_img, target_indices, patch_size, grid_w).to(device)
            decoded = decoder(pred_tokens)

            loss = nn.functional.mse_loss(decoded, gt_patches)
            loss_total += loss

        avg_loss = loss_total / batch_imgs.size(0)
        avg_loss.backward()
        optimizer.step()

        print(f"Batch {batch_count+1}/{num_batches} — Avg Loss: {avg_loss.item():.4f}")

    print("🏁 Decoder training complete.")

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

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    img = Image.open("c:/Users/dash/Documents/learning_ai/ijepa/images/my_image.jpg").convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Wrap input_tensor in a list to simulate a batch
    # Unpack the three values returned by MaskCollator
    _, context_masks, target_masks = mask_collator([input_tensor])
    target_indices = target_masks[0].squeeze(0) # get the target indices
    # prepare an image visualizing the context and target masks
    visualize_masks(input_tensor,
                    patch_size=cfg["mask"]["patch_size"],
                    context_masks=context_masks[0].squeeze(0),
                    target_masks=target_masks[0].squeeze(0),
                    save_path="c:/Users/dash/Documents/learning_ai/ijepa/images/mask_visualization.png")

    # get the IJEPA prediction 
    with torch.no_grad():
        z = model(input_tensor, context_masks)
        p = predictor(z, context_masks, target_masks)

    # define patch size and grid width
    patch_size = cfg["mask"]["patch_size"]
    grid_w = cfg["data"]["crop_size"] // patch_size

    # load the ImageNet-Sketch dataset
    data_root = "c:/Users/dash/Documents/learning_ai/ijepa/datasets/ImageNet-Sketch/"
    dataloader = prepare_dataloader(data_root)

    decoder1 = ShallowDecoder(input_dim=p.shape[-1], patch_size=patch_size).to(device)
    decoder2 = CNNDecoder(input_dim=p.shape[-1], patch_size=patch_size).to(device)
    gt_patches = extract_patches(input_tensor, target_indices, patch_size, grid_w).to(device)

    # # train the decoder
    # train_decoder(model, predictor, decoder, dataloader, mask_collator, device, patch_size=patch_size, grid_w=grid_w, num_batches=5)

    optimizer1 = torch.optim.Adam(decoder1.parameters(), lr=1e-3)
    decoder1.train()
    for epoch in range(300):
        optimizer1.zero_grad()
        decoded = decoder1(p.squeeze(0))
        loss = nn.functional.mse_loss(decoded, gt_patches)
        loss.backward()
        optimizer1.step()

    decoder1.eval()
    with torch.no_grad():
        recon_patches1 = decoder1(p.squeeze(0))

    canvas = input_tensor.clone()
    count = torch.zeros_like(input_tensor)
    for i, idx in enumerate(target_indices):
        row, col = idx // grid_w, idx % grid_w
        y, x = row * patch_size, col * patch_size
        canvas[:, :, y:y+patch_size, x:x+patch_size] = recon_patches1[i]

    save_path = "c:/Users/dash/Documents/learning_ai/ijepa/images/compare_targets_nn.png"
    show_comparison(input_tensor, canvas, context_masks, target_masks, patch_size, grid_w, save_path)

    # Now let's do the same for the CNN decoder
    optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
    decoder2.train()
    for epoch in range(300):
        optimizer2.zero_grad()
        decoded = decoder2(p.squeeze(0))
        loss = nn.functional.mse_loss(decoded, gt_patches)
        loss.backward()
        optimizer2.step()
    decoder2.eval()
    with torch.no_grad():
        recon_patches2 = decoder2(p.squeeze(0))
    canvas = input_tensor.clone()
    count = torch.zeros_like(input_tensor)
    for i, idx in enumerate(target_indices):
        row, col = idx // grid_w, idx % grid_w
        y, x = row * patch_size, col * patch_size
        canvas[:, :, y:y+patch_size, x:x+patch_size] = recon_patches2[i]
    save_path = "c:/Users/dash/Documents/learning_ai/ijepa/images/compare_targets_cnn.png"
    show_comparison(input_tensor, canvas, context_masks, target_masks, patch_size, grid_w, save_path)


if __name__ == "__main__":
    main()
