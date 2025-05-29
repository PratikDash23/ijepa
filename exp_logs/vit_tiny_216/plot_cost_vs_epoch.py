import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'vit_tiny_216_r0.csv')

# Read the CSV, skipping duplicate header rows and bad lines
df = pd.read_csv(csv_path, on_bad_lines='skip')
df = df[df['epoch'] != 'epoch']
df = df[df['epoch'].notnull() & df['loss'].notnull()]
df['epoch'] = df['epoch'].astype(int)
df['loss'] = df['loss'].astype(float)

# Compute average loss per epoch
avg_loss = df.groupby('epoch')['loss'].mean()

# Plot and save the figure
plt.figure(figsize=(8, 5))
plt.plot(avg_loss.index, avg_loss.values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Cost (Loss) per Epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'cost_vs_epoch.png'))  # Save in the same directory as script and csv
# Do not show the plot

# display the number of parameters in the model
# whose checkpoint is in the same directory
checkpoint_path = os.path.join(script_dir, 'vit_tiny_216-latest.pth.tar')
if os.path.exists(checkpoint_path):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_params = sum(p.numel() for p in checkpoint['encoder'].values())
    predictor_params = sum(p.numel() for p in checkpoint['predictor'].values())
    print(f'Number of parameters in encoder: {encoder_params}')
    print(f'Number of parameters in predictor: {predictor_params}')