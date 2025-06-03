import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_216')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep4')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep10')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep20')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep100')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_small_dataset_new_ep20')

csv_files = glob.glob(os.path.join(log_dir, '*.csv'))

all_data = []
for f in csv_files:
    df = pd.read_csv(f, on_bad_lines='skip')
    # Remove duplicate header rows
    df = df[df['epoch'] != 'epoch']
    # Drop rows with missing values in 'epoch' or 'loss'
    df = df[df['epoch'].notnull() & df['loss'].notnull()]
    # Convert to numeric
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    # Drop rows where conversion failed
    df = df.dropna(subset=['epoch', 'loss'])
    all_data.append(df)

if not all_data:
    raise RuntimeError("No CSV files found or all files are empty!")

data = pd.concat(all_data, ignore_index=True)

avg_loss = data.groupby('epoch')['loss'].mean()

print("Average loss per epoch:")
print(avg_loss)


# Plot and save the figure
plt.figure(figsize=(8, 5))
avg_loss.plot(marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Cost (Loss) per Epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'avg_loss_vs_epoch.png'))  
plt.close()
