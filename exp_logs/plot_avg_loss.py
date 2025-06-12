import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_216')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep4')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep10')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep20')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_dataset_new_ep100')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_small_dataset_new_ep20')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_with_val_ep5')
# log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_with_val_ep14')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_tiny_with_val_ep100')

csv_files = glob.glob(os.path.join(log_dir, '*.csv'))

all_data = []
for f in csv_files:
    df = pd.read_csv(f, on_bad_lines='skip')
    if 'loss' not in df.columns or 'epoch' not in df.columns:
        continue  # skip files that are not training logs
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

# Find the validation loss file
val_loss_file = None
for f in csv_files:
    if f.endswith('_val_loss.csv'):
        val_loss_file = f
        break

if val_loss_file is not None:
    val_df = pd.read_csv(val_loss_file, on_bad_lines='skip')
    # Remove duplicate header rows
    val_df = val_df[val_df['epoch'] != 'epoch']
    val_df = val_df[val_df['epoch'].notnull() & val_df['val_loss'].notnull()]
    val_df['epoch'] = pd.to_numeric(val_df['epoch'], errors='coerce')
    val_df['val_loss'] = pd.to_numeric(val_df['val_loss'], errors='coerce')
    val_df = val_df.dropna(subset=['epoch', 'val_loss'])
    val_loss = val_df.set_index('epoch')['val_loss']
else:
    val_loss = None

# Find the test loss file and print its value (do not plot)
test_loss_file = None
for f in csv_files:
    if f.endswith('_test_loss.csv'):
        test_loss_file = f
        break

test_loss_value = None
if test_loss_file is not None:
    test_df = pd.read_csv(test_loss_file, on_bad_lines='skip')
    # Remove duplicate header rows
    test_df = test_df[test_df.columns[0]]  # get the first column (should be 'test_loss')
    # Remove header rows
    test_df = test_df[test_df != 'test_loss']
    # Convert to numeric and drop NaN
    test_df = pd.to_numeric(test_df, errors='coerce').dropna()
    if not test_df.empty:
        test_loss_value = test_df.iloc[-1]  # last valid test loss
        print(f"Test loss: {test_loss_value:.5f}")

# Determine the max value for y-axis scaling
max_train = avg_loss.max() if not avg_loss.empty else 0
max_val = val_loss.max() if (val_loss is not None and not val_loss.empty) else 0
max_y = max(max_train, max_val)

# Plot and save the figure
plt.figure(figsize=(8, 5))
ax = plt.gca()
avg_loss.plot(marker='o', ax=ax, label='Train Loss', color='tab:blue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Loss', color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
plt.title('Cost (Loss) per Epoch')
plt.grid(True)
ax.set_ylim(0, max_y * 1.05)

# Plot validation loss on the right y-axis if available
if val_loss is not None and not val_loss.empty:
    ax2 = ax.twinx()
    val_loss.plot(marker='s', ax=ax2, label='Val Loss', color='tab:red')
    ax2.set_ylabel('Validation Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, max_y * 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
else:
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'loss_vs_epoch.png'))  
plt.close()
