import matplotlib.pyplot as plt
import torch
import os
import pandas as pd

def plot_performance(train_losses, val_losses, train_accs, val_accs, opt_name, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    # Convert tensors to numpy if needed
    train_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accs]
    val_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]
    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'{opt_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title(f'{opt_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Saved plot to {save_path}')
    plt.close()

def save_to_excel(data, excel_file):
    df = pd.DataFrame(data)
    
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_excel(excel_file, index=False)
    print(f'Saved training times and validation accuracy to {excel_file}')