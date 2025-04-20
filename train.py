import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import MultiModalFusionModel, get_loss_function
from dataset import MultiModalDataset

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MultiModalFusionModel(in_channels=1, base_filters=args.base_filters)
    model = model.to(device)
    
    # Create dataset and dataloader
    train_dataset = MultiModalDataset(
        data_dir=args.data_dir,
        split='train',
        transform=True
    )
    
    val_dataset = MultiModalDataset(
        data_dir=args.data_dir,
        split='val',
        transform=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Define loss function and optimizer
    criterion = get_loss_function(args.loss_type)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]") as pbar:
            for batch_idx, (ct_images, mri_images, target_images) in enumerate(pbar):
                # Move data to device
                ct_images = ct_images.to(device)
                mri_images = mri_images.to(device)
                target_images = target_images.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(ct_images, mri_images)
                
                # Calculate loss
                loss = criterion(outputs, target_images)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]") as pbar:
                for batch_idx, (ct_images, mri_images, target_images) in enumerate(pbar):
                    # Move data to device
                    ct_images = ct_images.to(device)
                    mri_images = mri_images.to(device)
                    target_images = target_images.to(device)
                    
                    # Forward pass
                    outputs = model(ct_images, mri_images)
                    
                    # Calculate loss
                    loss = criterion(outputs, target_images)
                    
                    # Update progress bar
                    val_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()
    
    print("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Modal Fusion Model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    
    # Model parameters
    parser.add_argument('--base_filters', type=int, default=64, help='Number of base filters in the model')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'mse'], help='Loss function type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    
    args = parser.parse_args()
    train(args) 