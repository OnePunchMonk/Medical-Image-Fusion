import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import MultiModalFusionModel
from dataset import MultiModalDataset

def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MultiModalFusionModel(in_channels=1, base_filters=args.base_filters)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    # Create dataset and dataloader
    test_dataset = MultiModalDataset(
        data_dir=args.data_dir,
        split='test',
        transform=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation metrics
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    
    # Evaluation loop
    model.eval()
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating") as pbar:
            for batch_idx, (ct_images, mri_images, target_images) in enumerate(pbar):
                # Move data to device
                ct_images = ct_images.to(device)
                mri_images = mri_images.to(device)
                target_images = target_images.to(device)
                
                # Forward pass
                outputs = model(ct_images, mri_images)
                
                # Calculate losses
                batch_l1_loss = l1_loss(outputs, target_images).item()
                batch_l2_loss = l2_loss(outputs, target_images).item()
                
                # Calculate PSNR and SSIM
                for i in range(outputs.size(0)):
                    # Convert to numpy arrays for PSNR and SSIM calculation
                    output_np = outputs[i, 0].cpu().numpy()
                    target_np = target_images[i, 0].cpu().numpy()
                    
                    # Normalize to [0, 1] for PSNR and SSIM calculation
                    output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min() + 1e-8)
                    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
                    
                    # Calculate PSNR and SSIM
                    batch_psnr = psnr(target_np, output_np, data_range=1.0)
                    batch_ssim = ssim(target_np, output_np, data_range=1.0)
                    
                    total_psnr += batch_psnr
                    total_ssim += batch_ssim
                
                # Update total losses
                total_l1_loss += batch_l1_loss
                total_l2_loss += batch_l2_loss
                
                # Update progress bar
                pbar.set_postfix(L1=batch_l1_loss, L2=batch_l2_loss)
                
                # Save sample images
                if batch_idx < args.num_samples_to_save:
                    for i in range(min(outputs.size(0), 4)):  # Save up to 4 images per batch
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        
                        # Denormalize images for visualization
                        ct_img = ct_images[i, 0].cpu().numpy()
                        mri_img = mri_images[i, 0].cpu().numpy()
                        target_img = target_images[i, 0].cpu().numpy()
                        output_img = outputs[i, 0].cpu().numpy()
                        
                        # Plot images
                        axes[0].imshow(ct_img, cmap='gray')
                        axes[0].set_title('CT Image')
                        axes[0].axis('off')
                        
                        axes[1].imshow(mri_img, cmap='gray')
                        axes[1].set_title('MRI Image')
                        axes[1].axis('off')
                        
                        axes[2].imshow(target_img, cmap='gray')
                        axes[2].set_title('Target Image')
                        axes[2].axis('off')
                        
                        axes[3].imshow(output_img, cmap='gray')
                        axes[3].set_title('Fused Image (Output)')
                        axes[3].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, f'sample_{batch_idx}_{i}.png'))
                        plt.close()
    
    # Calculate average metrics
    num_samples = len(test_dataset)
    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_l2_loss = total_l2_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Print results
    print(f"Evaluation Results:")
    print(f"L1 Loss: {avg_l1_loss:.4f}")
    print(f"L2 Loss: {avg_l2_loss:.4f}")
    print(f"PSNR: {avg_psnr:.4f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Evaluation Results:\n")
        f.write(f"L1 Loss: {avg_l1_loss:.4f}\n")
        f.write(f"L2 Loss: {avg_l2_loss:.4f}\n")
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi-Modal Fusion Model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    
    # Model parameters
    parser.add_argument('--base_filters', type=int, default=64, help='Number of base filters in the model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Directory to save outputs')
    parser.add_argument('--num_samples_to_save', type=int, default=5, help='Number of sample batches to save')
    
    args = parser.parse_args()
    evaluate(args) 