import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from skimage import io

class MultiModalDataset(Dataset):
    """Dataset class for multi-modal medical image fusion."""
    
    def __init__(self, data_dir, split='train', transform=True):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the dataset.
            split (str): Dataset split ('train', 'val', or 'test').
            transform (bool): Whether to apply data augmentation.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Define paths for CT, MRI, and target images
        self.split_dir = os.path.join(data_dir, split)
        self.ct_dir = os.path.join(self.split_dir, 'ct')
        self.mri_dir = os.path.join(self.split_dir, 'mri')
        self.target_dir = os.path.join(self.split_dir, 'target')
        
        # Get list of image filenames
        self.filenames = [f for f in os.listdir(self.ct_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        # Define transformations
        self.transform_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
        
        # Normalization transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (ct_image, mri_image, target_image)
        """
        # Get image filenames
        filename = self.filenames[idx]
        
        # Load images
        ct_path = os.path.join(self.ct_dir, filename)
        mri_path = os.path.join(self.mri_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        # Read images
        ct_image = Image.open(ct_path).convert('L')  # Convert to grayscale
        mri_image = Image.open(mri_path).convert('L')  # Convert to grayscale
        target_image = Image.open(target_path).convert('L')  # Convert to grayscale
        
        # Apply transformations
        if self.transform and self.split == 'train':
            # Apply same random transformations to all images
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            ct_image = self.transform_pipeline(ct_image)
            
            torch.manual_seed(seed)
            mri_image = self.transform_pipeline(mri_image)
            
            torch.manual_seed(seed)
            target_image = self.transform_pipeline(target_image)
        
        # Convert to tensor and normalize
        ct_tensor = self.normalize(ct_image)
        mri_tensor = self.normalize(mri_image)
        target_tensor = self.normalize(target_image)
        
        return ct_tensor, mri_tensor, target_tensor

    @staticmethod
    def create_dummy_dataset(output_dir, num_samples=10, image_size=256):
        """
        Create a dummy dataset for testing purposes.
        
        Args:
            output_dir (str): Directory to save the dummy dataset.
            num_samples (int): Number of samples to generate.
            image_size (int): Size of the generated images.
        """
        # Create directory structure
        os.makedirs(os.path.join(output_dir, 'train', 'ct'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', 'mri'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', 'target'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'ct'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'mri'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'target'), exist_ok=True)
        
        # Generate dummy images
        for split in ['train', 'val']:
            for i in range(num_samples):
                # Create random images
                ct_image = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
                mri_image = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
                
                # Create target image (simple average fusion for dummy data)
                target_image = (ct_image.astype(np.float32) + mri_image.astype(np.float32)) / 2
                target_image = target_image.astype(np.uint8)
                
                # Save images
                io.imsave(os.path.join(output_dir, split, 'ct', f'image_{i:03d}.png'), ct_image)
                io.imsave(os.path.join(output_dir, split, 'mri', f'image_{i:03d}.png'), mri_image)
                io.imsave(os.path.join(output_dir, split, 'target', f'image_{i:03d}.png'), target_image)
        
        print(f"Created dummy dataset with {num_samples} samples in {output_dir}") 