import os
import shutil
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

def create_directory_structure():
    """Create the required directory structure for the dataset."""
    os.makedirs('data/train/ct', exist_ok=True)
    os.makedirs('data/train/mri', exist_ok=True)
    os.makedirs('data/train/target', exist_ok=True)
    
    os.makedirs('data/val/ct', exist_ok=True)
    os.makedirs('data/val/mri', exist_ok=True)
    os.makedirs('data/val/target', exist_ok=True)
    
    os.makedirs('data/test/ct', exist_ok=True)
    os.makedirs('data/test/mri', exist_ok=True)
    os.makedirs('data/test/target', exist_ok=True)
    
    print("Created directory structure.")

def generate_target_image(ct_path, mri_path, target_path):
    """Generate a target image by averaging CT and MRI images."""
    # Open images
    ct_img = Image.open(ct_path).convert('L')  # Convert to grayscale
    mri_img = Image.open(mri_path).convert('L')  # Convert to grayscale
    
    # Convert to numpy arrays
    ct_array = np.array(ct_img, dtype=np.float32)
    mri_array = np.array(mri_img, dtype=np.float32)
    
    # Simple fusion by averaging
    target_array = (ct_array + mri_array) / 2
    
    # Convert back to uint8 for saving
    target_array = target_array.astype(np.uint8)
    
    # Create image from array and save
    target_img = Image.fromarray(target_array)
    target_img.save(target_path)

def split_and_organize_dataset(ct_dir, mri_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the dataset into train, validation, and test sets and organize files."""
    # Get list of CT images
    ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.png')]
    
    # Check if corresponding MRI images exist
    valid_files = []
    for ct_file in ct_files:
        mri_path = os.path.join(mri_dir, ct_file)
        if os.path.exists(mri_path):
            valid_files.append(ct_file)
    
    # Shuffle files for random split
    random.shuffle(valid_files)
    
    # Calculate split indices
    total_files = len(valid_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
    
    print(f"Splitting dataset: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images")
    
    # Process training files
    print("Processing training files...")
    for file in tqdm(train_files):
        ct_path = os.path.join(ct_dir, file)
        mri_path = os.path.join(mri_dir, file)
        
        # Copy CT and MRI images
        shutil.copy(ct_path, os.path.join('data/train/ct', file))
        shutil.copy(mri_path, os.path.join('data/train/mri', file))
        
        # Generate and save target image
        generate_target_image(
            ct_path, 
            mri_path, 
            os.path.join('data/train/target', file)
        )
    
    # Process validation files
    print("Processing validation files...")
    for file in tqdm(val_files):
        ct_path = os.path.join(ct_dir, file)
        mri_path = os.path.join(mri_dir, file)
        
        # Copy CT and MRI images
        shutil.copy(ct_path, os.path.join('data/val/ct', file))
        shutil.copy(mri_path, os.path.join('data/val/mri', file))
        
        # Generate and save target image
        generate_target_image(
            ct_path, 
            mri_path, 
            os.path.join('data/val/target', file)
        )
    
    # Process test files
    print("Processing test files...")
    for file in tqdm(test_files):
        ct_path = os.path.join(ct_dir, file)
        mri_path = os.path.join(mri_dir, file)
        
        # Copy CT and MRI images
        shutil.copy(ct_path, os.path.join('data/test/ct', file))
        shutil.copy(mri_path, os.path.join('data/test/mri', file))
        
        # Generate and save target image
        generate_target_image(
            ct_path, 
            mri_path, 
            os.path.join('data/test/target', file)
        )
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    # Set paths to your Harvard dataset
    ct_dir = "Havard-Medical-Image-Fusion-Datasets-main/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/CT"
    mri_dir = "Havard-Medical-Image-Fusion-Datasets-main/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/MRI"
    
    # Create directory structure
    create_directory_structure()
    
    # Split and organize dataset
    split_and_organize_dataset(ct_dir, mri_dir)
    
    print("\nDataset is ready for training!")
    print("You can now run the training script with:")
    print("python train.py --data_dir ./data --batch_size 8 --num_epochs 100 --output_dir ./output") 