import os
import argparse
from dataset import MultiModalDataset

def main(args):
    """Create a dummy dataset for testing the multi-modal fusion model."""
    print(f"Creating dummy dataset with {args.num_samples} samples...")
    
    # Create dummy dataset
    MultiModalDataset.create_dummy_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size
    )
    
    # Create test split as well
    os.makedirs(os.path.join(args.output_dir, 'test', 'ct'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test', 'mri'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test', 'target'), exist_ok=True)
    
    # Create test samples
    MultiModalDataset.create_dummy_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples // 5,  # Fewer test samples
        image_size=args.image_size
    )
    
    print(f"Dummy dataset created successfully at {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Dummy Dataset for Multi-Modal Fusion')
    
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save the dummy dataset')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the generated images')
    
    args = parser.parse_args()
    main(args) 