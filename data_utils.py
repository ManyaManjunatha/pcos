"""
Data Preparation and Loading Utilities
Provides helper functions for organizing and loading acne detection datasets
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import json


class DatasetOrganizer:
    """
    Organize raw images into proper directory structure
    Expected input: flat directory with labeled images
    Output: organized directory structure for training
    """
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
    
    def organize_binary_dataset(self, skin_keywords: List[str], 
                               non_skin_keywords: List[str]):
        """
        Organize images for binary classification (skin vs non-skin)
        
        Args:
            skin_keywords: List of keywords in filenames indicating skin images
            non_skin_keywords: List of keywords indicating non-skin images
        """
        # Create directories
        skin_dir = self.target_dir / 'skin'
        non_skin_dir = self.target_dir / 'non_skin'
        skin_dir.mkdir(parents=True, exist_ok=True)
        non_skin_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images
        for img_path in self.source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filename = img_path.name.lower()
                
                # Check if it's skin or non-skin
                if any(keyword in filename for keyword in skin_keywords):
                    shutil.copy(img_path, skin_dir / img_path.name)
                elif any(keyword in filename for keyword in non_skin_keywords):
                    shutil.copy(img_path, non_skin_dir / img_path.name)
        
        print(f"Binary dataset organized:")
        print(f"  Skin images: {len(list(skin_dir.glob('*')))}")
        print(f"  Non-skin images: {len(list(non_skin_dir.glob('*')))}")
    
    def organize_seven_class_dataset(self, class_mapping: Dict[str, List[str]]):
        """
        Organize images for seven-class classification
        
        Args:
            class_mapping: Dictionary mapping class names to filename keywords
                          e.g., {'papule': ['papule', 'pap'], ...}
        """
        # Create directories
        for class_name in class_mapping.keys():
            class_dir = self.target_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images
        for img_path in self.source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filename = img_path.name.lower()
                
                # Find matching class
                for class_name, keywords in class_mapping.items():
                    if any(keyword in filename for keyword in keywords):
                        class_dir = self.target_dir / class_name
                        shutil.copy(img_path, class_dir / img_path.name)
                        break
        
        # Print statistics
        print(f"Seven-class dataset organized:")
        for class_name in class_mapping.keys():
            class_dir = self.target_dir / class_name
            count = len(list(class_dir.glob('*')))
            print(f"  {class_name}: {count}")


class ImagePreprocessor:
    """
    Preprocess images for training
    - Resize to target size
    - Convert to RGB
    - Optional: crop to specific regions
    """
    
    def __init__(self, target_size: Tuple[int, int] = (50, 50)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str, save_path: str = None) -> Image.Image:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save processed image
        
        Returns:
            Preprocessed PIL Image
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize(self.target_size, Image.LANCZOS)
        
        # Save if path provided
        if save_path:
            img.save(save_path)
        
        return img
    
    def preprocess_directory(self, input_dir: str, output_dir: str):
        """
        Preprocess all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_path in input_path.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                save_path = output_path / img_path.name
                self.preprocess_image(str(img_path), str(save_path))
                count += 1
        
        print(f"Preprocessed {count} images from {input_dir}")
    
    def crop_patches(self, image_path: str, patch_size: int = 50, 
                    stride: int = 25, output_dir: str = None) -> List[Image.Image]:
        """
        Extract patches from a large image using sliding window
        
        Args:
            image_path: Path to input image
            patch_size: Size of patches to extract
            stride: Stride for sliding window
            output_dir: Optional directory to save patches
        
        Returns:
            List of patch images
        """
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        patches = []
        patch_idx = 0
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)
                
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    patch_name = f"{Path(image_path).stem}_patch_{patch_idx:04d}.jpg"
                    patch.save(output_path / patch_name)
                    patch_idx += 1
        
        return patches


class AcneDataLoader:
    """
    Create DataLoaders for training, validation, and testing
    """
    
    def __init__(self, data_dir: str, transforms_dict: Dict, 
                 train_split: float = 0.8, val_split: float = 0.1,
                 random_seed: int = 1337):
        self.data_dir = Path(data_dir)
        self.transforms_dict = transforms_dict
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        self.random_seed = random_seed
    
    def create_binary_loaders(self, batch_size: int = 64, 
                             num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for binary classification
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Collect image paths and labels
        image_paths = []
        labels = []
        
        # Class 0: skin
        skin_dir = self.data_dir / 'skin'
        if skin_dir.exists():
            for img_path in skin_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(0)
        
        # Class 1: non-skin
        non_skin_dir = self.data_dir / 'non_skin'
        if non_skin_dir.exists():
            for img_path in non_skin_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(1)
        
        # Split data
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, 
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels,
            test_size=self.val_split / (self.train_split + self.val_split),
            random_state=self.random_seed,
            stratify=train_labels
        )
        
        # Create datasets
        from acne_detection_main import AcneDataset
        
        train_dataset = AcneDataset(
            train_paths, train_labels, 
            transform=self.transforms_dict['train'],
            is_train=True
        )
        
        val_dataset = AcneDataset(
            val_paths, val_labels,
            transform=self.transforms_dict['val'],
            is_train=False
        )
        
        test_dataset = AcneDataset(
            test_paths, test_labels,
            transform=self.transforms_dict['val'],
            is_train=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nBinary Classification DataLoaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def create_seven_class_loaders(self, class_names: List[str], 
                                   batch_size: int = 64,
                                   num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for seven-class classification
        
        Args:
            class_names: List of class names matching directory names
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Collect image paths and labels
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
        
        # Split data
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels,
            test_size=self.val_split / (self.train_split + self.val_split),
            random_state=self.random_seed,
            stratify=train_labels
        )
        
        # Create datasets
        from acne_detection_main import AcneDataset
        
        train_dataset = AcneDataset(
            train_paths, train_labels,
            transform=self.transforms_dict['train'],
            is_train=True
        )
        
        val_dataset = AcneDataset(
            val_paths, val_labels,
            transform=self.transforms_dict['val'],
            is_train=False
        )
        
        test_dataset = AcneDataset(
            test_paths, test_labels,
            transform=self.transforms_dict['val'],
            is_train=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nSeven-Class Classification DataLoaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
        
        # Print per-class statistics
        print(f"\nClass distribution:")
        for class_idx, class_name in enumerate(class_names):
            train_count = sum(1 for label in train_labels if label == class_idx)
            val_count = sum(1 for label in val_labels if label == class_idx)
            test_count = sum(1 for label in test_labels if label == class_idx)
            print(f"  {class_name:15s}: Train={train_count:4d}, Val={val_count:4d}, Test={test_count:4d}")
        
        return train_loader, val_loader, test_loader


class DataAugmentationAnalyzer:
    """
    Analyze and visualize data augmentation effects
    """
    
    def __init__(self, transform):
        self.transform = transform
    
    def visualize_augmentation(self, image_path: str, num_samples: int = 10,
                              save_path: str = None):
        """
        Visualize augmentation effects on a single image
        
        Args:
            image_path: Path to input image
            num_samples: Number of augmented samples to generate
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        img = Image.open(image_path).convert('RGB')
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented samples
        for i in range(1, num_samples):
            augmented = self.transform(img)
            
            # Convert tensor to image for visualization
            if isinstance(augmented, torch.Tensor):
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                augmented = augmented * std + mean
                augmented = augmented.permute(1, 2, 0).numpy()
                augmented = np.clip(augmented, 0, 1)
            
            axes[i].imshow(augmented)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Augmentation visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_data_preparation():
    """Example of complete data preparation pipeline"""
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION EXAMPLE")
    print("=" * 80)
    
    # Step 1: Organize raw data
    print("\nStep 1: Organize binary classification data")
    organizer_binary = DatasetOrganizer(
        source_dir='./raw_data/binary',
        target_dir='./data/binary'
    )
    # Uncomment to run:
    # organizer_binary.organize_binary_dataset(
    #     skin_keywords=['skin', 'face', 'facial'],
    #     non_skin_keywords=['background', 'hair', 'nonskin']
    # )
    
    print("\nStep 2: Organize seven-class data")
    organizer_seven = DatasetOrganizer(
        source_dir='./raw_data/seven_class',
        target_dir='./data/seven_class'
    )
    # Uncomment to run:
    # organizer_seven.organize_seven_class_dataset({
    #     'papule': ['papule', 'pap'],
    #     'cyst': ['cyst'],
    #     'blackhead': ['blackhead', 'black'],
    #     'normal_skin': ['normal', 'healthy'],
    #     'pustule': ['pustule', 'pus'],
    #     'whitehead': ['whitehead', 'white'],
    #     'nodule': ['nodule', 'nod']
    # })
    
    # Step 3: Preprocess images
    print("\nStep 3: Preprocess images")
    preprocessor = ImagePreprocessor(target_size=(50, 50))
    # Uncomment to run:
    # preprocessor.preprocess_directory(
    #     input_dir='./data/binary/skin',
    #     output_dir='./data/binary_processed/skin'
    # )
    
    # Step 4: Extract patches from large images
    print("\nStep 4: Extract patches (if needed)")
    # Uncomment to run:
    # patches = preprocessor.crop_patches(
    #     image_path='./large_image.jpg',
    #     patch_size=50,
    #     stride=25,
    #     output_dir='./data/patches'
    # )
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)


def example_data_loading():
    """Example of creating data loaders"""
    
    print("\n" + "=" * 80)
    print("DATA LOADING EXAMPLE")
    print("=" * 80)
    
    from acne_detection_main import Config, AcneAugmentation
    
    config = Config()
    augmentation = AcneAugmentation(config)
    
    transforms_dict = {
        'train': augmentation.get_train_transform(),
        'val': augmentation.get_val_transform()
    }
    
    # Binary classification loaders
    print("\nCreating binary classification loaders...")
    loader_binary = AcneDataLoader(
        data_dir='./data/binary',
        transforms_dict=transforms_dict,
        train_split=0.8,
        val_split=0.1,
        random_seed=1337
    )
    
    # Uncomment to create actual loaders:
    # train_loader, val_loader, test_loader = loader_binary.create_binary_loaders(
    #     batch_size=64,
    #     num_workers=4
    # )
    
    # Seven-class loaders
    print("\nCreating seven-class loaders...")
    loader_seven = AcneDataLoader(
        data_dir='./data/seven_class',
        transforms_dict=transforms_dict,
        train_split=0.8,
        val_split=0.1,
        random_seed=1337
    )
    
    # Uncomment to create actual loaders:
    # class_names = ['papule', 'cyst', 'blackhead', 'normal_skin', 
    #                'pustule', 'whitehead', 'nodule']
    # train_loader, val_loader, test_loader = loader_seven.create_seven_class_loaders(
    #     class_names=class_names,
    #     batch_size=64,
    #     num_workers=4
    # )
    
    print("\n" + "=" * 80)
    print("DATA LOADING SETUP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    example_data_preparation()
    example_data_loading()
