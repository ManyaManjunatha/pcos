"""
Kaggle Dataset Integration Script
Helps download and prepare Kaggle datasets for acne detection
"""

import os
import subprocess
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import zipfile
from PIL import Image
import numpy as np


class KaggleDatasetHandler:
    """
    Handle Kaggle dataset download and preparation
    """
    
    def __init__(self, kaggle_dataset: str, download_path: str = './kaggle_data'):
        """
        Args:
            kaggle_dataset: Kaggle dataset identifier (e.g., 'username/dataset-name')
            download_path: Where to download the dataset
        """
        self.kaggle_dataset = kaggle_dataset
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
    def setup_kaggle_credentials(self):
        """
        Check and setup Kaggle API credentials
        """
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("\n" + "=" * 80)
            print("KAGGLE API SETUP REQUIRED")
            print("=" * 80)
            print("\nKaggle API credentials not found!")
            print("\nTo setup Kaggle API:")
            print("1. Go to https://www.kaggle.com/")
            print("2. Click on your profile picture → Account")
            print("3. Scroll to 'API' section")
            print("4. Click 'Create New API Token'")
            print("5. This downloads kaggle.json")
            print(f"6. Move kaggle.json to: {kaggle_dir}/")
            print("\nOn Linux/Mac:")
            print(f"   mkdir -p {kaggle_dir}")
            print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
            print(f"   chmod 600 {kaggle_dir}/kaggle.json")
            print("\nOn Windows:")
            print(f"   mkdir {kaggle_dir}")
            print(f"   move Downloads\\kaggle.json {kaggle_dir}\\")
            print("\n" + "=" * 80)
            return False
        
        # Check permissions on Linux/Mac
        if os.name != 'nt':
            current_permissions = oct(os.stat(kaggle_json).st_mode)[-3:]
            if current_permissions != '600':
                print(f"Setting correct permissions for kaggle.json...")
                os.chmod(kaggle_json, 0o600)
        
        print("✓ Kaggle API credentials found")
        return True
    
    def download_dataset(self, unzip: bool = True) -> bool:
        """
        Download dataset from Kaggle
        
        Args:
            unzip: Whether to unzip the downloaded files
            
        Returns:
            Success status
        """
        try:
            print(f"\nDownloading dataset: {self.kaggle_dataset}")
            print(f"Destination: {self.download_path}")
            
            # Install kaggle package if not present
            try:
                import kaggle
            except ImportError:
                print("Installing kaggle package...")
                subprocess.run(['pip', 'install', 'kaggle'], check=True)
                import kaggle
            
            # Download dataset
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_files(
                self.kaggle_dataset,
                path=str(self.download_path),
                unzip=unzip
            )
            
            print(f"✓ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            print("\nPlease check:")
            print("1. Dataset name is correct")
            print("2. You have accepted the dataset's terms on Kaggle")
            print("3. Your Kaggle credentials are set up correctly")
            return False
    
    def list_downloaded_files(self) -> List[Path]:
        """List all files in the downloaded dataset"""
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            files.extend(self.download_path.rglob(ext))
        return sorted(files)
    
    def analyze_dataset_structure(self) -> Dict:
        """
        Analyze the structure of the downloaded dataset
        """
        print("\n" + "=" * 80)
        print("ANALYZING DATASET STRUCTURE")
        print("=" * 80)
        
        files = self.list_downloaded_files()
        print(f"\nTotal images found: {len(files)}")
        
        # Analyze directory structure
        unique_dirs = set()
        for f in files:
            relative = f.relative_to(self.download_path)
            if len(relative.parts) > 1:
                unique_dirs.add(relative.parts[0])
        
        print(f"\nTop-level directories: {len(unique_dirs)}")
        for d in sorted(unique_dirs):
            dir_path = self.download_path / d
            count = len(list(dir_path.rglob('*.jpg')) + 
                       list(dir_path.rglob('*.jpeg')) + 
                       list(dir_path.rglob('*.png')))
            print(f"  {d}: {count} images")
        
        # Analyze image sizes
        if files:
            sample_sizes = []
            for f in files[:100]:  # Sample first 100
                try:
                    img = Image.open(f)
                    sample_sizes.append(img.size)
                except:
                    pass
            
            if sample_sizes:
                widths = [s[0] for s in sample_sizes]
                heights = [s[1] for s in sample_sizes]
                print(f"\nImage size statistics (from {len(sample_sizes)} samples):")
                print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
                print(f"  Height: min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
        
        analysis = {
            'total_images': len(files),
            'directories': list(unique_dirs),
            'files': files
        }
        
        return analysis


class KaggleAcneDatasetPreparator:
    """
    Prepare Kaggle acne datasets for our training pipeline
    Supports common Kaggle acne dataset formats
    """
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
    def detect_dataset_format(self) -> str:
        """
        Auto-detect the format of the Kaggle dataset
        """
        print("\n" + "=" * 80)
        print("DETECTING DATASET FORMAT")
        print("=" * 80)
        
        # Check for common patterns
        subdirs = [d.name.lower() for d in self.source_dir.iterdir() if d.is_dir()]
        
        # Format 1: Already organized by class
        if any(acne_type in subdirs for acne_type in 
               ['papule', 'pustule', 'blackhead', 'whitehead', 'nodule', 'cyst']):
            print("✓ Detected: Pre-organized by acne type")
            return 'pre_organized'
        
        # Format 2: Has train/test/val splits
        if 'train' in subdirs or 'test' in subdirs or 'val' in subdirs:
            print("✓ Detected: Train/Test/Val splits")
            return 'split'
        
        # Format 3: Single directory with images
        image_count = len(list(self.source_dir.glob('*.jpg')) + 
                         list(self.source_dir.glob('*.png')))
        if image_count > 0:
            print("✓ Detected: Flat directory with images")
            return 'flat'
        
        # Format 4: Has metadata/labels file
        if (self.source_dir / 'labels.csv').exists() or \
           (self.source_dir / 'metadata.csv').exists():
            print("✓ Detected: Has metadata/labels CSV")
            return 'with_csv'
        
        print("✗ Unknown format")
        return 'unknown'
    
    def prepare_from_pre_organized(self):
        """
        Prepare data that's already organized by class
        """
        print("\nPreparing pre-organized dataset...")
        
        # Map common class names to our standard names
        class_mapping = {
            'papule': 'papule',
            'papules': 'papule',
            'pap': 'papule',
            'pustule': 'pustule',
            'pustules': 'pustule',
            'pus': 'pustule',
            'blackhead': 'blackhead',
            'blackheads': 'blackhead',
            'black': 'blackhead',
            'comedo': 'blackhead',
            'whitehead': 'whitehead',
            'whiteheads': 'whitehead',
            'white': 'whitehead',
            'nodule': 'nodule',
            'nodules': 'nodule',
            'nod': 'nodule',
            'cyst': 'cyst',
            'cysts': 'cyst',
            'normal': 'normal_skin',
            'healthy': 'normal_skin',
            'clear': 'normal_skin',
        }
        
        # Create target directories
        seven_dir = self.target_dir / 'seven_class'
        for class_name in ['papule', 'cyst', 'blackhead', 'normal_skin',
                          'pustule', 'whitehead', 'nodule']:
            (seven_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images
        copied = 0
        for source_class_dir in self.source_dir.iterdir():
            if not source_class_dir.is_dir():
                continue
            
            class_name = source_class_dir.name.lower()
            if class_name in class_mapping:
                target_class = class_mapping[class_name]
                target_dir = seven_dir / target_class
                
                for img_file in source_class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy(img_file, target_dir / img_file.name)
                        copied += 1
                
                print(f"  {class_name} → {target_class}: {len(list((target_dir).glob('*')))} images")
        
        print(f"\n✓ Prepared {copied} images")
        return seven_dir
    
    def prepare_from_split(self):
        """
        Prepare data with train/test/val splits
        """
        print("\nPreparing split dataset...")
        print("Merging train/test/val into single organized structure...")
        
        seven_dir = self.target_dir / 'seven_class'
        for class_name in ['papule', 'cyst', 'blackhead', 'normal_skin',
                          'pustule', 'whitehead', 'nodule']:
            (seven_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Merge all splits
        copied = 0
        for split in ['train', 'test', 'val', 'validation']:
            split_dir = self.source_dir / split
            if not split_dir.exists():
                continue
            
            print(f"\n  Processing {split} split...")
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name.lower()
                # Find matching target class
                for target_class in ['papule', 'cyst', 'blackhead', 'normal_skin',
                                    'pustule', 'whitehead', 'nodule']:
                    if target_class in class_name or class_name in target_class:
                        target_dir = seven_dir / target_class
                        
                        for img_file in class_dir.glob('*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                # Add split prefix to avoid name conflicts
                                new_name = f"{split}_{img_file.name}"
                                shutil.copy(img_file, target_dir / new_name)
                                copied += 1
                        
                        break
        
        print(f"\n✓ Prepared {copied} images")
        return seven_dir
    
    def prepare_from_csv(self, csv_file: Optional[str] = None):
        """
        Prepare data using CSV metadata
        
        Args:
            csv_file: Path to CSV file (auto-detected if None)
        """
        import pandas as pd
        
        print("\nPreparing dataset using CSV metadata...")
        
        # Find CSV file
        if csv_file is None:
            for name in ['labels.csv', 'metadata.csv', 'annotations.csv']:
                csv_path = self.source_dir / name
                if csv_path.exists():
                    csv_file = csv_path
                    break
        
        if csv_file is None:
            print("✗ No CSV file found")
            return None
        
        print(f"Reading: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Try to identify relevant columns
        print("\nPlease specify the column names:")
        print(f"Available columns: {list(df.columns)}")
        
        image_col = input("Image filename column (or press Enter to use first column): ").strip()
        if not image_col:
            image_col = df.columns[0]
        
        label_col = input("Label/class column (or press Enter to use second column): ").strip()
        if not label_col:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        print(f"\nUsing: image_col='{image_col}', label_col='{label_col}'")
        
        # Create target directories
        seven_dir = self.target_dir / 'seven_class'
        unique_labels = df[label_col].unique()
        print(f"\nUnique labels found: {unique_labels}")
        
        for label in unique_labels:
            (seven_dir / str(label)).mkdir(parents=True, exist_ok=True)
        
        # Copy images according to labels
        copied = 0
        for _, row in df.iterrows():
            img_name = row[image_col]
            label = row[label_col]
            
            # Find image file
            img_path = self.source_dir / img_name
            if not img_path.exists():
                # Try in subdirectories
                found = list(self.source_dir.rglob(img_name))
                if found:
                    img_path = found[0]
                else:
                    continue
            
            # Copy to target
            target_dir = seven_dir / str(label)
            shutil.copy(img_path, target_dir / img_name)
            copied += 1
        
        print(f"\n✓ Prepared {copied} images")
        
        # Print statistics
        for label in unique_labels:
            count = len(list((seven_dir / str(label)).glob('*')))
            print(f"  {label}: {count} images")
        
        return seven_dir
    
    def auto_prepare(self):
        """
        Automatically detect format and prepare dataset
        """
        fmt = self.detect_dataset_format()
        
        if fmt == 'pre_organized':
            return self.prepare_from_pre_organized()
        elif fmt == 'split':
            return self.prepare_from_split()
        elif fmt == 'with_csv':
            return self.prepare_from_csv()
        elif fmt == 'flat':
            print("\n⚠ Flat directory detected.")
            print("Please organize images by class or provide a CSV with labels.")
            return None
        else:
            print("\n✗ Cannot auto-detect format.")
            print("Please organize manually or provide more information.")
            return None


def main():
    """
    Interactive script for Kaggle dataset integration
    """
    print("\n" + "=" * 80)
    print("KAGGLE DATASET INTEGRATION FOR ACNE DETECTION")
    print("=" * 80)
    
    print("\nThis script helps you download and prepare Kaggle datasets.")
    print("\nCommon Kaggle acne datasets:")
    print("  1. shonenkov/acne-types-classification-dataset")
    print("  2. rutviklathiya/acne-grading-classifcation-dataset")
    print("  3. Other acne-related datasets")
    
    # Step 1: Setup Kaggle credentials
    print("\n" + "-" * 80)
    print("STEP 1: KAGGLE API SETUP")
    print("-" * 80)
    
    handler = KaggleDatasetHandler('')
    if not handler.setup_kaggle_credentials():
        return
    
    # Step 2: Dataset selection
    print("\n" + "-" * 80)
    print("STEP 2: DATASET SELECTION")
    print("-" * 80)
    
    print("\nOptions:")
    print("  1. Download from Kaggle")
    print("  2. Use already downloaded dataset")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == '1':
        dataset_name = input("\nEnter Kaggle dataset name (e.g., username/dataset-name): ").strip()
        download_path = input("Download path (press Enter for './kaggle_data'): ").strip()
        if not download_path:
            download_path = './kaggle_data'
        
        handler = KaggleDatasetHandler(dataset_name, download_path)
        
        if handler.download_dataset():
            analysis = handler.analyze_dataset_structure()
            source_dir = handler.download_path
        else:
            return
    
    else:
        source_dir = input("\nEnter path to downloaded dataset: ").strip()
        if not os.path.exists(source_dir):
            print(f"✗ Directory not found: {source_dir}")
            return
        
        handler = KaggleDatasetHandler('', source_dir)
        analysis = handler.analyze_dataset_structure()
    
    # Step 3: Prepare dataset
    print("\n" + "-" * 80)
    print("STEP 3: DATASET PREPARATION")
    print("-" * 80)
    
    target_dir = input("\nEnter target directory (press Enter for './data'): ").strip()
    if not target_dir:
        target_dir = './data'
    
    preparator = KaggleAcneDatasetPreparator(source_dir, target_dir)
    result = preparator.auto_prepare()
    
    if result:
        print("\n" + "=" * 80)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("=" * 80)
        print(f"\nDataset prepared in: {result}")
        print("\nNext steps:")
        print("1. Verify the prepared data structure")
        print("2. Run: python train.py --task seven --epochs 50")
        print("   or: python train.py --task all --epochs 50")
    else:
        print("\n" + "=" * 80)
        print("✗ DATASET PREPARATION FAILED")
        print("=" * 80)
        print("\nPlease prepare the dataset manually following README.md instructions.")


if __name__ == "__main__":
    main()
