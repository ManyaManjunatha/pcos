# 📊 Complete Guide: Using Kaggle Datasets with Acne Detection System

## 🎯 Quick Start

### Option 1: Automated (Recommended)
```bash
python kaggle_integration.py
```
Follow the interactive prompts!

### Option 2: Manual Steps
Follow this guide for manual setup.

---

## 📋 Step-by-Step Guide

### Step 1: Install Kaggle Package

```bash
pip install kaggle
```

### Step 2: Setup Kaggle API Credentials

1. **Get your API token:**
   - Go to https://www.kaggle.com/
   - Click on your profile picture → **Account**
   - Scroll to **API** section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json`

2. **Place the credentials file:**

   **On Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **On Windows:**
   ```cmd
   mkdir %USERPROFILE%\.kaggle
   move Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

3. **Verify setup:**
   ```bash
   kaggle datasets list
   ```
   If this works, you're all set!

---

## 🗂️ Popular Kaggle Acne Datasets

### 1. Acne Types Classification Dataset
- **ID:** `shonenkov/acne-types-classification-dataset`
- **Size:** ~5,000 images
- **Classes:** Multiple acne types
- **Format:** Pre-organized by class

**Download:**
```bash
kaggle datasets download -d shonenkov/acne-types-classification-dataset
unzip acne-types-classification-dataset.zip -d ./kaggle_data/
```

### 2. Acne Grading Classification
- **ID:** `rutviklathiya/acne-grading-classifcation-dataset`
- **Size:** Variable
- **Classes:** Severity grades
- **Format:** May need adaptation

**Download:**
```bash
kaggle datasets download -d rutviklathiya/acne-grading-classifcation-dataset
unzip acne-grading-classifcation-dataset.zip -d ./kaggle_data/
```

### 3. Search for More
```bash
kaggle datasets list -s acne
```

---

## 🔄 Dataset Format Conversion

Kaggle datasets come in various formats. Here's how to convert them:

### Format 1: Already Organized by Type (Best Case)

**Structure:**
```
kaggle_data/
├── papule/
├── pustule/
├── blackhead/
└── ...
```

**Action:** Direct copy!
```python
from kaggle_integration import KaggleAcneDatasetPreparator

prep = KaggleAcneDatasetPreparator(
    source_dir='./kaggle_data',
    target_dir='./data'
)
prep.prepare_from_pre_organized()
```

### Format 2: Train/Test/Val Splits

**Structure:**
```
kaggle_data/
├── train/
│   ├── papule/
│   ├── pustule/
│   └── ...
├── test/
│   └── ...
└── val/
    └── ...
```

**Action:** Merge splits
```python
prep = KaggleAcneDatasetPreparator(
    source_dir='./kaggle_data',
    target_dir='./data'
)
prep.prepare_from_split()
```

### Format 3: CSV with Labels

**Structure:**
```
kaggle_data/
├── images/
├── labels.csv
```

**labels.csv:**
```csv
filename,label
img001.jpg,papule
img002.jpg,pustule
...
```

**Action:** Use CSV
```python
prep = KaggleAcneDatasetPreparator(
    source_dir='./kaggle_data',
    target_dir='./data'
)
prep.prepare_from_csv('kaggle_data/labels.csv')
```

### Format 4: Flat Directory (Needs Manual Work)

**Structure:**
```
kaggle_data/
├── img001.jpg
├── img002.jpg
└── ...
```

**Action:** You'll need to manually organize or create a CSV with labels.

---

## 🐍 Python Script Examples

### Example 1: Download and Prepare in One Go

```python
from kaggle_integration import KaggleDatasetHandler, KaggleAcneDatasetPreparator

# Download
handler = KaggleDatasetHandler(
    kaggle_dataset='shonenkov/acne-types-classification-dataset',
    download_path='./kaggle_data'
)

if handler.setup_kaggle_credentials():
    handler.download_dataset(unzip=True)
    handler.analyze_dataset_structure()
    
    # Prepare
    prep = KaggleAcneDatasetPreparator(
        source_dir='./kaggle_data',
        target_dir='./data'
    )
    prep.auto_prepare()
```

### Example 2: Use Already Downloaded Dataset

```python
from kaggle_integration import KaggleDatasetHandler, KaggleAcneDatasetPreparator

# Analyze existing dataset
handler = KaggleDatasetHandler('', './my_downloaded_data')
analysis = handler.analyze_dataset_structure()

# Prepare for training
prep = KaggleAcneDatasetPreparator(
    source_dir='./my_downloaded_data',
    target_dir='./data'
)
result = prep.auto_prepare()
```

### Example 3: Custom CSV Mapping

```python
from kaggle_integration import KaggleAcneDatasetPreparator

prep = KaggleAcneDatasetPreparator(
    source_dir='./kaggle_data',
    target_dir='./data'
)

# Specify custom CSV columns
prep.prepare_from_csv(
    csv_file='./kaggle_data/annotations.csv'
)
# Script will ask for column names interactively
```

---

## 📁 Expected Output Structure

After preparation, your data should look like:

```
data/
└── seven_class/
    ├── papule/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── cyst/
    ├── blackhead/
    ├── normal_skin/
    ├── pustule/
    ├── whitehead/
    └── nodule/
```

---

## 🚀 After Dataset Preparation

Once your dataset is prepared:

### 1. Verify the Structure
```python
from data_utils import AcneDataLoader
from acne_detection_main import Config, AcneAugmentation

config = Config()
augmentation = AcneAugmentation(config)
transforms_dict = {
    'train': augmentation.get_train_transform(),
    'val': augmentation.get_val_transform()
}

loader = AcneDataLoader(
    data_dir='./data/seven_class',
    transforms_dict=transforms_dict
)

# This will print statistics
train_loader, val_loader, test_loader = loader.create_seven_class_loaders(
    class_names=config.CLASS_NAMES,
    batch_size=64
)
```

### 2. Start Training
```bash
# Train seven-class model
python train.py --task seven --epochs 50 --batch-size 64 --gpu 0

# Or train everything
python train.py --task all --epochs 50 --gpu 0
```

### 3. Monitor Progress
Training will create:
- `checkpoints/` - Saved models
- `results/` - Training curves and metrics

---

## 🔧 Troubleshooting

### Issue 1: "Kaggle API credentials not found"
**Solution:**
```bash
# Verify file exists
ls -la ~/.kaggle/kaggle.json

# Check permissions (should be 600)
chmod 600 ~/.kaggle/kaggle.json
```

### Issue 2: "403 Forbidden" when downloading
**Solution:**
- Accept the dataset's terms on Kaggle website
- Make sure you're logged in to Kaggle

### Issue 3: Dataset structure not recognized
**Solution:**
```python
# Manually inspect
from pathlib import Path
import os

data_path = Path('./kaggle_data')
for root, dirs, files in os.walk(data_path):
    print(f"\n{root}:")
    print(f"  Subdirs: {dirs[:5]}")  # First 5 subdirs
    print(f"  Files: {len(files)} files")
    if files:
        print(f"  Examples: {files[:3]}")
```

### Issue 4: Class names don't match
**Solution:**
Create custom mapping in `kaggle_integration.py`:
```python
class_mapping = {
    'your_class_name': 'papule',
    'another_name': 'pustule',
    # ... add your mappings
}
```

### Issue 5: Images wrong size
**Solution:**
Images will be automatically resized during training. But if you want to preprocess:
```python
from data_utils import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(50, 50))
preprocessor.preprocess_directory(
    input_dir='./data/seven_class/papule',
    output_dir='./data_processed/seven_class/papule'
)
```

---

## 📊 Dataset Quality Checks

### Check 1: Verify Image Count
```python
from pathlib import Path

data_dir = Path('./data/seven_class')
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        count = len(list(class_dir.glob('*.jpg')) + 
                   list(class_dir.glob('*.png')))
        print(f"{class_dir.name}: {count} images")
```

### Check 2: Verify Image Loading
```python
from PIL import Image
from pathlib import Path

data_dir = Path('./data/seven_class')
corrupt_files = []

for img_path in data_dir.rglob('*.jpg'):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        corrupt_files.append(img_path)
        print(f"Corrupt: {img_path}")

print(f"\nTotal corrupt files: {len(corrupt_files)}")
```

### Check 3: Visualize Samples
```python
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

data_dir = Path('./data/seven_class')
fig, axes = plt.subplots(3, 7, figsize=(21, 9))

for idx, class_dir in enumerate(data_dir.iterdir()):
    if class_dir.is_dir():
        images = list(class_dir.glob('*.jpg'))[:3]
        for row, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[row, idx].imshow(img)
            axes[row, idx].set_title(class_dir.name)
            axes[row, idx].axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150)
print("✓ Saved dataset_samples.png")
```

---

## 📝 Common Kaggle Dataset Scenarios

### Scenario A: Perfect Pre-organized Dataset
```bash
# Download
python kaggle_integration.py

# Train immediately
python train.py --task seven --epochs 50
```

### Scenario B: Dataset with Train/Test Splits
```python
# Use the integration script
python kaggle_integration.py
# Select option 1 or 2
# Script will automatically merge splits

# Then train
python train.py --task seven --epochs 50
```

### Scenario C: Dataset with CSV Labels
```python
# Run integration script
python kaggle_integration.py
# When prompted, provide CSV column names

# Verify preparation
ls -R data/seven_class/

# Train
python train.py --task seven --epochs 50
```

### Scenario D: Custom/Unknown Format
```bash
# 1. Download manually from Kaggle website
# 2. Analyze structure
python -c "
from kaggle_integration import KaggleDatasetHandler
h = KaggleDatasetHandler('', './your_data')
h.analyze_dataset_structure()
"

# 3. Organize manually or modify kaggle_integration.py
# 4. Train
python train.py --task seven --epochs 50
```

---

## 🎓 Advanced: Custom Dataset Adapter

If your Kaggle dataset has a unique structure:

```python
# custom_adapter.py
from pathlib import Path
import shutil

def adapt_my_kaggle_dataset(source, target):
    """
    Custom adapter for your specific Kaggle dataset
    """
    source = Path(source)
    target = Path(target)
    
    # Your custom logic here
    # Example: if images are in weird subdirectories
    for img_path in source.rglob('*.jpg'):
        # Extract label from filename or path
        if 'pimple' in str(img_path):
            label = 'papule'
        elif 'cyst' in str(img_path):
            label = 'cyst'
        # ... etc
        
        # Copy to target
        target_dir = target / 'seven_class' / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, target_dir / img_path.name)

# Use it
adapt_my_kaggle_dataset('./kaggle_data', './data')
```

---

## ✅ Complete Workflow Example

```bash
# 1. Setup Kaggle API
pip install kaggle
# (Place kaggle.json in ~/.kaggle/)

# 2. Download and prepare dataset
python kaggle_integration.py

# 3. Verify preparation
python -c "
from data_utils import AcneDataLoader
from acne_detection_main import Config, AcneAugmentation

config = Config()
aug = AcneAugmentation(config)
transforms = {'train': aug.get_train_transform(), 'val': aug.get_val_transform()}

loader = AcneDataLoader('./data/seven_class', transforms)
train, val, test = loader.create_seven_class_loaders(config.CLASS_NAMES)
print(f'✓ Data ready: {len(train.dataset)} train, {len(val.dataset)} val, {len(test.dataset)} test')
"

# 4. Train model
python train.py --task seven --epochs 50 --batch-size 64 --gpu 0

# 5. Run inference
python inference.py --mode demo \
    --binary-model checkpoints/seven_class_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image test_face.jpg
```

---

## 🆘 Need Help?

1. **Check dataset format:**
   ```bash
   python kaggle_integration.py
   ```

2. **Inspect manually:**
   ```bash
   ls -R kaggle_data/ | head -50
   ```

3. **Test with synthetic data first:**
   ```bash
   python quickstart.py
   # Select option 7 to create synthetic data
   ```

4. **Check the main README:**
   ```bash
   cat README.md
   ```

---

## 📚 Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

**Happy Training! 🚀**
