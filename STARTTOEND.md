#  Complete Step-by-Step Guide: Kaggle Dataset → Trained Model

Let me walk you through the **entire process from scratch**!

##  Part 1: Initial Setup (One-Time)

### Step 1: Install Python Packages

```bash
# Install all required packages
pip install torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm kaggle
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
pip install kaggle
```

### Step 2: Setup Kaggle API Credentials

#### A. Get Your API Token

1. Go to **https://www.kaggle.com/**
2. Click on your **profile picture** (top right)
3. Click **"Account"**
4. Scroll down to **"API"** section
5. Click **"Create New API Token"**
6. This downloads `kaggle.json` to your Downloads folder

#### B. Place the Credentials File

**On Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows (Command Prompt):**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

**On Windows (PowerShell):**
```powershell
New-Item -Path "$env:USERPROFILE\.kaggle" -ItemType Directory -Force
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"
```

#### C. Verify Kaggle API Works

```bash
kaggle datasets list
```

If you see a list of datasets, you're good! 

---

## 📊 Part 2: Download & Prepare Dataset

### Option A: Fully Automated (Easiest!) 

```bash
python kaggle_integration.py
```

**Follow the prompts:**
```
1. Select: "1. Download from Kaggle"
2. Enter dataset name: shonenkov/acne-types-classification-dataset
3. Press Enter for default download path
4. Press Enter for default target directory
5. Wait for download and preparation
```

That's it! Skip to **Part 3: Training**.

---

### Option B: Manual Step-by-Step

#### Step 1: Download Dataset from Kaggle

**Popular acne datasets:**

```bash
# Option 1: Acne Types Classification (Recommended)
kaggle datasets download -d shonenkov/acne-types-classification-dataset
unzip acne-types-classification-dataset.zip -d ./kaggle_data/

# Option 2: Acne Grading Classification
kaggle datasets download -d rutviklathiya/acne-grading-classifcation-dataset
unzip acne-grading-classifcation-dataset.zip -d ./kaggle_data/

# Option 3: Search for other datasets
kaggle datasets list -s acne
```

#### Step 2: Analyze What You Downloaded

```bash
# See what's in the dataset
ls -R kaggle_data/ | head -50
```

#### Step 3: Prepare for Training

**Method 1: Use the integration script (recommended)**
```bash
python kaggle_integration.py
# Select: "2. Use already downloaded dataset"
# Enter path: ./kaggle_data
```

**Method 2: Use Python directly**
```python
from kaggle_integration import KaggleAcneDatasetPreparator

prep = KaggleAcneDatasetPreparator(
    source_dir='./kaggle_data',
    target_dir='./data'
)

# Auto-detect format and prepare
prep.auto_prepare()
```

---

##  Part 3: Verify Data Preparation

### Check 1: Directory Structure

```bash
# Should show 7 directories
ls -la data/seven_class/
```

**Expected output:**
```
papule/
cyst/
blackhead/
normal_skin/
pustule/
whitehead/
nodule/
```

### Check 2: Image Counts

```bash
# Count images in each class
for dir in data/seven_class/*; do
    echo "$(basename $dir): $(ls $dir | wc -l) images"
done
```

### Check 3: Test Data Loading

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

train_loader, val_loader, test_loader = loader.create_seven_class_loaders(
    class_names=config.CLASS_NAMES,
    batch_size=64
)

print(f"✓ Training set: {len(train_loader.dataset)} images")
print(f"✓ Validation set: {len(val_loader.dataset)} images")
print(f"✓ Test set: {len(test_loader.dataset)} images")
```

---

## 🎓 Part 4: Training

### Quick Test Training (2 minutes)

```bash
# Test with just 2 epochs to make sure everything works
python train.py \
    --task seven \
    --epochs 2 \
    --batch-size 16 \
    --gpu 0
```

### Full Training (1-4 hours depending on GPU)

**Option 1: Seven-Class Only**
```bash
python train.py \
    --task seven \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --gpu 0
```

**Option 2: Everything (Binary + Seven-Class)**
```bash
python train.py \
    --task all \
    --epochs 50 \
    --batch-size 64 \
    --gpu 0
```

**Option 3: With Fine-Tuning**
```bash
python train.py \
    --task seven \
    --fine-tune \
    --epochs 50 \
    --batch-size 64 \
    --gpu 0
```

**If you don't have GPU:**
```bash
python train.py \
    --task seven \
    --epochs 50 \
    --batch-size 32 \
    --gpu -1
```

### Monitor Training

Training will display:
```
Epoch 1/50
Training: 100%|███████████| 125/125 [02:15<00:00]
Train Loss: 1.2345, Train Acc: 65.23%
Val Loss: 1.1234, Val Acc: 68.45%
✓ Best accuracy model saved!
```

Files created during training:
```
checkpoints/
├── seven_class_vgg16_best_loss.pth    # Model with best loss
└── seven_class_vgg16_best_acc.pth     # Model with best accuracy

results/
├── seven_class_vgg16_training_history.png   # Training curves
└── seven_class_vgg16_confusion_matrix.png   # Test results
```

---

## 🔍 Part 5: Evaluation

After training completes, the script automatically evaluates on the test set and shows:

```
================================================================================
SEVEN-CLASS CLASSIFICATION RESULTS
================================================================================
Overall Accuracy: 86.45%

Per-Class Accuracy:
--------------------------------------------------------------------------------
  papule         : 82.30%
  cyst           : 84.50%
  blackhead      : 91.20%
  normal_skin    : 94.80%
  pustule        : 81.00%
  whitehead      : 88.50%
  nodule         : 86.10%
================================================================================
```

View the results:
```bash
# Training history
open results/seven_class_vgg16_training_history.png

# Confusion matrix
open results/seven_class_vgg16_confusion_matrix.png
```

---

##  Part 6: Inference / Prediction

### Test on a Single Image

```bash
python inference.py \
    --mode single \
    --binary-model checkpoints/seven_class_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image test_face.jpg \
    --output results/single_test/
```

**Output:**
- `results/single_test/test_face_results.png` - Visualization
- `results/single_test/test_face_report.json` - JSON report

### Create Demo Visualization

```bash
python inference.py \
    --mode demo \
    --binary-model checkpoints/seven_class_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image demo_face.jpg \
    --output demo_results/
```

### Batch Process Multiple Images

```bash
python inference.py \
    --mode batch \
    --binary-model checkpoints/seven_class_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image-dir test_images/ \
    --output batch_results/
```

---

##  Complete Example: Copy-Paste Script

Create a file `run_complete_pipeline.sh`:

```bash
#!/bin/bash

echo "================================"
echo "ACNE DETECTION COMPLETE PIPELINE"
echo "================================"

# 1. Install dependencies
echo -e "\n[1/6] Installing dependencies..."
pip install -q torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm kaggle

# 2. Check Kaggle credentials
echo -e "\n[2/6] Checking Kaggle API..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo " Kaggle credentials not found!"
    echo "Please download kaggle.json from https://www.kaggle.com/account"
    echo "And place it in ~/.kaggle/kaggle.json"
    exit 1
fi
echo "✓ Kaggle credentials found"

# 3. Download dataset
echo -e "\n[3/6] Downloading dataset..."
kaggle datasets download -d shonenkov/acne-types-classification-dataset
unzip -q acne-types-classification-dataset.zip -d ./kaggle_data/
echo "✓ Dataset downloaded"

# 4. Prepare dataset
echo -e "\n[4/6] Preparing dataset..."
python -c "
from kaggle_integration import KaggleAcneDatasetPreparator
prep = KaggleAcneDatasetPreparator('./kaggle_data', './data')
result = prep.auto_prepare()
print(f'✓ Dataset prepared: {result}')
"

# 5. Quick test training (2 epochs)
echo -e "\n[5/6] Quick test training (2 epochs)..."
python train.py --task seven --epochs 2 --batch-size 16 --gpu 0

# 6. Full training
echo -e "\n[6/6] Starting full training..."
read -p "Start full training (50 epochs, ~1-4 hours)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train.py --task seven --epochs 50 --batch-size 64 --gpu 0
    echo "✓ Training complete!"
    echo "Models saved in: checkpoints/"
    echo "Results in: results/"
fi

echo -e "\n================================"
echo "PIPELINE COMPLETE!"
echo "================================"
```

**Run it:**
```bash
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

---

##  Complete Python Script

Create `run_all.py`:

```python
"""
Complete pipeline: Download → Prepare → Train → Evaluate
"""

import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print progress"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f" Failed: {description}")
        return False
    print(f"✓ Completed: {description}")
    return True

def main():
    print("\n" + "="*60)
    print("ACNE DETECTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Check Kaggle credentials
    kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_path.exists():
        print("\n Kaggle credentials not found!")
        print("Please setup Kaggle API first:")
        print("1. Download kaggle.json from https://www.kaggle.com/account")
        print(f"2. Place it at: {kaggle_path}")
        return
    
    # Step 1: Download dataset
    if not run_command(
        "python kaggle_integration.py",
        "Step 1/4: Downloading and preparing dataset"
    ):
        return
    
    # Step 2: Verify data
    print("\n" + "="*60)
    print("Step 2/4: Verifying dataset")
    print("="*60)
    
    from data_utils import AcneDataLoader
    from acne_detection_main import Config, AcneAugmentation
    
    config = Config()
    aug = AcneAugmentation(config)
    transforms = {
        'train': aug.get_train_transform(),
        'val': aug.get_val_transform()
    }
    
    loader = AcneDataLoader('./data/seven_class', transforms)
    train, val, test = loader.create_seven_class_loaders(
        config.CLASS_NAMES, batch_size=64
    )
    
    print(f"✓ Training: {len(train.dataset)} images")
    print(f"✓ Validation: {len(val.dataset)} images")
    print(f"✓ Test: {len(test.dataset)} images")
    
    # Step 3: Quick test
    print("\n" + "="*60)
    choice = input("Run quick test training (2 epochs)? (y/n): ")
    if choice.lower() == 'y':
        run_command(
            "python train.py --task seven --epochs 2 --batch-size 16 --gpu 0",
            "Step 3/4: Quick test training"
        )
    
    # Step 4: Full training
    print("\n" + "="*60)
    choice = input("Run full training (50 epochs, ~1-4 hours)? (y/n): ")
    if choice.lower() == 'y':
        run_command(
            "python train.py --task seven --epochs 50 --batch-size 64 --gpu 0",
            "Step 4/4: Full training"
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  checkpoints/ - Trained models")
        print("  results/     - Training curves and metrics")
        
        print("\nNext: Run inference")
        print("  python inference.py --mode demo \\")
        print("      --binary-model checkpoints/seven_class_vgg16_best_acc.pth \\")
        print("      --seven-model checkpoints/seven_class_vgg16_best_acc.pth \\")
        print("      --image your_test_image.jpg")

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python run_all.py
```

---

## 📊 Expected Timeline

| Step | Time | What Happens |
|------|------|--------------|
| Setup Kaggle API | 5 min | One-time credential setup |
| Download dataset | 2-10 min | Depends on dataset size |
| Prepare data | 1-5 min | Convert to training format |
| Quick test (2 epochs) | 2-5 min | Verify everything works |
| Full training (50 epochs) | 1-4 hours | Actual model training |
| Inference | Seconds | Test on new images |

---

## 🔍 Monitoring Training Progress

While training, you can:

1. **Watch the console output:**
   ```
   Epoch 23/50
   Training: 100%|████████| Loss: 0.456, Acc: 87.2%
   Val Loss: 0.523, Val Acc: 85.1%
   ```

2. **Check saved files:**
   ```bash
   ls -lh checkpoints/
   ls -lh results/
   ```

3. **View training curves (after completion):**
   ```python
   from PIL import Image
   img = Image.open('results/seven_class_vgg16_training_history.png')
   img.show()
   ```

---

##  Troubleshooting Common Issues

### Issue 1: "Kaggle credentials not found"
```bash
# Verify file exists
ls -la ~/.kaggle/kaggle.json

# Re-download if needed from https://www.kaggle.com/account
```

### Issue 2: "CUDA out of memory"
```bash
# Reduce batch size
python train.py --task seven --epochs 50 --batch-size 32 --gpu 0

# Or use CPU (slower)
python train.py --task seven --epochs 50 --batch-size 16 --gpu -1
```

### Issue 3: "No images found"
```bash
# Check data structure
ls -R data/seven_class/

# Re-run preparation
python kaggle_integration.py
```

### Issue 4: "Model not found" during inference
```bash
# Check if model was saved
ls -lh checkpoints/

# Use the correct model path
python inference.py --seven-model checkpoints/seven_class_vgg16_best_acc.pth ...
```

---

##  Final Checklist

Before starting:
- [ ] Python 3.8+ installed
- [ ] GPU drivers installed (if using GPU)
- [ ] Kaggle account created
- [ ] kaggle.json downloaded and placed correctly
- [ ] All files from this project downloaded

After setup:
- [ ] `kaggle datasets list` works
- [ ] Dataset downloaded to `./kaggle_data/`
- [ ] Data prepared in `./data/seven_class/`
- [ ] Can see 7 subdirectories (papule, cyst, etc.)

After training:
- [ ] Models in `./checkpoints/`
- [ ] Results in `./results/`
- [ ] Can run inference on test images

---

##  You're Ready!

**Start here:**
```bash
python kaggle_integration.py
```

Good luck! 
