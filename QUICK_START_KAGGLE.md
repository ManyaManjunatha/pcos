# 🚀 Quick Start: Kaggle Dataset → Training (5 Minutes)

## One-Command Setup

```bash
# Install everything
pip install -r requirements.txt kaggle

# Setup Kaggle API (one-time)
# 1. Get kaggle.json from https://www.kaggle.com/account
# 2. Place it: ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)

# Download and prepare dataset (interactive)
python kaggle_integration.py

# Start training
python train.py --task seven --epochs 50 --gpu 0
```

Done! 🎉

---

## Popular Kaggle Datasets

### Most Common: Acne Types Classification
```bash
# Option 1: Fully automated
python kaggle_integration.py
# Then enter: shonenkov/acne-types-classification-dataset

# Option 2: Manual
kaggle datasets download -d shonenkov/acne-types-classification-dataset
unzip acne-types-classification-dataset.zip -d ./kaggle_data/
python kaggle_integration.py
# Select option 2 (already downloaded)
```

---

## Troubleshooting (90% of Issues)

### Problem: "Kaggle credentials not found"
```bash
# Download kaggle.json from https://www.kaggle.com/account
# Then:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Problem: "403 Forbidden"
- Go to the dataset page on Kaggle
- Click "Download" button to accept terms
- Try again

### Problem: Dataset structure not recognized
```bash
# Let the script analyze it
python kaggle_integration.py
# Or check manually:
ls -R kaggle_data/ | head -30
```

---

## Complete Example (Copy-Paste Ready)

```bash
#!/bin/bash

# 1. Install dependencies
pip install torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm kaggle

# 2. Setup Kaggle (if not done)
echo "Place your kaggle.json in ~/.kaggle/"
echo "Download from: https://www.kaggle.com/account"
read -p "Press Enter when ready..."

# 3. Download dataset
python kaggle_integration.py << EOF
1
shonenkov/acne-types-classification-dataset

EOF

# 4. Train
python train.py --task seven --epochs 50 --batch-size 64 --gpu 0

# 5. Test inference
python inference.py --mode demo \
    --binary-model checkpoints/seven_class_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image test_image.jpg \
    --output demo_results/

echo "✓ Complete! Check results in demo_results/"
```

---

## Alternative: Test with Synthetic Data First

```bash
# Generate fake data for testing
python quickstart.py
# Select option 7

# Train on synthetic data
python train.py --task seven --epochs 5 --batch-size 32

# Verify everything works
# Then use real Kaggle data
```

---

## Most Common Workflow

```python
# Step 1: Download (one time)
from kaggle_integration import KaggleDatasetHandler

handler = KaggleDatasetHandler(
    kaggle_dataset='shonenkov/acne-types-classification-dataset',
    download_path='./kaggle_data'
)
handler.download_dataset()

# Step 2: Prepare (one time)
from kaggle_integration import KaggleAcneDatasetPreparator

prep = KaggleAcneDatasetPreparator('./kaggle_data', './data')
prep.auto_prepare()

# Step 3: Train (run as needed)
import subprocess
subprocess.run([
    'python', 'train.py',
    '--task', 'seven',
    '--epochs', '50',
    '--gpu', '0'
])
```

---

## File Locations Reference

```
Your Project/
├── kaggle_integration.py      # ← New! Handles Kaggle downloads
├── train.py                    # ← Use this to train
├── inference.py                # ← Use this for predictions
├── requirements.txt            # ← pip install -r requirements.txt
│
├── kaggle_data/               # ← Downloaded Kaggle data
│   └── (dataset files)
│
├── data/                      # ← Prepared for training
│   └── seven_class/
│       ├── papule/
│       ├── cyst/
│       └── ...
│
├── checkpoints/               # ← Trained models
│   └── *.pth
│
└── results/                   # ← Training curves, metrics
    └── *.png
```

---

## 3 Ways to Get Started

### Way 1: Fully Automated (Easiest)
```bash
python kaggle_integration.py
# Follow prompts
python train.py --task seven --epochs 50
```

### Way 2: Semi-Automated
```bash
# Download manually from Kaggle website
# Unzip to ./kaggle_data/
python kaggle_integration.py
# Select "Use already downloaded"
python train.py --task seven --epochs 50
```

### Way 3: Manual (Full Control)
```bash
# 1. Download from Kaggle
# 2. Organize into:
#    data/seven_class/papule/
#    data/seven_class/cyst/
#    etc.
# 3. Train
python train.py --task seven --epochs 50
```

---

## Quick Checks

### Check 1: Is Kaggle API working?
```bash
kaggle datasets list
```
Should show datasets. If not, check credentials.

### Check 2: Is data prepared correctly?
```bash
ls -R data/seven_class/ | grep -E '/$' | wc -l
```
Should show 7 directories (one per class).

### Check 3: Can I load the data?
```python
from data_utils import AcneDataLoader
from acne_detection_main import Config, AcneAugmentation

config = Config()
aug = AcneAugmentation(config)
loader = AcneDataLoader('./data/seven_class', 
                       {'train': aug.get_train_transform(), 
                        'val': aug.get_val_transform()})
train, val, test = loader.create_seven_class_loaders(config.CLASS_NAMES)
print(f"✓ {len(train.dataset)} training images")
```

### Check 4: Can I train?
```bash
# Quick test with 2 epochs
python train.py --task seven --epochs 2 --batch-size 16
```

---

## Expected Timeline

- **Kaggle API Setup:** 5 minutes (one-time)
- **Dataset Download:** 2-10 minutes (depends on size)
- **Data Preparation:** 1-5 minutes
- **Training (50 epochs):** 1-4 hours (depends on GPU/dataset size)
- **Inference:** Seconds per image

---

## Need Help?

1. **Read full guide:** `KAGGLE_GUIDE.md`
2. **Check main docs:** `README.md`
3. **Try examples:** `python quickstart.py`

---

**You're ready! Start with:**
```bash
python kaggle_integration.py
```
