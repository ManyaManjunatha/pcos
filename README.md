# Automatic Diagnosis of Facial Acne Vulgaris using Convolutional Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📄 Paper Reference

This implementation is based on the research paper:

**"An Automatic Diagnosis Method of Facial Acne Vulgaris Based on Convolutional Neural Network"**  
*Xiaolei Shen, Jiachi Zhang, Chenjun Yan & Hong Zhou*  
*Scientific Reports (2018) 8:5839*  
DOI: [10.1038/s41598-018-24204-6](https://doi.org/10.1038/s41598-018-24204-6)

## 🎯 Overview

This project provides a complete research-level implementation of an automatic facial acne detection and classification system using deep learning. The system achieves:

- **Binary Classification**: Skin vs non-skin detection (skin area localization)
- **Seven-Class Classification**: Classification of 6 acne types + healthy skin
  - Papule
  - Cyst
  - Blackhead
  - Normal skin
  - Pustule
  - Whitehead
  - Nodule

## 🏗️ Architecture

### Models Implemented

1. **VGG16-Based Classifier** (Pre-trained on ImageNet)
   - Feature extraction using VGG16 backbone
   - Custom classifier head
   - Transfer learning with optional fine-tuning

2. **Custom CNN** (From scratch)
   - Lightweight architecture for binary classification
   - Based on paper's Table 2 specifications

### Key Features

- ✅ Complete data preprocessing and augmentation pipeline
- ✅ Binary classifier for skin detection
- ✅ Seven-class classifier for acne type identification
- ✅ Comprehensive evaluation metrics (ROC, AUC, Confusion Matrix)
- ✅ Sliding window inference for full-face diagnosis
- ✅ Visualization tools for results
- ✅ Batch processing capabilities
- ✅ Model checkpointing and resuming
- ✅ GPU acceleration support

## 📦 Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.7+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/acne-detection.git
cd acne-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
acne-detection/
├── acne_detection_main.py    # Core models and training classes
├── data_utils.py              # Data preparation and loading utilities
├── train.py                   # Complete training script
├── inference.py               # Inference and deployment script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Dataset directory
│   ├── binary/                # Binary classification data
│   │   ├── skin/
│   │   └── non_skin/
│   └── seven_class/           # Seven-class data
│       ├── papule/
│       ├── cyst/
│       ├── blackhead/
│       ├── normal_skin/
│       ├── pustule/
│       ├── whitehead/
│       └── nodule/
│
├── checkpoints/               # Model checkpoints
├── results/                   # Training results and visualizations
└── inference_results/         # Inference outputs
```

## 📊 Dataset Preparation

### Data Organization

Your dataset should be organized as follows:

**For Binary Classification:**
```
data/binary/
├── skin/
│   ├── skin_001.jpg
│   ├── skin_002.jpg
│   └── ...
└── non_skin/
    ├── nonskin_001.jpg
    ├── nonskin_002.jpg
    └── ...
```

**For Seven-Class Classification:**
```
data/seven_class/
├── papule/
├── cyst/
├── blackhead/
├── normal_skin/
├── pustule/
├── whitehead/
└── nodule/
```

### Image Specifications

- **Format**: JPG, JPEG, or PNG
- **Size**: 50×50 pixels (will be automatically resized)
- **Color**: RGB
- **Recommended**: At least 1000 images per class for training

### Data Augmentation

The following augmentations are automatically applied during training:
- Random rotation (±20°)
- Random translation (±10%)
- Random shear (±10°)
- Random zoom (±10%)
- Random horizontal flip
- Normalization (ImageNet statistics)

## 🚀 Usage

### 1. Training

#### Train Binary Classifier (VGG16)

```bash
python train.py \
    --task binary \
    --model vgg16 \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --gpu 0
```

#### Train Binary Classifier (Custom CNN)

```bash
python train.py \
    --task binary \
    --model custom \
    --epochs 50 \
    --batch-size 64 \
    --gpu 0
```

#### Train Seven-Class Classifier

```bash
python train.py \
    --task seven \
    --fine-tune \
    --epochs 50 \
    --batch-size 64 \
    --gpu 0
```

#### Compare Models

```bash
python train.py \
    --task compare \
    --epochs 30 \
    --gpu 0
```

#### Train Everything

```bash
python train.py \
    --task all \
    --fine-tune \
    --epochs 50 \
    --gpu 0
```

### 2. Inference

#### Single Image Diagnosis

```bash
python inference.py \
    --mode single \
    --binary-model checkpoints/binary_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image test_image.jpg \
    --output results/
```

#### Batch Processing

```bash
python inference.py \
    --mode batch \
    --binary-model checkpoints/binary_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image-dir test_images/ \
    --output results/
```

#### Demo Visualization

```bash
python inference.py \
    --mode demo \
    --binary-model checkpoints/binary_vgg16_best_acc.pth \
    --seven-model checkpoints/seven_class_vgg16_best_acc.pth \
    --image demo_image.jpg \
    --output demo/
```

### 3. Python API

```python
from acne_detection_main import Config, VGG16Classifier
from inference import AcneInference

# Create configuration
config = Config()

# Initialize inference
inference = AcneInference(
    binary_model_path='checkpoints/binary_vgg16_best_acc.pth',
    seven_model_path='checkpoints/seven_class_vgg16_best_acc.pth',
    config=config
)

# Diagnose single image
results = inference.diagnose_image(
    image_path='patient_face.jpg',
    window_size=50,
    stride=25,
    save_dir='results/'
)

# Print summary
inference._print_summary(results)
```

## 📈 Training Results

### Binary Classification (Skin Detection)

| Model | AUC | Accuracy | Sensitivity | Specificity |
|-------|-----|----------|-------------|-------------|
| VGG16 | 0.971 | 91.1% | 0.900 | 0.923 |
| Custom CNN | 0.961 | 89.5% | 0.920 | 0.870 |

### Seven-Class Classification

| Class | Accuracy |
|-------|----------|
| Papule | 82.5% |
| Cyst | 83.5% |
| Blackhead | 91.0% |
| Normal Skin | 95.0% |
| Pustule | 81.2% |
| Whitehead | 88.2% |
| Nodule | 86.2% |

**Overall Accuracy**: 86.8%

*Note: Results are based on paper's reported performance. Actual results may vary depending on your dataset.*

## 🔬 Technical Details

### Model Architecture

#### VGG16-Based Classifier

```
Input (50×50×3)
    ↓
VGG16 Feature Extractor (pre-trained)
    ├─ Block 1: Conv2D×2 + MaxPool
    ├─ Block 2: Conv2D×2 + MaxPool
    ├─ Block 3: Conv2D×3 + MaxPool
    ├─ Block 4: Conv2D×3 + MaxPool
    └─ Block 5: Conv2D×3 + MaxPool
    ↓
Global Average Pooling
    ↓
Feature Vector (512-d)
    ↓
Classifier Head
    ├─ Dense (512 → 256) + ReLU + Dropout
    └─ Dense (256 → num_classes)
    ↓
Output (class probabilities)
```

### Training Configuration

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 50
- **Loss Function**: Cross-Entropy
- **Dropout Rate**: 0.5
- **Data Split**: 80% train, 10% validation, 10% test

### Evaluation Metrics

**Binary Classification:**
- ROC Curve and AUC
- Accuracy, Sensitivity, Specificity
- Youden's Index for optimal threshold

**Multi-Class Classification:**
- Confusion Matrix (normalized)
- Per-class Accuracy
- Overall Accuracy

## 🎨 Visualization Examples

The system provides comprehensive visualizations:

1. **Training History**: Loss and accuracy curves
2. **ROC Curves**: For binary classification evaluation
3. **Confusion Matrix**: For seven-class evaluation
4. **Diagnosis Results**: 
   - Original image
   - Skin detection mask
   - Acne type distribution
   - Detailed statistics

## ⚙️ Configuration

Edit `Config` class in `acne_detection_main.py` to customize:

```python
class Config:
    # Data parameters
    IMAGE_SIZE = 50
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Model parameters
    DROPOUT_RATE = 0.5
    
    # Paths
    DATA_ROOT = './data'
    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 32 ...

# Or use CPU
python train.py --gpu -1 ...
```

**2. Dataset Not Found**
```
Ensure your data is organized correctly in the data/ directory
See "Dataset Preparation" section above
```

**3. Model Loading Error**
```
Make sure model checkpoints exist and paths are correct
Check that model architecture matches saved checkpoint
```

## 📚 References

1. Shen, X., Zhang, J., Yan, C., & Zhou, H. (2018). An Automatic Diagnosis Method of Facial Acne Vulgaris Based on Convolutional Neural Network. *Scientific Reports*, 8(1), 5839.

2. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.

3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{shen2018automatic,
  title={An Automatic Diagnosis Method of Facial Acne Vulgaris Based on Convolutional Neural Network},
  author={Shen, Xiaolei and Zhang, Jiachi and Yan, Chenjun and Zhou, Hong},
  journal={Scientific Reports},
  volume={8},
  number={1},
  pages={5839},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original paper authors: Xiaolei Shen, Jiachi Zhang, Chenjun Yan & Hong Zhou
- VGG16 architecture by Visual Geometry Group, Oxford
- PyTorch team for the excellent deep learning framework

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## 🔄 Updates

### Version 1.0.0 (Current)
- Initial release
- Complete implementation of paper's methodology
- Binary and seven-class classification
- Comprehensive evaluation metrics
- Inference pipeline with visualization

---

**Note**: This is a research implementation. For clinical use, please ensure proper validation and regulatory approval.
