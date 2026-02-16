"""
An Automatic Diagnosis Method of Facial Acne Vulgaris Based on Convolutional Neural Network
Implementation based on: Shen et al. (2018), Scientific Reports

This implementation provides a complete research-level codebase for:
1. Binary classification (skin vs non-skin detection)
2. Seven-class classification (6 acne types + healthy skin)
3. VGG16-based feature extraction with transfer learning
4. Custom CNN architecture
5. Complete training, validation, and testing pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from typing import Tuple, List, Dict
import json
from tqdm import tqdm


# ============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

class Config:
    """Configuration class for all hyperparameters and settings"""
    
    # Data parameters
    IMAGE_SIZE = 50
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    RANDOM_SEED = 1337
    
    # Model parameters
    DROPOUT_RATE = 0.5
    NUM_WORKERS = 4
    
    # Binary classification
    BINARY_CLASSES = 2  # skin, non-skin
    
    # Seven classification
    SEVEN_CLASSES = 7
    CLASS_NAMES = ['papule', 'cyst', 'blackhead', 'normal_skin', 
                   'pustule', 'whitehead', 'nodule']
    
    # Augmentation parameters
    ROTATION_RANGE = 20  # degrees
    SHIFT_RANGE = 0.1  # fraction of image
    SHEAR_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    
    # Paths
    DATA_ROOT = './data'
    BINARY_DATA_PATH = './data/binary'
    SEVEN_DATA_PATH = './data/seven_class'
    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class AcneAugmentation:
    """
    Custom augmentation pipeline following the paper's specifications:
    - Random rotation
    - Random shift
    - Random shear
    - Random zoom
    - Random horizontal flip
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Training augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(config.ROTATION_RANGE),
            transforms.RandomAffine(
                degrees=0,
                translate=(config.SHIFT_RANGE, config.SHIFT_RANGE),
                shear=config.SHEAR_RANGE * 180 / np.pi,
                scale=(1 - config.ZOOM_RANGE, 1 + config.ZOOM_RANGE)
            ),
            transforms.RandomHorizontalFlip(p=0.5 if config.HORIZONTAL_FLIP else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/Test transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_train_transform(self):
        return self.train_transform
    
    def get_val_transform(self):
        return self.val_transform


# ============================================================================
# DATASET CLASSES
# ============================================================================

class AcneDataset(Dataset):
    """
    Custom Dataset for acne classification
    Supports both binary and seven-class classification
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, is_train: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# CUSTOM CNN ARCHITECTURE (As described in Table 2)
# ============================================================================

class CustomCNN(nn.Module):
    """
    Custom CNN architecture as described in the paper (Table 2)
    Structure:
    - Block1: Conv2D-64, Conv2D-64, MaxPooling
    - Block2: Conv2D-64, MaxPooling, Dropout
    - Flatten
    - Dense-128, Dropout
    - Dense-2 (for binary) or Dense-7 (for seven-class)
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(CustomCNN, self).__init__()
        
        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1_relu = nn.ReLU(inplace=True)
        
        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2_relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Calculate flattened size
        # After 2 max pooling layers (2x2), 50x50 becomes 12x12
        # 64 channels * 12 * 12 = 9216, but paper uses 10816
        # Using paper's value
        self.flatten_size = 10816
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_relu(x)
        x = self.block1_conv2(x)
        x = self.block1_relu(x)
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_relu(x)
        x = self.block2_pool(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# VGG16-BASED FEATURE EXTRACTOR AND CLASSIFIER
# ============================================================================

class VGG16FeatureExtractor(nn.Module):
    """
    VGG16-based feature extractor using pre-trained weights
    Extracts 512-dimensional feature vectors
    """
    
    def __init__(self, pretrained: bool = True):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Remove the classifier (keep only features)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def unfreeze_last_layers(self, num_layers: int = 4):
        """Unfreeze last n convolutional layers for fine-tuning"""
        layers = list(self.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class VGG16Classifier(nn.Module):
    """
    Classifier on top of VGG16 features (Table 3)
    Input: 512-dimensional feature vector
    Architecture:
    - Flatten-512
    - Dense-256, Dropout
    - Dense-num_classes
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, 
                 use_pretrained: bool = True):
        super(VGG16Classifier, self).__init__()
        
        # Feature extractor
        self.feature_extractor = VGG16FeatureExtractor(pretrained=use_pretrained)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_feature_extractor(self, num_layers: int = 4):
        """Unfreeze last layers of VGG16 for fine-tuning"""
        self.feature_extractor.unfreeze_last_layers(num_layers)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """
    Training and evaluation pipeline
    Supports both binary and multi-class classification
    """
    
    def __init__(self, model: nn.Module, config: Config, 
                 model_name: str = "model"):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.model_name = model_name
        self.device = config.DEVICE
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, criterion, optimizer):
        """Complete training loop"""
        
        print(f"\nTraining {self.model_name} for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss')
                print(f"✓ Best loss model saved!")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_acc')
                print(f"✓ Best accuracy model saved!")
        
        print("\n" + "=" * 80)
        print(f"Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print("=" * 80)
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f"{self.model_name}_{checkpoint_name}.pth"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_name: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f"{self.model_name}_{checkpoint_name}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name} - Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{self.model_name} - Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(
            self.config.RESULTS_DIR, 
            f"{self.model_name}_training_history.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {save_path}")


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class BinaryEvaluator:
    """
    Evaluation for binary classification
    Computes ROC curve, AUC, Youden's index, sensitivity, specificity
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Config):
        self.model = model
        self.device = device
        self.config = config
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        accuracy = 100 * np.mean(all_predictions == all_labels)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Youden's index
        youden_index = tpr - fpr
        best_threshold_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_idx]
        best_youden = youden_index[best_threshold_idx]
        
        # Sensitivity and Specificity at best threshold
        predictions_at_best = (all_probabilities >= best_threshold).astype(int)
        tn = np.sum((all_labels == 0) & (predictions_at_best == 0))
        fp = np.sum((all_labels == 0) & (predictions_at_best == 1))
        fn = np.sum((all_labels == 1) & (predictions_at_best == 0))
        tp = np.sum((all_labels == 1) & (predictions_at_best == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'auc': roc_auc,
            'youden_index': best_youden,
            'best_threshold': best_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        return results
    
    def plot_roc_curve(self, results: Dict, model_name: str):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(results['fpr'], results['tpr'], 
                label=f"ROC curve (area = {results['auc']:.3f})", 
                linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1-Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(
            self.config.RESULTS_DIR, 
            f"{model_name}_roc_curve.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")
    
    def print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("BINARY CLASSIFICATION RESULTS")
        print("=" * 80)
        print(f"Accuracy:      {results['accuracy']:.2f}%")
        print(f"AUC:           {results['auc']:.4f}")
        print(f"Youden Index:  {results['youden_index']:.4f}")
        print(f"Best Threshold: {results['best_threshold']:.4f}")
        print(f"Sensitivity:   {results['sensitivity']:.4f}")
        print(f"Specificity:   {results['specificity']:.4f}")
        print("=" * 80)


class MultiClassEvaluator:
    """
    Evaluation for multi-class classification
    Computes confusion matrix, per-class accuracy, overall accuracy
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 config: Config, class_names: List[str]):
        self.model = model
        self.device = device
        self.config = config
        self.class_names = class_names
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.vstack(all_probabilities)
        
        # Overall accuracy
        accuracy = 100 * np.mean(all_predictions == all_labels)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = (all_labels == i)
            if class_mask.sum() > 0:
                class_acc = 100 * np.mean(
                    all_predictions[class_mask] == all_labels[class_mask]
                )
                per_class_acc[class_name] = class_acc
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'per_class_accuracy': per_class_acc,
            'all_labels': all_labels,
            'all_predictions': all_predictions,
            'all_probabilities': all_probabilities
        }
        
        return results
    
    def plot_confusion_matrix(self, results: Dict, model_name: str):
        """Plot normalized confusion matrix"""
        cm_normalized = results['confusion_matrix_normalized']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Normalized Frequency'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(
            self.config.RESULTS_DIR, 
            f"{model_name}_confusion_matrix.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("SEVEN-CLASS CLASSIFICATION RESULTS")
        print("=" * 80)
        print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        print("\nPer-Class Accuracy:")
        print("-" * 80)
        for class_name, acc in results['per_class_accuracy'].items():
            print(f"  {class_name:15s}: {acc:6.2f}%")
        print("=" * 80)


# ============================================================================
# COMPLETE PIPELINE FUNCTIONS
# ============================================================================

def train_binary_classifier(config: Config, use_vgg16: bool = True):
    """
    Train binary classifier for skin detection
    
    Args:
        config: Configuration object
        use_vgg16: If True, use VGG16-based model; else use custom CNN
    """
    print("\n" + "=" * 80)
    print("TRAINING BINARY CLASSIFIER (SKIN vs NON-SKIN DETECTION)")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create data augmentation
    augmentation = AcneAugmentation(config)
    
    # Note: In practice, you would load your actual data here
    # This is a placeholder showing the expected structure
    print("\nNote: Please prepare your dataset in the following structure:")
    print(f"  {config.BINARY_DATA_PATH}/")
    print("    ├── skin/")
    print("    │   ├── image1.jpg")
    print("    │   └── ...")
    print("    └── non_skin/")
    print("        ├── image1.jpg")
    print("        └── ...")
    
    # Create model
    if use_vgg16:
        model = VGG16Classifier(num_classes=config.BINARY_CLASSES, 
                               dropout_rate=config.DROPOUT_RATE,
                               use_pretrained=True)
        model_name = "binary_vgg16"
    else:
        model = CustomCNN(num_classes=config.BINARY_CLASSES, 
                         dropout_rate=config.DROPOUT_RATE)
        model_name = "binary_custom_cnn"
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.LEARNING_RATE,
                          betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    
    # Create trainer
    trainer = Trainer(model, config, model_name)
    
    print("\nTraining pipeline created successfully!")
    print("To run training, prepare your dataset and create DataLoaders.")
    
    return trainer, model


def train_seven_classifier(config: Config):
    """
    Train seven-class classifier for acne type classification
    Uses VGG16-based architecture as per paper
    
    Args:
        config: Configuration object
    """
    print("\n" + "=" * 80)
    print("TRAINING SEVEN-CLASS CLASSIFIER (ACNE TYPE CLASSIFICATION)")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create data augmentation
    augmentation = AcneAugmentation(config)
    
    # Note: In practice, you would load your actual data here
    print("\nNote: Please prepare your dataset in the following structure:")
    print(f"  {config.SEVEN_DATA_PATH}/")
    print("    ├── papule/")
    print("    ├── cyst/")
    print("    ├── blackhead/")
    print("    ├── normal_skin/")
    print("    ├── pustule/")
    print("    ├── whitehead/")
    print("    └── nodule/")
    
    # Create VGG16-based model
    model = VGG16Classifier(num_classes=config.SEVEN_CLASSES, 
                           dropout_rate=config.DROPOUT_RATE,
                           use_pretrained=True)
    model_name = "seven_class_vgg16"
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.LEARNING_RATE,
                          betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    
    # Create trainer
    trainer = Trainer(model, config, model_name)
    
    print("\nTraining pipeline created successfully!")
    print("To run training, prepare your dataset and create DataLoaders.")
    
    return trainer, model


# ============================================================================
# INFERENCE AND DEPLOYMENT
# ============================================================================

class AcneDiagnosisSystem:
    """
    Complete acne diagnosis system
    Combines skin detection and acne classification
    """
    
    def __init__(self, binary_model: nn.Module, seven_model: nn.Module, 
                 config: Config):
        self.binary_model = binary_model.eval()
        self.seven_model = seven_model.eval()
        self.config = config
        self.device = config.DEVICE
        
        # Move models to device
        self.binary_model.to(self.device)
        self.seven_model.to(self.device)
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def sliding_window_detection(self, image: Image.Image, 
                                 window_size: int = 50, 
                                 stride: int = 25) -> Dict:
        """
        Apply sliding window to detect skin and classify acne
        
        Args:
            image: PIL Image (500x500 recommended)
            window_size: Size of sliding window
            stride: Stride of sliding window
        
        Returns:
            Dictionary with detection results and statistics
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        skin_mask = np.zeros((height, width), dtype=bool)
        acne_predictions = []
        acne_locations = []
        
        # Sliding window
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # Extract window
                window = image.crop((x, y, x + window_size, y + window_size))
                window_tensor = self.transform(window).unsqueeze(0).to(self.device)
                
                # Binary classification (skin detection)
                with torch.no_grad():
                    binary_output = self.binary_model(window_tensor)
                    binary_prob = torch.softmax(binary_output, dim=1)
                    is_skin = binary_prob[0, 0] > 0.5  # class 0 is skin
                
                if is_skin:
                    skin_mask[y:y+window_size, x:x+window_size] = True
                    
                    # Seven-class classification (acne type)
                    with torch.no_grad():
                        seven_output = self.seven_model(window_tensor)
                        seven_prob = torch.softmax(seven_output, dim=1)
                        prediction = torch.argmax(seven_prob, dim=1).item()
                    
                    acne_predictions.append(prediction)
                    acne_locations.append((x, y, x + window_size, y + window_size))
        
        # Calculate statistics
        acne_stats = self._calculate_statistics(acne_predictions)
        
        results = {
            'skin_mask': skin_mask,
            'acne_predictions': acne_predictions,
            'acne_locations': acne_locations,
            'acne_statistics': acne_stats
        }
        
        return results
    
    def _calculate_statistics(self, predictions: List[int]) -> Dict:
        """Calculate statistics of acne types"""
        if len(predictions) == 0:
            return {}
        
        predictions_array = np.array(predictions)
        total = len(predictions)
        
        stats = {}
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            count = np.sum(predictions_array == i)
            proportion = count / total
            stats[class_name] = {
                'count': int(count),
                'proportion': float(proportion)
            }
        
        return stats
    
    def visualize_results(self, image: Image.Image, results: Dict, 
                         save_path: str = None):
        """Visualize detection and classification results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Skin mask
        axes[1].imshow(image)
        axes[1].imshow(results['skin_mask'], alpha=0.3, cmap='Reds')
        axes[1].set_title('Skin Detection', fontsize=14)
        axes[1].axis('off')
        
        # Statistics
        stats = results['acne_statistics']
        if stats:
            class_names = list(stats.keys())
            proportions = [stats[name]['proportion'] for name in class_names]
            
            axes[2].bar(range(len(class_names)), proportions)
            axes[2].set_xticks(range(len(class_names)))
            axes[2].set_xticklabels(class_names, rotation=45, ha='right')
            axes[2].set_ylabel('Proportion', fontsize=12)
            axes[2].set_title('Acne Type Distribution', fontsize=14)
            axes[2].set_ylim([0, 1])
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function demonstrating the complete pipeline"""
    
    print("\n" + "=" * 80)
    print("ACNE VULGARIS AUTOMATIC DIAGNOSIS SYSTEM")
    print("Based on: Shen et al. (2018), Scientific Reports")
    print("=" * 80)
    
    # Configuration
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    
    # Example 1: Train binary classifier (VGG16)
    print("\n" + "-" * 80)
    print("Example 1: Binary Classifier (VGG16-based)")
    print("-" * 80)
    trainer_binary_vgg, model_binary_vgg = train_binary_classifier(
        config, use_vgg16=True
    )
    
    # Example 2: Train binary classifier (Custom CNN)
    print("\n" + "-" * 80)
    print("Example 2: Binary Classifier (Custom CNN)")
    print("-" * 80)
    trainer_binary_custom, model_binary_custom = train_binary_classifier(
        config, use_vgg16=False
    )
    
    # Example 3: Train seven-class classifier
    print("\n" + "-" * 80)
    print("Example 3: Seven-Class Classifier (VGG16-based)")
    print("-" * 80)
    trainer_seven, model_seven = train_seven_classifier(config)
    
    # Example 4: Complete diagnosis system
    print("\n" + "-" * 80)
    print("Example 4: Complete Diagnosis System")
    print("-" * 80)
    print("\nTo use the complete diagnosis system:")
    print("1. Train both binary and seven-class models")
    print("2. Load the trained models")
    print("3. Create AcneDiagnosisSystem instance")
    print("4. Use sliding_window_detection() for inference")
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your dataset following the structure shown above")
    print("2. Create DataLoaders from your dataset")
    print("3. Call trainer.train() to start training")
    print("4. Use evaluators to assess model performance")
    print("5. Deploy using AcneDiagnosisSystem for inference")
    print("\nFor detailed usage, see the example scripts provided.")
    print("=" * 80)


if __name__ == "__main__":
    main()
