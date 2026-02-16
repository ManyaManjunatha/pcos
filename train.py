"""
Complete Training Script
End-to-end training example for acne detection models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from acne_detection_main import (
    Config, AcneAugmentation, VGG16Classifier, CustomCNN,
    Trainer, BinaryEvaluator, MultiClassEvaluator
)
from data_utils import AcneDataLoader
import argparse
import os


def train_binary_model(config, use_vgg16=True, fine_tune=False):
    """
    Complete training pipeline for binary classification
    
    Args:
        config: Configuration object
        use_vgg16: Use VGG16-based model (True) or custom CNN (False)
        fine_tune: Fine-tune VGG16 feature extractor
    """
    
    print("\n" + "=" * 80)
    print("BINARY CLASSIFICATION TRAINING")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    
    # Create data augmentation
    augmentation = AcneAugmentation(config)
    transforms_dict = {
        'train': augmentation.get_train_transform(),
        'val': augmentation.get_val_transform()
    }
    
    # Create data loaders
    print("\nPreparing data loaders...")
    data_loader = AcneDataLoader(
        data_dir=config.BINARY_DATA_PATH,
        transforms_dict=transforms_dict,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    train_loader, val_loader, test_loader = data_loader.create_binary_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Create model
    print("\nCreating model...")
    if use_vgg16:
        model = VGG16Classifier(
            num_classes=config.BINARY_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            use_pretrained=True
        )
        model_name = "binary_vgg16"
    else:
        model = CustomCNN(
            num_classes=config.BINARY_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
        model_name = "binary_custom_cnn"
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2)
    )
    
    # Create trainer
    trainer = Trainer(model, config, model_name)
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.EPOCHS,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Fine-tuning (optional)
    if use_vgg16 and fine_tune:
        print("\n" + "-" * 80)
        print("FINE-TUNING VGG16 FEATURE EXTRACTOR")
        print("-" * 80)
        
        # Unfreeze last layers
        model.unfreeze_feature_extractor(num_layers=4)
        
        # Use SGD for fine-tuning (lower learning rate)
        optimizer_ft = optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        
        # Fine-tune for fewer epochs
        trainer_ft = Trainer(model, config, f"{model_name}_finetuned")
        trainer_ft.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,  # Fewer epochs for fine-tuning
            criterion=criterion,
            optimizer=optimizer_ft
        )
        
        trainer_ft.plot_training_history()
        trainer = trainer_ft
    
    # Evaluate on test set
    print("\n" + "-" * 80)
    print("EVALUATION ON TEST SET")
    print("-" * 80)
    
    # Load best model
    trainer.load_checkpoint('best_acc')
    
    # Create evaluator
    evaluator = BinaryEvaluator(model, config.DEVICE, config)
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)
    evaluator.plot_roc_curve(results, model_name)
    
    return trainer, model, results


def train_seven_class_model(config, fine_tune=False):
    """
    Complete training pipeline for seven-class classification
    
    Args:
        config: Configuration object
        fine_tune: Fine-tune VGG16 feature extractor
    """
    
    print("\n" + "=" * 80)
    print("SEVEN-CLASS CLASSIFICATION TRAINING")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    
    # Create data augmentation
    augmentation = AcneAugmentation(config)
    transforms_dict = {
        'train': augmentation.get_train_transform(),
        'val': augmentation.get_val_transform()
    }
    
    # Create data loaders
    print("\nPreparing data loaders...")
    data_loader = AcneDataLoader(
        data_dir=config.SEVEN_DATA_PATH,
        transforms_dict=transforms_dict,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    train_loader, val_loader, test_loader = data_loader.create_seven_class_loaders(
        class_names=config.CLASS_NAMES,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Create model (always use VGG16 for seven-class as per paper)
    print("\nCreating model...")
    model = VGG16Classifier(
        num_classes=config.SEVEN_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        use_pretrained=True
    )
    model_name = "seven_class_vgg16"
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2)
    )
    
    # Create trainer
    trainer = Trainer(model, config, model_name)
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.EPOCHS,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Fine-tuning (optional)
    if fine_tune:
        print("\n" + "-" * 80)
        print("FINE-TUNING VGG16 FEATURE EXTRACTOR")
        print("-" * 80)
        
        # Unfreeze last layers
        model.unfreeze_feature_extractor(num_layers=4)
        
        # Use SGD for fine-tuning
        optimizer_ft = optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        
        # Fine-tune for fewer epochs
        trainer_ft = Trainer(model, config, f"{model_name}_finetuned")
        trainer_ft.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            criterion=criterion,
            optimizer=optimizer_ft
        )
        
        trainer_ft.plot_training_history()
        trainer = trainer_ft
    
    # Evaluate on test set
    print("\n" + "-" * 80)
    print("EVALUATION ON TEST SET")
    print("-" * 80)
    
    # Load best model
    trainer.load_checkpoint('best_acc')
    
    # Create evaluator
    evaluator = MultiClassEvaluator(
        model, config.DEVICE, config, config.CLASS_NAMES
    )
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)
    evaluator.plot_confusion_matrix(results, model_name)
    
    return trainer, model, results


def compare_binary_models(config):
    """
    Train and compare VGG16-based and custom CNN for binary classification
    """
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: VGG16 vs CUSTOM CNN")
    print("=" * 80)
    
    # Train VGG16 model
    print("\n[1/2] Training VGG16-based model...")
    trainer_vgg, model_vgg, results_vgg = train_binary_model(
        config, use_vgg16=True, fine_tune=False
    )
    
    # Train custom CNN
    print("\n[2/2] Training custom CNN...")
    trainer_custom, model_custom, results_custom = train_binary_model(
        config, use_vgg16=False, fine_tune=False
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print("\nVGG16-based Model:")
    print(f"  AUC:         {results_vgg['auc']:.4f}")
    print(f"  Accuracy:    {results_vgg['accuracy']:.2f}%")
    print(f"  Sensitivity: {results_vgg['sensitivity']:.4f}")
    print(f"  Specificity: {results_vgg['specificity']:.4f}")
    
    print("\nCustom CNN:")
    print(f"  AUC:         {results_custom['auc']:.4f}")
    print(f"  Accuracy:    {results_custom['accuracy']:.2f}%")
    print(f"  Sensitivity: {results_custom['sensitivity']:.4f}")
    print(f"  Specificity: {results_custom['specificity']:.4f}")
    
    # Plot comparison ROC curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.plot(results_vgg['fpr'], results_vgg['tpr'], 
            label=f"VGG16 (AUC = {results_vgg['auc']:.4f})", 
            linewidth=2)
    plt.plot(results_custom['fpr'], results_custom['tpr'],
            label=f"Custom CNN (AUC = {results_custom['auc']:.4f})",
            linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(config.RESULTS_DIR, "model_comparison_roc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison ROC curve saved to {save_path}")


def main():
    """Main training script with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Train Acne Detection Models'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['binary', 'seven', 'compare', 'all'],
        default='binary',
        help='Training task to perform'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['vgg16', 'custom', 'both'],
        default='vgg16',
        help='Model architecture (for binary task)'
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Fine-tune VGG16 feature extractor'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    if args.gpu >= 0 and torch.cuda.is_available():
        config.DEVICE = torch.device(f'cuda:{args.gpu}')
    else:
        config.DEVICE = torch.device('cpu')
    
    print("\n" + "=" * 80)
    print("ACNE DETECTION TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Task:       {args.task}")
    print(f"  Model:      {args.model}")
    print(f"  Fine-tune:  {args.fine_tune}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {config.DEVICE}")
    
    # Execute task
    if args.task == 'binary':
        if args.model == 'vgg16':
            train_binary_model(config, use_vgg16=True, fine_tune=args.fine_tune)
        elif args.model == 'custom':
            train_binary_model(config, use_vgg16=False, fine_tune=False)
        else:  # both
            compare_binary_models(config)
    
    elif args.task == 'seven':
        train_seven_class_model(config, fine_tune=args.fine_tune)
    
    elif args.task == 'compare':
        compare_binary_models(config)
    
    elif args.task == 'all':
        print("\n[Task 1/2] Binary Classification")
        compare_binary_models(config)
        
        print("\n[Task 2/2] Seven-Class Classification")
        train_seven_class_model(config, fine_tune=args.fine_tune)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # If running without command line arguments
    import sys
    if len(sys.argv) == 1:
        print("\nRunning in demonstration mode...")
        print("For full training, use command line arguments.")
        print("\nExample commands:")
        print("  python train.py --task binary --model vgg16 --epochs 50")
        print("  python train.py --task seven --fine-tune --epochs 50")
        print("  python train.py --task compare --epochs 30")
        print("  python train.py --task all --fine-tune --epochs 50")
        print("\nSee --help for all options.")
    else:
        main()
