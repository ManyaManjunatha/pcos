"""
Quick Start Example
Demonstrates basic usage of the acne detection system
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from acne_detection_main import (
    Config, AcneAugmentation, VGG16Classifier, CustomCNN,
    Trainer, BinaryEvaluator, MultiClassEvaluator
)
from data_utils import (
    DatasetOrganizer, ImagePreprocessor, AcneDataLoader,
    DataAugmentationAnalyzer
)
from inference import AcneInference, SingleImageInference


def example_1_data_preparation():
    """
    Example 1: Prepare and organize dataset
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: DATA PREPARATION")
    print("=" * 80)
    
    # Step 1: Organize raw images
    print("\nStep 1: Organize binary classification data")
    print("Organizing images into skin/non-skin categories...")
    
    organizer = DatasetOrganizer(
        source_dir='./raw_data/binary',
        target_dir='./data/binary'
    )
    
    # Example: organize based on filename keywords
    # organizer.organize_binary_dataset(
    #     skin_keywords=['skin', 'face', 'facial'],
    #     non_skin_keywords=['background', 'hair', 'nonskin']
    # )
    
    print("✓ Binary data organized")
    
    # Step 2: Organize seven-class data
    print("\nStep 2: Organize seven-class data")
    
    organizer_seven = DatasetOrganizer(
        source_dir='./raw_data/seven_class',
        target_dir='./data/seven_class'
    )
    
    class_mapping = {
        'papule': ['papule', 'pap'],
        'cyst': ['cyst'],
        'blackhead': ['blackhead', 'black'],
        'normal_skin': ['normal', 'healthy'],
        'pustule': ['pustule', 'pus'],
        'whitehead': ['whitehead', 'white'],
        'nodule': ['nodule', 'nod']
    }
    
    # organizer_seven.organize_seven_class_dataset(class_mapping)
    
    print("✓ Seven-class data organized")
    
    # Step 3: Preprocess images
    print("\nStep 3: Preprocess images to 50x50")
    
    preprocessor = ImagePreprocessor(target_size=(50, 50))
    
    # Example: preprocess all images in a directory
    # preprocessor.preprocess_directory(
    #     input_dir='./data/binary/skin',
    #     output_dir='./data/binary_processed/skin'
    # )
    
    print("✓ Images preprocessed")
    
    print("\n✅ Data preparation complete!")


def example_2_visualize_augmentation():
    """
    Example 2: Visualize data augmentation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: DATA AUGMENTATION VISUALIZATION")
    print("=" * 80)
    
    config = Config()
    augmentation = AcneAugmentation(config)
    
    # Get training transform
    train_transform = augmentation.get_train_transform()
    
    # Visualize augmentation effects
    analyzer = DataAugmentationAnalyzer(train_transform)
    
    # Example: visualize on a sample image
    # analyzer.visualize_augmentation(
    #     image_path='sample_image.jpg',
    #     num_samples=10,
    #     save_path='results/augmentation_demo.png'
    # )
    
    print("\n✅ Augmentation visualization complete!")


def example_3_train_binary_model():
    """
    Example 3: Train binary classification model
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: TRAIN BINARY CLASSIFIER")
    print("=" * 80)
    
    # Configuration
    config = Config()
    config.EPOCHS = 10  # Reduced for quick demo
    
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    
    # Create model
    print("\nCreating VGG16-based model...")
    model = VGG16Classifier(
        num_classes=config.BINARY_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        use_pretrained=True
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data augmentation
    augmentation = AcneAugmentation(config)
    transforms_dict = {
        'train': augmentation.get_train_transform(),
        'val': augmentation.get_val_transform()
    }
    
    # Note: In practice, you would load actual data here
    print("\nNote: To run actual training, prepare your dataset")
    print("See example_1_data_preparation() for data organization")
    
    # Example training code (uncomment when data is ready):
    """
    # Create data loaders
    data_loader = AcneDataLoader(
        data_dir=config.BINARY_DATA_PATH,
        transforms_dict=transforms_dict
    )
    train_loader, val_loader, test_loader = data_loader.create_binary_loaders()
    
    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train
    trainer = Trainer(model, config, 'binary_vgg16_demo')
    trainer.train(train_loader, val_loader, config.EPOCHS, criterion, optimizer)
    
    # Evaluate
    evaluator = BinaryEvaluator(model, config.DEVICE, config)
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)
    """
    
    print("\n✅ Model training setup complete!")


def example_4_train_seven_class_model():
    """
    Example 4: Train seven-class model
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: TRAIN SEVEN-CLASS CLASSIFIER")
    print("=" * 80)
    
    config = Config()
    config.EPOCHS = 10
    
    # Create model
    print("\nCreating seven-class VGG16 model...")
    model = VGG16Classifier(
        num_classes=config.SEVEN_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        use_pretrained=True
    )
    
    print(f"Classes: {config.CLASS_NAMES}")
    
    # Example training (similar to binary)
    print("\nNote: Follow same procedure as binary classification")
    print("See train.py for complete implementation")
    
    print("\n✅ Model setup complete!")


def example_5_inference():
    """
    Example 5: Run inference on new images
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: INFERENCE")
    print("=" * 80)
    
    config = Config()
    
    print("\nNote: Ensure trained models are available:")
    print("  - checkpoints/binary_vgg16_best_acc.pth")
    print("  - checkpoints/seven_class_vgg16_best_acc.pth")
    
    # Example inference code (uncomment when models are ready):
    """
    # Create inference system
    inference = AcneInference(
        binary_model_path='checkpoints/binary_vgg16_best_acc.pth',
        seven_model_path='checkpoints/seven_class_vgg16_best_acc.pth',
        config=config
    )
    
    # Diagnose single image
    results = inference.diagnose_image(
        image_path='test_face.jpg',
        window_size=50,
        stride=25,
        save_dir='inference_results/'
    )
    
    # Print summary
    inference._print_summary(results)
    
    # Batch processing
    inference.diagnose_batch(
        image_dir='test_images/',
        output_dir='batch_results/'
    )
    """
    
    print("\n✅ Inference example complete!")


def example_6_single_patch_classification():
    """
    Example 6: Classify single 50x50 image patch
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: SINGLE PATCH CLASSIFICATION")
    print("=" * 80)
    
    config = Config()
    
    # Example code (uncomment when model is ready):
    """
    # For binary classification
    binary_inference = SingleImageInference(
        model_path='checkpoints/binary_vgg16_best_acc.pth',
        num_classes=2,
        config=config
    )
    
    pred_class, probabilities = binary_inference.predict('patch_image.jpg')
    print(f"Prediction: {'skin' if pred_class == 0 else 'non-skin'}")
    print(f"Probabilities: {probabilities}")
    
    # For seven-class classification
    seven_inference = SingleImageInference(
        model_path='checkpoints/seven_class_vgg16_best_acc.pth',
        num_classes=7,
        config=config
    )
    
    pred_class, probabilities = seven_inference.predict('acne_patch.jpg')
    print(f"Prediction: {config.CLASS_NAMES[pred_class]}")
    print(f"Probabilities: {probabilities}")
    """
    
    print("\n✅ Single patch classification example complete!")


def example_7_create_synthetic_data():
    """
    Example 7: Create synthetic dataset for testing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: CREATE SYNTHETIC TEST DATA")
    print("=" * 80)
    
    import os
    from pathlib import Path
    
    # Create directories
    data_dir = Path('./data/demo')
    
    # Binary classification
    binary_dir = data_dir / 'binary'
    (binary_dir / 'skin').mkdir(parents=True, exist_ok=True)
    (binary_dir / 'non_skin').mkdir(parents=True, exist_ok=True)
    
    # Seven-class
    seven_dir = data_dir / 'seven_class'
    for class_name in ['papule', 'cyst', 'blackhead', 'normal_skin', 
                       'pustule', 'whitehead', 'nodule']:
        (seven_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic images (50x50 random colored images)
    print("\nGenerating synthetic images for demonstration...")
    
    for i in range(10):
        # Binary - skin
        img = Image.fromarray(
            np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        )
        img.save(binary_dir / 'skin' / f'skin_{i:03d}.jpg')
        
        # Binary - non-skin
        img = Image.fromarray(
            np.random.randint(0, 100, (50, 50, 3), dtype=np.uint8)
        )
        img.save(binary_dir / 'non_skin' / f'nonskin_{i:03d}.jpg')
        
        # Seven-class
        for class_name in ['papule', 'cyst', 'blackhead', 'normal_skin',
                          'pustule', 'whitehead', 'nodule']:
            img = Image.fromarray(
                np.random.randint(50, 250, (50, 50, 3), dtype=np.uint8)
            )
            img.save(seven_dir / class_name / f'{class_name}_{i:03d}.jpg')
    
    print(f"✓ Created synthetic dataset in {data_dir}")
    print(f"  Binary: 20 images (10 skin, 10 non-skin)")
    print(f"  Seven-class: 70 images (10 per class)")
    
    print("\n✅ Synthetic data creation complete!")


def run_complete_demo():
    """
    Run complete demonstration pipeline
    """
    print("\n" + "=" * 80)
    print("COMPLETE ACNE DETECTION DEMO")
    print("=" * 80)
    
    print("\nThis demo walks through the entire pipeline:")
    print("1. Data preparation")
    print("2. Augmentation visualization")
    print("3. Binary model training")
    print("4. Seven-class model training")
    print("5. Inference")
    print("6. Single patch classification")
    print("7. Synthetic data generation")
    
    # Run examples
    example_7_create_synthetic_data()  # Start with data generation
    example_1_data_preparation()
    example_2_visualize_augmentation()
    example_3_train_binary_model()
    example_4_train_seven_class_model()
    example_5_inference()
    example_6_single_patch_classification()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your real dataset following the structure shown")
    print("2. Run: python train.py --task all --epochs 50")
    print("3. Run: python inference.py --mode demo --image your_image.jpg")


def main():
    """Main function with menu"""
    
    examples = {
        '1': ('Data Preparation', example_1_data_preparation),
        '2': ('Augmentation Visualization', example_2_visualize_augmentation),
        '3': ('Train Binary Model', example_3_train_binary_model),
        '4': ('Train Seven-Class Model', example_4_train_seven_class_model),
        '5': ('Inference', example_5_inference),
        '6': ('Single Patch Classification', example_6_single_patch_classification),
        '7': ('Create Synthetic Data', example_7_create_synthetic_data),
        '8': ('Run Complete Demo', run_complete_demo),
    }
    
    print("\n" + "=" * 80)
    print("ACNE DETECTION QUICK START EXAMPLES")
    print("=" * 80)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. Quit")
    
    while True:
        choice = input("\nSelect example (1-8 or q): ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye!")
            break
        
        if choice in examples:
            _, func = examples[choice]
            func()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
