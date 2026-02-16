"""
Inference Script for Acne Detection
Deploy trained models for automatic diagnosis
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple

from acne_detection_main import (
    Config, VGG16Classifier, CustomCNN, AcneDiagnosisSystem
)
import torchvision.transforms as transforms


class AcneInference:
    """
    Inference wrapper for easy deployment
    """
    
    def __init__(self, binary_model_path: str, seven_model_path: str, 
                 config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # Load models
        print("Loading models...")
        self.binary_model = self._load_binary_model(binary_model_path)
        self.seven_model = self._load_seven_model(seven_model_path)
        
        # Create diagnosis system
        self.diagnosis_system = AcneDiagnosisSystem(
            binary_model=self.binary_model,
            seven_model=self.seven_model,
            config=config
        )
        
        print("Models loaded successfully!")
    
    def _load_binary_model(self, model_path: str) -> nn.Module:
        """Load binary classification model"""
        # Create model architecture
        model = VGG16Classifier(
            num_classes=self.config.BINARY_CLASSES,
            dropout_rate=self.config.DROPOUT_RATE,
            use_pretrained=False  # Load our trained weights
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_seven_model(self, model_path: str) -> nn.Module:
        """Load seven-class classification model"""
        # Create model architecture
        model = VGG16Classifier(
            num_classes=self.config.SEVEN_CLASSES,
            dropout_rate=self.config.DROPOUT_RATE,
            use_pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def diagnose_image(self, image_path: str, window_size: int = 50,
                      stride: int = 25, save_dir: str = None) -> Dict:
        """
        Perform complete diagnosis on a single image
        
        Args:
            image_path: Path to input image
            window_size: Size of sliding window
            stride: Stride for sliding window
            save_dir: Directory to save results
        
        Returns:
            Dictionary with diagnosis results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Run diagnosis
        print(f"Analyzing {image_path}...")
        results = self.diagnosis_system.sliding_window_detection(
            image=image,
            window_size=window_size,
            stride=stride
        )
        
        # Add image info
        results['image_path'] = image_path
        results['image_size'] = image.size
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            vis_path = save_path / f"{Path(image_path).stem}_results.png"
            self.diagnosis_system.visualize_results(
                image=image,
                results=results,
                save_path=str(vis_path)
            )
            
            # Save JSON report
            json_path = save_path / f"{Path(image_path).stem}_report.json"
            self._save_report(results, str(json_path))
        
        return results
    
    def diagnose_batch(self, image_dir: str, output_dir: str,
                      window_size: int = 50, stride: int = 25):
        """
        Diagnose all images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.jpeg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        all_results = []
        for img_path in image_paths:
            print(f"\n{'='*60}")
            results = self.diagnose_image(
                image_path=str(img_path),
                window_size=window_size,
                stride=stride,
                save_dir=output_dir
            )
            all_results.append(results)
            self._print_summary(results)
        
        # Save batch summary
        summary_path = Path(output_dir) / "batch_summary.json"
        self._save_batch_summary(all_results, str(summary_path))
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Results saved to {output_dir}")
    
    def _save_report(self, results: Dict, save_path: str):
        """Save diagnosis report as JSON"""
        # Prepare JSON-serializable report
        report = {
            'image_path': results['image_path'],
            'image_size': results['image_size'],
            'acne_statistics': results['acne_statistics']
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {save_path}")
    
    def _save_batch_summary(self, all_results: List[Dict], save_path: str):
        """Save summary of batch processing"""
        summary = {
            'total_images': len(all_results),
            'images': []
        }
        
        for results in all_results:
            image_summary = {
                'image_path': results['image_path'],
                'statistics': results['acne_statistics']
            }
            summary['images'].append(image_summary)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch summary saved to {save_path}")
    
    def _print_summary(self, results: Dict):
        """Print diagnosis summary"""
        print("\nDiagnosis Summary:")
        print("-" * 60)
        
        stats = results['acne_statistics']
        if not stats:
            print("  No acne detected")
            return
        
        # Sort by proportion
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1]['proportion'],
            reverse=True
        )
        
        for class_name, info in sorted_stats:
            if info['proportion'] > 0.01:  # Show if >1%
                print(f"  {class_name:15s}: {info['proportion']*100:5.2f}%  "
                     f"({info['count']} regions)")


class SingleImageInference:
    """
    Lightweight inference for single image classification
    (without sliding window)
    """
    
    def __init__(self, model_path: str, num_classes: int, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_path, num_classes)
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str, num_classes: int) -> nn.Module:
        """Load model from checkpoint"""
        model = VGG16Classifier(
            num_classes=num_classes,
            dropout_rate=self.config.DROPOUT_RATE,
            use_pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_path: str) -> Tuple[int, np.ndarray]:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image
        
        Returns:
            predicted_class: Predicted class index
            probabilities: Class probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def predict_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            predictions: Array of predicted classes
            probabilities: Array of class probabilities
        """
        predictions = []
        all_probs = []
        
        for img_path in image_paths:
            pred, probs = self.predict(img_path)
            predictions.append(pred)
            all_probs.append(probs)
        
        return np.array(predictions), np.array(all_probs)


def create_demo_visualization(inference: AcneInference, image_path: str,
                              output_path: str):
    """
    Create comprehensive visualization for demo/publication
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Run diagnosis
    results = inference.diagnosis_system.sliding_window_detection(
        image=image,
        window_size=50,
        stride=25
    )
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(image)
    ax1.set_title('Original Facial Image', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Skin detection mask
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.imshow(image)
    ax2.imshow(results['skin_mask'], alpha=0.3, cmap='Reds')
    ax2.set_title('Skin Detection Result', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Acne type distribution
    ax3 = fig.add_subplot(gs[2, :2])
    stats = results['acne_statistics']
    if stats:
        class_names = list(stats.keys())
        proportions = [stats[name]['proportion'] for name in class_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        bars = ax3.bar(range(len(class_names)), proportions, color=colors)
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
        ax3.set_ylabel('Proportion', fontsize=14)
        ax3.set_title('Acne Type Distribution', fontsize=16, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, prop in zip(bars, proportions):
            if prop > 0.01:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prop*100:.1f}%',
                        ha='center', va='bottom', fontsize=10)
    
    # Detailed statistics table
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')
    
    if stats:
        # Create table data
        table_data = [['Acne Type', 'Proportion', 'Count']]
        sorted_stats = sorted(stats.items(), 
                            key=lambda x: x[1]['proportion'],
                            reverse=True)
        
        for class_name, info in sorted_stats:
            if info['proportion'] > 0.001:
                table_data.append([
                    class_name,
                    f"{info['proportion']*100:.2f}%",
                    str(info['count'])
                ])
        
        table = ax4.table(cellText=table_data, 
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.5, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Detailed Analysis', fontsize=16, 
                     fontweight='bold', pad=20)
    
    plt.suptitle('Automatic Facial Acne Vulgaris Diagnosis System',
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Demo visualization saved to {output_path}")


def main():
    """Main inference script"""
    
    parser = argparse.ArgumentParser(
        description='Acne Detection Inference'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch', 'demo'],
        default='single',
        help='Inference mode'
    )
    parser.add_argument(
        '--binary-model',
        type=str,
        required=True,
        help='Path to binary classification model'
    )
    parser.add_argument(
        '--seven-model',
        type=str,
        required=True,
        help='Path to seven-class model'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Input image path (for single mode)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Input image directory (for batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./inference_results',
        help='Output directory'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=50,
        help='Sliding window size'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=25,
        help='Sliding window stride'
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
    if args.gpu >= 0 and torch.cuda.is_available():
        config.DEVICE = torch.device(f'cuda:{args.gpu}')
    else:
        config.DEVICE = torch.device('cpu')
    
    print("\n" + "=" * 80)
    print("ACNE DETECTION INFERENCE")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.DEVICE}")
    
    # Create inference object
    inference = AcneInference(
        binary_model_path=args.binary_model,
        seven_model_path=args.seven_model,
        config=config
    )
    
    # Run inference
    if args.mode == 'single':
        if not args.image:
            raise ValueError("--image required for single mode")
        
        results = inference.diagnose_image(
            image_path=args.image,
            window_size=args.window_size,
            stride=args.stride,
            save_dir=args.output
        )
        inference._print_summary(results)
    
    elif args.mode == 'batch':
        if not args.image_dir:
            raise ValueError("--image-dir required for batch mode")
        
        inference.diagnose_batch(
            image_dir=args.image_dir,
            output_dir=args.output,
            window_size=args.window_size,
            stride=args.stride
        )
    
    elif args.mode == 'demo':
        if not args.image:
            raise ValueError("--image required for demo mode")
        
        output_path = Path(args.output) / "demo_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_demo_visualization(
            inference=inference,
            image_path=args.image,
            output_path=str(output_path)
        )
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("\nUsage examples:")
        print("\n1. Single image diagnosis:")
        print("   python inference.py --mode single \\")
        print("       --binary-model checkpoints/binary_vgg16_best_acc.pth \\")
        print("       --seven-model checkpoints/seven_class_vgg16_best_acc.pth \\")
        print("       --image test_image.jpg \\")
        print("       --output results/")
        print("\n2. Batch diagnosis:")
        print("   python inference.py --mode batch \\")
        print("       --binary-model checkpoints/binary_vgg16_best_acc.pth \\")
        print("       --seven-model checkpoints/seven_class_vgg16_best_acc.pth \\")
        print("       --image-dir test_images/ \\")
        print("       --output results/")
        print("\n3. Demo visualization:")
        print("   python inference.py --mode demo \\")
        print("       --binary-model checkpoints/binary_vgg16_best_acc.pth \\")
        print("       --seven-model checkpoints/seven_class_vgg16_best_acc.pth \\")
        print("       --image demo_image.jpg \\")
        print("       --output demo/")
    else:
        main()
