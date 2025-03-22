#!/usr/bin/env python3
"""
evaluate_model.py - Script for evaluating YOLOv5 models on road object detection dataset
with special focus on cyclist detection performance.

This script handles:
1. Loading a trained model
2. Running evaluation on test data
3. Generating detailed metrics for each class, with focus on cyclists
4. Creating visualizations of model performance
5. Generating confusion matrices and precision-recall curves
"""

import argparse
import os
import sys
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("evaluate_model")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 model on road object detection dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv5 model (.pt file)')
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to dataset configuration YAML')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for evaluation')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--cyclist-focus', action='store_true',
                        help='Generate additional metrics and visualizations for cyclist class')
    
    return parser.parse_args()

def run_evaluation(args):
    """Run model evaluation with YOLOv5 val.py script."""
    logger.info(f"Evaluating model {args.model} with data config {args.data_config}")
    
    try:
        # Import YOLOv5 modules (assuming YOLOv5 is installed or in PYTHONPATH)
        sys.path.append('/opt/yolov5')  # Adjust path if needed
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Construct command for validation
        cmd = [
            "python", "/opt/yolov5/val.py",
            "--data", args.data_config,
            "--weights", args.model,
            "--img", str(args.img_size),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--conf-thres", str(args.conf_thres),
            "--iou-thres", str(args.iou_thres),
            "--task", "test",
            "--save-txt",
            "--save-conf",
            "--save-json",
            "--verbose"
        ]
        
        # Execute validation command
        logger.info(f"Executing validation command: {' '.join(cmd)}")
        
        # Use subprocess to run the command
        import subprocess
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Capture output for parsing
        output = ""
        for line in process.stdout:
            logger.info(line.strip())
            output += line
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Evaluation failed with return code {process.returncode}")
            return None
        
        # Parse results
        results = parse_evaluation_results(output)
        
        # Save results to JSON
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        return results
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return None

def parse_evaluation_results(output):
    """Parse YOLOv5 evaluation output to extract metrics."""
    results = {
        'overall': {},
        'classes': {}
    }
    
    # Extract overall mAP
    import re
    map_match = re.search(r'all\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)', output)
    if map_match:
        results['overall']['mAP@.5'] = float(map_match.group(3))
        results['overall']['mAP@.5:.95'] = float(map_match.group(4))
    
    # Extract class-specific metrics
    class_pattern = r'(\w+)\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)'
    class_matches = re.finditer(class_pattern, output)
    
    for match in class_matches:
        class_name = match.group(1)
        if class_name != 'all':  # Skip the overall metrics
            results['classes'][class_name] = {
                'precision': float(match.group(3)),
                'recall': float(match.group(4)),
                'mAP@.5': float(match.group(5))
            }
    
    return results

def generate_visualizations(args, results):
    """Generate visualizations of model performance."""
    logger.info("Generating performance visualizations")
    
    try:
        # Create output directory for visualizations
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot class mAP comparison
        if 'classes' in results and results['classes']:
            class_names = list(results['classes'].keys())
            map_values = [results['classes'][cls]['mAP@.5'] for cls in class_names]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(class_names, map_values, color='skyblue')
            
            # Highlight cyclist class if present
            if args.cyclist_focus:
                for i, cls in enumerate(class_names):
                    if cls.lower() == 'cyclist':
                        bars[i].set_color('red')
            
            plt.xlabel('Class')
            plt.ylabel('mAP@0.5')
            plt.title('Class-wise mAP@0.5')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', rotation=0)
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, 'class_map_comparison.png'), dpi=300)
            plt.close()
        
        # Plot precision-recall comparison
        if 'classes' in results and results['classes']:
            plt.figure(figsize=(10, 8))
            
            for cls_name, metrics in results['classes'].items():
                if args.cyclist_focus and cls_name.lower() == 'cyclist':
                    # Highlight cyclist class
                    plt.scatter(metrics['recall'], metrics['precision'], color='red', s=100, label=f"{cls_name} (focus)")
                else:
                    plt.scatter(metrics['recall'], metrics['precision'], label=cls_name)
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall by Class')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, 'precision_recall_comparison.png'), dpi=300)
            plt.close()
        
        # Generate confusion matrix if available
        confusion_matrix_path = find_confusion_matrix(args.model)
        if confusion_matrix_path and os.path.exists(confusion_matrix_path):
            import shutil
            dest_path = os.path.join(viz_dir, 'confusion_matrix.png')
            shutil.copy(confusion_matrix_path, dest_path)
            logger.info(f"Copied confusion matrix to {dest_path}")
        
        logger.info(f"Visualizations saved to {viz_dir}")
        return viz_dir
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return None

def find_confusion_matrix(model_path):
    """Find confusion matrix generated during evaluation."""
    # YOLOv5 typically saves confusion matrix in the same directory as the model
    model_dir = os.path.dirname(os.path.abspath(model_path))
    
    # Look for confusion matrix in parent directories
    for parent in range(3):  # Check up to 3 levels up
        current_dir = model_dir
        for _ in range(parent):
            current_dir = os.path.dirname(current_dir)
        
        # Check for confusion matrix file
        cm_path = os.path.join(current_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            return cm_path
    
    return None

def analyze_cyclist_performance(args, results):
    """Perform detailed analysis of cyclist detection performance."""
    if not args.cyclist_focus:
        return None
    
    logger.info("Performing detailed analysis of cyclist detection performance")
    
    try:
        # Check if cyclist class exists in results
        cyclist_metrics = None
        for cls_name, metrics in results.get('classes', {}).items():
            if cls_name.lower() == 'cyclist':
                cyclist_metrics = metrics
                break
        
        if not cyclist_metrics:
            logger.warning("Cyclist class not found in evaluation results")
            return None
        
        # Create output directory for cyclist analysis
        cyclist_dir = os.path.join(args.output_dir, 'cyclist_analysis')
        os.makedirs(cyclist_dir, exist_ok=True)
        
        # Generate detailed report
        report = {
            'cyclist_metrics': cyclist_metrics,
            'comparison_to_overall': {
                'mAP_ratio': cyclist_metrics['mAP@.5'] / results['overall']['mAP@.5'],
                'precision_percentile': percentile_among_classes(results, 'precision', 'cyclist'),
                'recall_percentile': percentile_among_classes(results, 'recall', 'cyclist')
            },
            'recommendations': generate_recommendations(cyclist_metrics, results)
        }
        
        # Save report
        report_path = os.path.join(cyclist_dir, 'cyclist_analysis.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary text file
        summary_path = os.path.join(cyclist_dir, 'cyclist_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CYCLIST DETECTION PERFORMANCE ANALYSIS\n")
            f.write("=====================================\n\n")
            f.write(f"Precision: {cyclist_metrics['precision']:.4f}\n")
            f.write(f"Recall: {cyclist_metrics['recall']:.4f}\n")
            f.write(f"mAP@.5: {cyclist_metrics['mAP@.5']:.4f}\n\n")
            
            f.write("COMPARISON TO OVERALL MODEL PERFORMANCE\n")
            f.write("--------------------------------------\n")
            f.write(f"mAP Ratio (Cyclist/Overall): {report['comparison_to_overall']['mAP_ratio']:.4f}\n")
            f.write(f"Precision Percentile: {report['comparison_to_overall']['precision_percentile']:.1f}\n")
            f.write(f"Recall Percentile: {report['comparison_to_overall']['recall_percentile']:.1f}\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("---------------\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Cyclist analysis saved to {cyclist_dir}")
        return report
    
    except Exception as e:
        logger.error(f"Error analyzing cyclist performance: {str(e)}")
        return None

def percentile_among_classes(results, metric, target_class):
    """Calculate percentile of a class for a given metric among all classes."""
    values = []
    target_value = None
    
    for cls_name, metrics in results.get('classes', {}).items():
        if metric in metrics:
            values.append(metrics[metric])
            if cls_name.lower() == target_class.lower():
                target_value = metrics[metric]
    
    if target_value is None or not values:
        return 0
    
    # Calculate percentile
    values.sort()
    rank = values.index(target_value)
    percentile = (rank / (len(values) - 1)) * 100
    
    return percentile

def generate_recommendations(cyclist_metrics, results):
    """Generate recommendations for improving cyclist detection."""
    recommendations = []
    
    # Check precision
    if cyclist_metrics['precision'] < 0.7:
        recommendations.append(
            "Improve precision by adding more diverse cyclist examples to reduce false positives."
        )
    
    # Check recall
    if cyclist_metrics['recall'] < 0.7:
        recommendations.append(
            "Improve recall by adding more cyclist examples in various poses, lighting conditions, and occlusion levels."
        )
    
    # Check mAP ratio
    map_ratio = cyclist_metrics['mAP@.5'] / results['overall']['mAP@.5']
    if map_ratio < 0.9:
        recommendations.append(
            f"Cyclist performance is {(1-map_ratio)*100:.1f}% below overall model performance. "
            "Consider increasing class weight for cyclists during training."
        )
    
    # Add general recommendations
    recommendations.append(
        "Generate more synthetic data with cyclists in challenging scenarios: low light, partial occlusion, unusual poses."
    )
    
    recommendations.append(
        "Implement test-time augmentation (TTA) specifically for cyclist detection to improve performance."
    )
    
    return recommendations

def main():
    """Main function to run the evaluation pipeline."""
    args = parse_args()
    
    # Run evaluation
    results = run_evaluation(args)
    if not results:
        logger.error("Evaluation failed. Exiting.")
        return 1
    
    # Generate visualizations
    viz_dir = generate_visualizations(args, results)
    if not viz_dir:
        logger.warning("Failed to generate visualizations.")
    
    # Analyze cyclist performance if requested
    if args.cyclist_focus:
        cyclist_report = analyze_cyclist_performance(args, results)
        if not cyclist_report:
            logger.warning("Failed to analyze cyclist performance.")
    
    logger.info("Evaluation pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
