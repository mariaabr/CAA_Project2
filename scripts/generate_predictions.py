#!/usr/bin/env python3
"""
Standalone prediction generator for XAI analysis
Processes one model at a time to avoid memory issues
"""

import os
import sys
import numpy as np
import pickle
import argparse
import tensorflow as tf
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def parse_model_info(model_filename):
    """Parse model information from filename."""
    parts = model_filename.replace('.keras', '').split('_')
    
    # Extract architecture
    if 'alexnet' in model_filename:
        architecture = 'AlexNet'
    elif 'densenet' in model_filename:
        architecture = 'DenseNet169'
    elif 'resnet_18' in model_filename:
        architecture = 'ResNet18'
    elif 'resnet_50' in model_filename:
        architecture = 'ResNet50'
    else:
        architecture = 'Unknown'
    
    # Extract resolution
    if '128px' in model_filename:
        resolution = 128
    elif '64px' in model_filename:
        resolution = 64
    elif '28px' in model_filename:
        resolution = 28
    else:
        resolution = 28  # default for non-tuned models
    
    # Check if tuned
    is_tuned = 'tuned' in model_filename
    
    return {
        'architecture': architecture,
        'resolution': resolution,
        'is_tuned': is_tuned,
        'filename': model_filename
    }

def load_test_data(data_path, image_size):
    """Load test data for a specific image size."""
    test_images_path = os.path.join(data_path, f'pneumoniamnist_{image_size}', 'test_images.npy')
    test_labels_path = os.path.join(data_path, f'pneumoniamnist_{image_size}', 'test_labels.npy')
    
    if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"Test data not found for size {image_size}px")
    
    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path).flatten()
    
    # Normalize images
    test_images = test_images.astype(np.float32) / 255.0
    
    # Add channel dimension if needed
    if len(test_images.shape) == 3:
        test_images = np.expand_dims(test_images, -1)
    
    return test_images, test_labels

def save_model_predictions(model, test_images, test_labels, model_info, output_path):
    """Save model predictions to disk."""
    
    print("Generating predictions...")
    predictions = model.predict(test_images, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    pred_probs = np.max(predictions, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(pred_classes == test_labels)
    
    # Prepare data to save
    prediction_data = {
        'model_info': model_info,
        'predictions': pred_classes,
        'probabilities': pred_probs,
        'prediction_scores': predictions,  # Save full scores for analysis
        'true_labels': test_labels,
        'indices': list(range(len(test_labels))),
        'accuracy': accuracy,
        'num_samples': len(test_labels)
    }
    
    # Save to file
    filename = f"{model_info['architecture']}_{model_info['resolution']}px_predictions.pkl"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(prediction_data, f)
    
    print(f"✓ Predictions saved: {filename} (Accuracy: {accuracy:.3f})")
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for a single model')
    parser.add_argument('model_file', help='Model filename (e.g., alexnet_augmented_b96_lr0.001_dr0.5_pneumonia_model.keras)')
    parser.add_argument('--data_path', default='../data/', help='Path to data directory')
    parser.add_argument('--model_path', default='../output/models/', help='Path to models directory')
    parser.add_argument('--output_path', default='../output/predictions/', help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Set up TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Parse paths
    script_dir = Path(__file__).parent
    data_path = script_dir / args.data_path
    model_path = script_dir / args.model_path / args.model_file
    output_path = script_dir / args.output_path
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {args.model_file}")
    
    # Parse model information
    model_info = parse_model_info(args.model_file)
    print(f"Architecture: {model_info['architecture']}")
    print(f"Resolution: {model_info['resolution']}px")
    print(f"Is tuned: {model_info['is_tuned']}")
    
    # Check if predictions already exist
    pred_filename = f"{model_info['architecture']}_{model_info['resolution']}px_predictions.pkl"
    pred_filepath = output_path / pred_filename
    
    if pred_filepath.exists():
        print(f"✓ Predictions already exist: {pred_filename}")
        return
    
    # Check if model file exists
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return
    
    try:
        # Load test data for this resolution
        test_images, test_labels = load_test_data(str(data_path), model_info['resolution'])
        print(f"✓ Test data loaded: {test_images.shape}")
        
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model(str(model_path))
        print("✓ Model loaded successfully!")
        
        # Save predictions
        save_model_predictions(model, test_images, test_labels, model_info, str(output_path))
        
        print("✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"✗ Error processing {args.model_file}: {e}")
        return 1
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'test_images' in locals():
            del test_images, test_labels
        tf.keras.backend.clear_session()
    
    return 0

if __name__ == "__main__":
    exit(main())
