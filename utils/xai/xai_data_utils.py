"""
Data Loading and Analysis Utilities for XAI
Provides data loading, metrics calculation, and reporting functions
"""

import os
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr


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
        resolution = 28  # default
    
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
    
    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path).flatten()
    
    # Normalize images
    test_images = test_images.astype(np.float32) / 255.0
    
    # Add channel dimension if needed
    if len(test_images.shape) == 3:
        test_images = np.expand_dims(test_images, -1)
    
    return test_images, test_labels


def select_sample_images(images, labels, num_samples_per_class=10, seed=42):
    """Select balanced sample images for analysis."""
    np.random.seed(seed)
    
    selected_indices = []
    selected_images = []
    selected_labels = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) >= num_samples_per_class:
            # Randomly select samples
            selected_idx = np.random.choice(label_indices, num_samples_per_class, replace=False)
        else:
            # Use all available samples
            selected_idx = label_indices
        
        selected_indices.extend(selected_idx)
        selected_images.extend(images[selected_idx])
        selected_labels.extend(labels[selected_idx])
    
    return np.array(selected_images), np.array(selected_labels), selected_indices


def save_model_predictions(model, test_images, test_labels, model_info, output_path):
    """Save model predictions to disk."""
    
    # Generate predictions
    predictions = model.predict(test_images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    pred_probs = np.max(predictions, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(pred_classes == test_labels)
    
    # Prepare data to save
    prediction_data = {
        'model_info': model_info,
        'predictions': pred_classes,
        'probabilities': pred_probs,
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


def load_model_predictions(filepath):
    """Load model predictions from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_xai_metrics(heatmap1, heatmap2):
    """Calculate similarity metrics between two XAI heatmaps."""
    if heatmap1 is None or heatmap2 is None:
        return {'cosine': np.nan, 'pearson': np.nan, 'spearman': np.nan}
    
    # Flatten heatmaps
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    # Cosine similarity
    cosine_sim = cosine_similarity([h1_flat], [h2_flat])[0, 0]
    
    # Pearson correlation
    try:
        pearson_corr, _ = pearsonr(h1_flat, h2_flat)
    except:
        pearson_corr = np.nan
    
    # Spearman correlation
    try:
        spearman_corr, _ = spearmanr(h1_flat, h2_flat)
    except:
        spearman_corr = np.nan
    
    return {
        'cosine': cosine_sim,
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }


def calculate_xai_similarity_metrics(xai_results):
    """Calculate similarity metrics between different XAI methods for each model."""
    
    similarity_results = {}
    
    for model_info, results in xai_results.items():
        # Unpack tuple key (architecture, resolution)
        arch, resolution = model_info
        model_key = f"{arch}_{resolution}px"
        similarity_results[model_key] = {}
        
        # Compare GradCAM vs LIME
        gradcam_lime_metrics = []
        for i in range(len(results['gradcam'])):
            if (results['gradcam'][i] is not None and 
                results['lime'][i] is not None):
                try:
                    # Get LIME heatmap
                    lime_img, lime_mask = results['lime'][i].get_image_and_mask(
                        results['pred_classes'][i], positive_only=True, hide_rest=False)
                    lime_heatmap = lime_mask.astype(float)
                    
                    metrics = calculate_xai_metrics(results['gradcam'][i], lime_heatmap)
                    gradcam_lime_metrics.append(metrics)
                except:
                    pass
        
        # Aggregate metrics
        if gradcam_lime_metrics:
            similarity_results[model_key]['GradCAM_vs_LIME'] = {
                'mean_cosine': np.nanmean([m['cosine'] for m in gradcam_lime_metrics]),
                'mean_pearson': np.nanmean([m['pearson'] for m in gradcam_lime_metrics]),
                'mean_spearman': np.nanmean([m['spearman'] for m in gradcam_lime_metrics]),
                'num_comparisons': len(gradcam_lime_metrics)
            }
        else:
            similarity_results[model_key]['GradCAM_vs_LIME'] = {
                'mean_cosine': np.nan,
                'mean_pearson': np.nan,
                'mean_spearman': np.nan,
                'num_comparisons': 0
            }
        
        # Compare GradCAM vs SHAP
        gradcam_shap_metrics = []
        if results['shap'] is not None:
            for i in range(min(len(results['gradcam']), len(results['shap'][0]))):
                if results['gradcam'][i] is not None:
                    try:
                        shap_heatmap = results['shap'][results['pred_classes'][i]][i]
                        metrics = calculate_xai_metrics(results['gradcam'][i], shap_heatmap)
                        gradcam_shap_metrics.append(metrics)
                    except:
                        pass
        
        if gradcam_shap_metrics:
            similarity_results[model_key]['GradCAM_vs_SHAP'] = {
                'mean_cosine': np.nanmean([m['cosine'] for m in gradcam_shap_metrics]),
                'mean_pearson': np.nanmean([m['pearson'] for m in gradcam_shap_metrics]),
                'mean_spearman': np.nanmean([m['spearman'] for m in gradcam_shap_metrics]),
                'num_comparisons': len(gradcam_shap_metrics)
            }
        else:
            similarity_results[model_key]['GradCAM_vs_SHAP'] = {
                'mean_cosine': np.nan,
                'mean_pearson': np.nan,
                'mean_spearman': np.nan,
                'num_comparisons': 0
            }
    
    return similarity_results


def save_xai_analysis_report(xai_results, similarity_metrics, misclassification_analysis, output_path):
    """Generate and save comprehensive XAI analysis report."""
    
    report = {
        'summary': {
            'total_models_analyzed': len(xai_results),
            'xai_methods': ['GradCAM', 'LIME', 'SHAP'],
            'total_misclassifications': len(misclassification_analysis['misclassified_by_all']),
            'total_correct_classifications': len(misclassification_analysis['correctly_classified_by_all'])
        },
        'model_summaries': {},
        'similarity_analysis': similarity_metrics,
        'misclassification_analysis': misclassification_analysis
    }
    
    # Generate model summaries
    for model_info, results in xai_results.items():
        # Unpack tuple key (architecture, resolution)
        arch, resolution = model_info
        model_key = f"{arch}_{resolution}px"
        
        # Calculate success rates
        gradcam_success = sum(1 for x in results['gradcam'] if x is not None) / len(results['gradcam'])
        lime_success = sum(1 for x in results['lime'] if x is not None) / len(results['lime'])
        shap_success = 1.0 if results['shap'] is not None else 0.0
        
        # Calculate accuracy
        accuracy = np.mean(results['pred_classes'] == results['true_classes'])
        
        report['model_summaries'][model_key] = {
            'model_accuracy': float(accuracy),
            'gradcam_success_rate': float(gradcam_success),
            'lime_success_rate': float(lime_success),
            'shap_success_rate': float(shap_success),
            'total_samples_analyzed': len(results['pred_classes'])
        }
    
    # Save report
    report_path = os.path.join(output_path, 'xai_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Analysis report saved: {report_path}")
    return report
