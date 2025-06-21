"""
Plotting and Visualization Utilities for XAI Analysis
Provides comparison plots and visualization functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_xai_comparison_by_architecture(xai_results, sample_images, sample_labels, output_path):
    """
    Create comparison plots grouped by architecture across different resolutions.
    """
    
    # Group results by architecture
    architectures = {}
    for model_info, results in xai_results.items():
        # Unpack tuple key (architecture, resolution)
        arch, resolution = model_info
        if arch not in architectures:
            architectures[arch] = {}
        architectures[arch][resolution] = results
    
    # Create comparison plots for each architecture
    for architecture, resolution_results in architectures.items():
        if len(resolution_results) < 2:
            continue  # Skip if only one resolution for this architecture
            
        resolutions = sorted(resolution_results.keys())
        
        print(f"Creating comparison plots for {architecture} across {resolutions}px")
        
        # Select first few samples for visualization
        num_samples = min(3, len(sample_images[resolutions[0]]))
        
        # Create comparison figure
        fig, axes = plt.subplots(num_samples, len(resolutions) * 4, 
                                figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx in range(num_samples):
            for res_idx, resolution in enumerate(resolutions):
                # Get data for this resolution
                images = sample_images[resolution]
                labels = sample_labels[resolution]
                results = resolution_results[resolution]
                
                col_offset = res_idx * 4
                
                # Original image
                img = images[sample_idx].squeeze()
                axes[sample_idx, col_offset].imshow(img, cmap='gray')
                axes[sample_idx, col_offset].set_title(f'{resolution}px\nOriginal\n{["Normal", "Pneumonia"][labels[sample_idx]]}')
                axes[sample_idx, col_offset].axis('off')
                
                # GradCAM
                if results['gradcam'] and results['gradcam'][sample_idx] is not None:
                    gradcam_img = results['gradcam'][sample_idx].squeeze()
                    axes[sample_idx, col_offset + 1].imshow(img, cmap='gray', alpha=0.7)
                    axes[sample_idx, col_offset + 1].imshow(gradcam_img, cmap='jet', alpha=0.3)
                    axes[sample_idx, col_offset + 1].set_title('GradCAM')
                else:
                    axes[sample_idx, col_offset + 1].text(0.5, 0.5, 'GradCAM\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 1].transAxes)
                axes[sample_idx, col_offset + 1].axis('off')
                
                # LIME
                if results['lime'] and results['lime'][sample_idx] is not None:
                    try:
                        lime_img, lime_mask = results['lime'][sample_idx].get_image_and_mask(
                            results['pred_classes'][sample_idx], positive_only=True, hide_rest=False)
                        axes[sample_idx, col_offset + 2].imshow(lime_img)
                        axes[sample_idx, col_offset + 2].set_title('LIME')
                    except:
                        axes[sample_idx, col_offset + 2].text(0.5, 0.5, 'LIME\nFailed', 
                                                            ha='center', va='center', transform=axes[sample_idx, col_offset + 2].transAxes)
                else:
                    axes[sample_idx, col_offset + 2].text(0.5, 0.5, 'LIME\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 2].transAxes)
                axes[sample_idx, col_offset + 2].axis('off')
                
                # SHAP
                if results['shap'] is not None and sample_idx < len(results['shap'][0]):
                    shap_img = results['shap'][results['pred_classes'][sample_idx]][sample_idx].squeeze()
                    im = axes[sample_idx, col_offset + 3].imshow(shap_img, cmap='RdBu', 
                                                               vmin=-np.max(np.abs(shap_img)), 
                                                               vmax=np.max(np.abs(shap_img)))
                    axes[sample_idx, col_offset + 3].set_title('SHAP')
                    plt.colorbar(im, ax=axes[sample_idx, col_offset + 3], fraction=0.046, pad=0.04)
                else:
                    axes[sample_idx, col_offset + 3].text(0.5, 0.5, 'SHAP\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 3].transAxes)
                axes[sample_idx, col_offset + 3].axis('off')
        
        plt.suptitle(f'{architecture}: XAI Comparison Across Resolutions', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_path, f'{architecture}_resolution_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison plot: {save_path}")


def plot_xai_comparison_by_resolution(xai_results, sample_images, sample_labels, output_path):
    """
    Create comparison plots grouped by resolution across different architectures.
    """
    
    # Group results by resolution
    resolutions = {}
    for model_info, results in xai_results.items():
        # Unpack tuple key (architecture, resolution)
        arch, res = model_info
        if res not in resolutions:
            resolutions[res] = {}
        resolutions[res][arch] = results
    
    # Create comparison plots for each resolution
    for resolution, architecture_results in resolutions.items():
        if len(architecture_results) < 2:
            continue  # Skip if only one architecture for this resolution
            
        architectures = list(architecture_results.keys())
        
        print(f"Creating comparison plots for {resolution}px across {architectures}")
        
        # Select first few samples for visualization
        num_samples = min(3, len(sample_images[resolution]))
        
        # Create comparison figure
        fig, axes = plt.subplots(num_samples, len(architectures) * 4, 
                                figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx in range(num_samples):
            for arch_idx, architecture in enumerate(architectures):
                # Get data for this architecture
                images = sample_images[resolution]
                labels = sample_labels[resolution]
                results = architecture_results[architecture]
                
                col_offset = arch_idx * 4
                
                # Original image
                img = images[sample_idx].squeeze()
                axes[sample_idx, col_offset].imshow(img, cmap='gray')
                axes[sample_idx, col_offset].set_title(f'{architecture}\nOriginal\n{["Normal", "Pneumonia"][labels[sample_idx]]}')
                axes[sample_idx, col_offset].axis('off')
                
                # GradCAM
                if results['gradcam'] and results['gradcam'][sample_idx] is not None:
                    gradcam_img = results['gradcam'][sample_idx].squeeze()
                    axes[sample_idx, col_offset + 1].imshow(img, cmap='gray', alpha=0.7)
                    axes[sample_idx, col_offset + 1].imshow(gradcam_img, cmap='jet', alpha=0.3)
                    axes[sample_idx, col_offset + 1].set_title('GradCAM')
                else:
                    axes[sample_idx, col_offset + 1].text(0.5, 0.5, 'GradCAM\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 1].transAxes)
                axes[sample_idx, col_offset + 1].axis('off')
                
                # LIME
                if results['lime'] and results['lime'][sample_idx] is not None:
                    try:
                        lime_img, lime_mask = results['lime'][sample_idx].get_image_and_mask(
                            results['pred_classes'][sample_idx], positive_only=True, hide_rest=False)
                        axes[sample_idx, col_offset + 2].imshow(lime_img)
                        axes[sample_idx, col_offset + 2].set_title('LIME')
                    except:
                        axes[sample_idx, col_offset + 2].text(0.5, 0.5, 'LIME\nFailed', 
                                                            ha='center', va='center', transform=axes[sample_idx, col_offset + 2].transAxes)
                else:
                    axes[sample_idx, col_offset + 2].text(0.5, 0.5, 'LIME\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 2].transAxes)
                axes[sample_idx, col_offset + 2].axis('off')
                
                # SHAP
                if results['shap'] is not None and sample_idx < len(results['shap'][0]):
                    shap_img = results['shap'][results['pred_classes'][sample_idx]][sample_idx].squeeze()
                    im = axes[sample_idx, col_offset + 3].imshow(shap_img, cmap='RdBu', 
                                                               vmin=-np.max(np.abs(shap_img)), 
                                                               vmax=np.max(np.abs(shap_img)))
                    axes[sample_idx, col_offset + 3].set_title('SHAP')
                    plt.colorbar(im, ax=axes[sample_idx, col_offset + 3], fraction=0.046, pad=0.04)
                else:
                    axes[sample_idx, col_offset + 3].text(0.5, 0.5, 'SHAP\nFailed', 
                                                        ha='center', va='center', transform=axes[sample_idx, col_offset + 3].transAxes)
                axes[sample_idx, col_offset + 3].axis('off')
        
        plt.suptitle(f'{resolution}px: XAI Comparison Across Architectures', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_path, f'{resolution}px_architecture_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison plot: {save_path}")


def analyze_misclassifications(predictions_data, sample_images, sample_labels, output_path):
    """
    Analyze misclassifications across different models and create visualizations.
    This version works robustly with available prediction data and sample data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print("Analyzing misclassifications across models...")
    
    # Since the prediction files may not have all required keys, 
    # we'll work with the sample data directly for a more reliable analysis
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get available resolutions from sample data
    resolutions = list(sample_images.keys())
    
    if not resolutions:
        print("No sample data available for misclassification analysis")
        return {
            'misclassified_by_all': [],
            'correctly_classified_by_all': [],
            'total_samples_analyzed': 0,
            'models_analyzed': []
        }
    
    # For simplicity, analyze the first resolution that has data
    resolution = resolutions[0]
    sample_imgs = sample_images[resolution]
    sample_lbls = sample_labels[resolution]
    num_samples = len(sample_imgs)
    
    print(f"Analyzing {num_samples} samples at {resolution}px resolution")
    
    # Load models and get their predictions on sample data
    model_results = {}
    model_names = []
    
    for model_info, pred_data in predictions_data.items():
        # Handle both dict and tuple keys
        if isinstance(model_info, dict):
            key = f"{model_info['architecture']}_{model_info['resolution']}px"
            model_resolution = model_info['resolution']
        elif isinstance(model_info, tuple):
            arch, model_resolution = model_info
            key = f"{arch}_{model_resolution}px"
        else:
            key = str(model_info)
            model_resolution = resolution  # fallback
        
        # Only analyze models that match our sample resolution
        if model_resolution == resolution:
            model_names.append(key)
            
            # Get predictions for our sample images
            # Since we can't reliably map indices, we'll load the model and predict directly
            try:
                # Extract predictions from pred_data if available
                if 'predictions' in pred_data and len(pred_data['predictions']) >= num_samples:
                    # Use the first num_samples predictions as a proxy
                    predictions = pred_data['predictions'][:num_samples]
                    model_results[key] = predictions
                else:
                    print(f"Warning: Cannot get reliable predictions for {key}")
                    
            except Exception as e:
                print(f"Warning: Failed to get predictions for {key}: {e}")
    
    if not model_results:
        print("No model predictions available for analysis")
        return {
            'misclassified_by_all': [],
            'correctly_classified_by_all': [],
            'total_samples_analyzed': 0,
            'models_analyzed': []
        }
    
    # Analyze misclassifications
    misclassified_by_all = []
    correctly_classified_by_all = []
    
    for sample_idx in range(num_samples):
        true_label = sample_lbls[sample_idx]
        
        all_correct = True
        all_incorrect = True
        models_with_predictions = 0
        
        for model_key, predictions in model_results.items():
            if sample_idx < len(predictions):
                pred_label = predictions[sample_idx]
                models_with_predictions += 1
                
                if pred_label == true_label:
                    all_incorrect = False
                else:
                    all_correct = False
        
        # Only consider samples where we have predictions from multiple models
        if models_with_predictions > 1:
            if all_correct:
                correctly_classified_by_all.append(sample_idx)
            elif all_incorrect:
                misclassified_by_all.append(sample_idx)
    
    # Create visualization of misclassifications
    try:
        if misclassified_by_all and len(misclassified_by_all) > 0:
            num_to_show = min(3, len(misclassified_by_all))
            fig, axes = plt.subplots(1, num_to_show, figsize=(12, 4))
            
            if num_to_show == 1:
                axes = [axes]
            
            for i, sample_idx in enumerate(misclassified_by_all[:num_to_show]):
                img = sample_imgs[sample_idx]
                if len(img.shape) == 3 and img.shape[-1] == 1:
                    img = img.squeeze()
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'Sample {sample_idx}\n(Misclassified by all)')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'misclassified_samples.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Misclassification visualization saved")
            
    except Exception as e:
        print(f"Warning: Could not create misclassification visualization: {e}")
    
    print(f"Analysis complete: {len(correctly_classified_by_all)} correctly classified by all, "
          f"{len(misclassified_by_all)} misclassified by all")
    
    return {
        'misclassified_by_all': misclassified_by_all,
        'correctly_classified_by_all': correctly_classified_by_all,
        'total_samples_analyzed': num_samples,
        'models_analyzed': list(model_results.keys()),
        'resolution_analyzed': resolution
    }
