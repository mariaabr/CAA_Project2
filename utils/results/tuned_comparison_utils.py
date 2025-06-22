"""
Utilities for analyzing tuned vs non-tuned model comparisons.
Contains plotting, analysis, and reporting functions specific to model evolution studies.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_model_comparison_plots(df):
    """
    Create comprehensive model comparison plots for architecture and resolution analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model data with columns: model_name_from_txt, test_auc, 
        test_accuracy, fn, fp, etc.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with 6 subplots for comprehensive analysis
    """
    # Prepare data
    viz_df = df.copy()
    
    # Extract architecture name for grouping
    viz_df['architecture'] = viz_df['model_name_from_txt'].apply(
        lambda x: x.split('_')[0] if x.startswith('alexnet') else 
                  'densenet_169' if x.startswith('densenet_169') else
                  'resnet_18' if 'resnet_18' in x else
                  'resnet_50' if 'resnet_50' in x else 'unknown'
    )
    
    # Calculate false negative rate and false positive rate if not available
    if 'false_negative_rate' not in viz_df.columns:
        viz_df['false_negative_rate'] = viz_df['fn'] / (viz_df['fn'] + viz_df['tp'])
    if 'false_positive_rate' not in viz_df.columns:
        viz_df['false_positive_rate'] = viz_df['fp'] / (viz_df['fp'] + viz_df['tn'])
    
    # Create the plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison: Resolution and Architecture Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. False Negative Rate by Architecture and Resolution (Bar plot instead of box)
    _plot_metric_by_arch_resolution(viz_df, 'false_negative_rate', 'False Negative Rate', axes[0,0])
    
    # 2. AUC by Architecture and Resolution (Bar plot instead of box)
    _plot_metric_by_arch_resolution(viz_df, 'test_auc', 'AUC Score', axes[0,1])
    
    # 3. Accuracy by Architecture and Resolution (Bar plot instead of box)
    _plot_metric_by_arch_resolution(viz_df, 'test_accuracy', 'Accuracy', axes[0,2])
    
    # 4. Trade-off: FN Rate vs FP Rate scatter plot
    _plot_fn_vs_fp_scatter(viz_df, axes[1,0])
    
    # 5. Architecture Performance: Mean metrics comparison
    _plot_architecture_comparison(viz_df, axes[1,1])
    
    # 6. Resolution Impact: Performance change from base to higher resolutions
    _plot_resolution_improvements(viz_df, axes[1,2])
    
    plt.tight_layout()
    return plt.show()


def _plot_metric_by_arch_resolution(df, metric, title, ax):
    """Helper function to create bar plots for metrics by architecture and resolution."""
    # Create pivot table for bar plot
    pivot_data = df.pivot_table(values=metric, index='architecture', columns='resolution', aggfunc='mean')
    
    # Create bar plot
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{title} by Architecture and Resolution')
    ax.set_ylabel(title)
    ax.set_xlabel('Architecture')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Resolution (px)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')


def _plot_fn_vs_fp_scatter(df, ax):
    """Helper function to create scatter plot of FN rate vs FP rate by resolution."""
    resolution_colors = {28: 'blue', 64: 'orange', 128: 'green'}
    
    for resolution in df['resolution'].unique():
        subset = df[df['resolution'] == resolution]
        ax.scatter(subset['false_negative_rate'], subset['false_positive_rate'], 
                  label=f'{resolution}px', alpha=0.7, s=100,
                  color=resolution_colors.get(resolution, 'gray'))
    
    ax.set_xlabel('False Negative Rate')
    ax.set_ylabel('False Positive Rate')
    ax.set_title('Trade-off: FN Rate vs FP Rate by Resolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to side
    ax.grid(True, alpha=0.3)


def _plot_architecture_comparison(df, ax):
    """Helper function to create architecture comparison bar plot."""
    arch_means = df.groupby('architecture')[['false_negative_rate', 'false_positive_rate', 'test_auc', 'test_accuracy']].mean()
    arch_means.plot(kind='bar', ax=ax)
    ax.set_title('Average Performance by Architecture')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(['FN Rate', 'FP Rate', 'AUC', 'Accuracy'], bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to side
    ax.grid(True, alpha=0.3, axis='y')


def _plot_resolution_improvements(df, ax):
    """Helper function to create resolution improvement analysis."""
    tuned_only = df[df['tuned'] == True].copy()
    base_models = df[df['tuned'] == False].copy()
    
    # Calculate improvements for tuned models
    improvements = []
    for arch in tuned_only['architecture'].unique():
        arch_base = base_models[base_models['architecture'] == arch]
        arch_tuned = tuned_only[tuned_only['architecture'] == arch]
        
        if not arch_base.empty:
            base_fn_rate = arch_base['false_negative_rate'].iloc[0]
            base_fp_rate = arch_base['false_positive_rate'].iloc[0]
            base_auc = arch_base['test_auc'].iloc[0]
            base_acc = arch_base['test_accuracy'].iloc[0]
            
            for _, tuned_row in arch_tuned.iterrows():
                improvements.append({
                    'architecture': arch,
                    'resolution': tuned_row['resolution'],
                    'fn_rate_change': tuned_row['false_negative_rate'] - base_fn_rate,
                    'fp_rate_change': tuned_row['false_positive_rate'] - base_fp_rate,
                    'auc_change': tuned_row['test_auc'] - base_auc,
                    'acc_change': tuned_row['test_accuracy'] - base_acc
                })
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        res_improvements = imp_df.groupby('resolution')[['fn_rate_change', 'fp_rate_change', 'auc_change', 'acc_change']].mean()
        
        x_pos = np.arange(len(res_improvements.index))
        width = 0.2  # Reduced width to accommodate 4 bars
        
        ax.bar(x_pos - 1.5*width, res_improvements['fn_rate_change'], width, label='FN Rate Change', alpha=0.8)
        ax.bar(x_pos - 0.5*width, res_improvements['fp_rate_change'], width, label='FP Rate Change', alpha=0.8)
        ax.bar(x_pos + 0.5*width, res_improvements['auc_change'], width, label='AUC Change', alpha=0.8)
        ax.bar(x_pos + 1.5*width, res_improvements['acc_change'], width, label='Accuracy Change', alpha=0.8)
        
        ax.set_xlabel('Resolution (px)')
        ax.set_ylabel('Change from Base Model')
        ax.set_title('Average Improvement by Resolution')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(res_improvements.index)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to side
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'No tuned models\nfor comparison', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Resolution Impact Analysis')


def create_performance_heatmaps(df):
    """
    Create performance heatmaps for architecture vs resolution analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model data
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with 3 heatmap subplots
    """
    # Prepare data
    viz_df = df.copy()
    viz_df['architecture'] = viz_df['model_name_from_txt'].apply(
        lambda x: x.split('_')[0] if x.startswith('alexnet') else 
                  'densenet_169' if x.startswith('densenet_169') else
                  'resnet_18' if 'resnet_18' in x else
                  'resnet_50' if 'resnet_50' in x else 'unknown'
    )
    
    if 'false_negative_rate' not in viz_df.columns:
        viz_df['false_negative_rate'] = viz_df['fn'] / (viz_df['fn'] + viz_df['tp'])
    
    # Create heatmap plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Performance Heatmaps: Architecture vs Resolution', fontsize=14, fontweight='bold')
    
    metrics_to_plot = ['false_negative_rate', 'test_auc', 'test_accuracy']
    metric_titles = ['False Negative Rate', 'AUC Score', 'Test Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        pivot_data = viz_df.pivot_table(values=metric, index='architecture', columns='resolution', aggfunc='mean')
        
        if metric == 'false_negative_rate':
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                       ax=axes[i], cbar_kws={'label': title})
        else:
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                       ax=axes[i], cbar_kws={'label': title})
        
        axes[i].set_title(f'{title} by Architecture and Resolution')
        axes[i].set_xlabel('Resolution (px)')
        axes[i].set_ylabel('Architecture')
    
    plt.tight_layout()
    return plt.show()


def print_visual_analysis_insights(df):
    """
    Print insights from the visual analysis of model performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model data
    """
    viz_df = df.copy()
    viz_df['architecture'] = viz_df['model_name_from_txt'].apply(
        lambda x: x.split('_')[0] if x.startswith('alexnet') else 
                  'densenet_169' if x.startswith('densenet_169') else
                  'resnet_18' if 'resnet_18' in x else
                  'resnet_50' if 'resnet_50' in x else 'unknown'
    )
    
    if 'false_negative_rate' not in viz_df.columns:
        viz_df['false_negative_rate'] = viz_df['fn'] / (viz_df['fn'] + viz_df['tp'])
    if 'false_positive_rate' not in viz_df.columns:
        viz_df['false_positive_rate'] = viz_df['fp'] / (viz_df['fp'] + viz_df['tn'])
    
    print("=== VISUAL ANALYSIS INSIGHTS ===")
    print("\n1. Resolution Impact:")
    print(f"   - Models are available at resolutions: {sorted(viz_df['resolution'].unique())}px")
    print(f"   - Higher resolutions generally require more computational resources")
    print(f"   - Improvement interpretation: +0.02 FN rate = +2% increase in false negative rate")
    
    print("\n2. Architecture Comparison:")
    for arch in sorted(viz_df['architecture'].unique()):
        arch_data = viz_df[viz_df['architecture'] == arch]
        avg_fn_rate = arch_data['false_negative_rate'].mean()
        avg_fp_rate = arch_data['false_positive_rate'].mean()
        avg_auc = arch_data['test_auc'].mean()
        print(f"   - {arch}: Avg FN Rate = {avg_fn_rate:.4f}, Avg FP Rate = {avg_fp_rate:.4f}, Avg AUC = {avg_auc:.4f}")
    
    print("\n3. Tuning Effects:")
    tuned_count = viz_df['tuned'].sum()
    total_count = len(viz_df)
    print(f"   - {tuned_count}/{total_count} models are tuned versions")
    print(f"   - Tuned models span resolutions: {sorted(viz_df[viz_df['tuned']]['resolution'].unique())}px")
    
    print("\n4. Trade-offs Analysis:")
    print(f"   - FN vs FP trade-off: Medical applications typically prefer lower FN (missing disease)")
    print(f"   - Resolution impact: Higher resolution may improve fine-grained feature detection")
    print(f"   - Architecture differences: Each architecture has different capacity and inductive biases")


def create_detailed_metrics_summary(df):
    """
    Create a detailed summary table and identify best performers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model data
    
    Returns:
    --------
    summary_df : pandas.DataFrame
        Summary table with performance metrics
    best_performers : dict
        Dictionary with best performing models for each metric
    """
    viz_df = df.copy()
    viz_df['architecture'] = viz_df['model_name_from_txt'].apply(
        lambda x: x.split('_')[0] if x.startswith('alexnet') else 
                  'densenet_169' if x.startswith('densenet_169') else
                  'resnet_18' if 'resnet_18' in x else
                  'resnet_50' if 'resnet_50' in x else 'unknown'
    )
    
    if 'false_negative_rate' not in viz_df.columns:
        viz_df['false_negative_rate'] = viz_df['fn'] / (viz_df['fn'] + viz_df['tp'])
    
    # Create summary table
    summary_table = []
    for arch in sorted(viz_df['architecture'].unique()):
        for res in sorted(viz_df['resolution'].unique()):
            subset = viz_df[(viz_df['architecture'] == arch) & (viz_df['resolution'] == res)]
            if not subset.empty:
                fn_rate = subset['false_negative_rate'].mean()
                auc = subset['test_auc'].mean()
                acc = subset['test_accuracy'].mean()
                count = len(subset)
                summary_table.append({
                    'Architecture': arch,
                    'Resolution': f'{res}px',
                    'Count': count,
                    'FN_Rate': f'{fn_rate:.6f}',
                    'AUC': f'{auc:.6f}',
                    'Accuracy': f'{acc:.6f}',
                    'Combined': f'{fn_rate:.4f} | {auc:.4f} | {acc:.4f}'
                })
    
    summary_df = pd.DataFrame(summary_table)
    
    # Identify best performers
    best_performers = {
        'lowest_fn': viz_df.loc[viz_df['false_negative_rate'].idxmin()],
        'highest_auc': viz_df.loc[viz_df['test_auc'].idxmax()],
        'highest_acc': viz_df.loc[viz_df['test_accuracy'].idxmax()]
    }
    
    return summary_df, best_performers


def analyze_model_evolution(df):
    """
    Analyze and display model evolution from non-tuned to tuned versions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model data with columns: model_name_from_txt, test_auc, 
        test_accuracy, fn, fp, precision, recall, f1_score, tuned, resolution, etc.
    
    Returns:
    --------
    results : dict
        Dictionary containing comparison tables for each model architecture
    """
    import pandas as pd
    
    results = {}
    
    # Identify base model names
    base_names = set()
    for name in df['model_name_from_txt']:
        if '_tuned_' in name:
            base_names.add(name.split('_tuned_')[0])
        else:
            base_names.add(name.split('_augmented_')[0].split('_b')[0])
    
    # Use appropriate column names for metrics
    available_precision = 'precision' if 'precision' in df.columns else 'precision_pneumonia'
    available_recall = 'recall' if 'recall' in df.columns else 'recall_pneumonia' 
    available_f1 = 'f1_score' if 'f1_score' in df.columns else 'f1_pneumonia'
    
    # Define comparison metrics
    comparison_metrics = [
        'fn', 'fp', 'test_auc', 'test_accuracy', available_precision, available_recall, available_f1,
        'tuned', 'resolution', 'exec_time'
    ]
    
    for base in sorted(base_names):
        print(f"\n=== {base.upper()} EVOLUTION ===")
        non_tuned = [m for m in df['model_name_from_txt'] if m.startswith(base) and '_tuned_' not in m]
        tuned = [m for m in df['model_name_from_txt'] if m.startswith(base) and '_tuned_' in m]
        
        if not non_tuned or not tuned:
            print("Missing non-tuned or tuned models.")
            continue
        
        for nt in non_tuned:
            print(f"\nBase model: {nt}")
            nt_row = df[df['model_name_from_txt'] == nt]
            
            # Find all tuned versions for this base model
            matching_tuned = [t for t in tuned if t.startswith(base)]
            
            if not matching_tuned:
                print("No matching tuned versions found.")
                continue
            
            # Create comparison DataFrame
            comparison_rows = [nt_row.iloc[0]]  # Base model
            
            for t in matching_tuned:
                t_row = df[df['model_name_from_txt'] == t]
                if not t_row.empty:
                    comparison_rows.append(t_row.iloc[0])
            
            if len(comparison_rows) > 1:
                # Display comparison table
                compare_df = pd.DataFrame(comparison_rows)
                labels = ['Non-Tuned'] + [f'Tuned-{row["resolution"]}px' for row in comparison_rows[1:]]
                compare_df.index = labels
                
                # Format and display
                display_columns = [col for col in comparison_metrics if col in compare_df.columns]
                comparison_display = compare_df[display_columns].copy()
                
                # Format numeric columns
                for col in ['test_accuracy', 'test_auc', available_precision, available_recall, available_f1]:
                    if col in comparison_display.columns:
                        comparison_display[col] = comparison_display[col].round(6)
                
                # Rename columns
                column_renames = {
                    'fn': 'FN', 'fp': 'FP', 'test_auc': 'AUC', 'test_accuracy': 'Accuracy',
                    available_precision: 'Precision', available_recall: 'Recall', available_f1: 'F1-Score',
                    'tuned': 'Tuned', 'resolution': 'Resolution', 'exec_time': 'Exec Time'
                }
                comparison_display = comparison_display.rename(columns=column_renames)
                
                # Store results for notebook display
                results[f"{base}_{nt}"] = comparison_display
                print(comparison_display.to_string())
                
                # Show detailed changes
                _print_model_changes(comparison_rows, available_precision, available_recall, available_f1)
        
        print("\n" + "-"*50)
    
    return results


def _print_model_changes(comparison_rows, available_precision, available_recall, available_f1):
    """Helper function to print detailed changes between non-tuned and tuned models."""
    import pandas as pd
    
    base_row = comparison_rows[0]
    print("\nChanges from non-tuned to tuned versions:")
    
    for i, tuned_row in enumerate(comparison_rows[1:], 1):
        resolution = tuned_row.get('resolution', 'unknown')
        print(f"  {resolution}px version:")
        
        metrics_to_compare = [
            'fn', 'fp', 'test_accuracy', 'test_auc', 
            available_precision, available_recall, available_f1
        ]
        
        for metric in metrics_to_compare:
            base_val = base_row.get(metric)
            tuned_val = tuned_row.get(metric)
            
            if base_val is not None and tuned_val is not None and not pd.isna(base_val) and not pd.isna(tuned_val):
                if metric in ['fn', 'fp']:
                    diff = int(tuned_val) - int(base_val)
                    print(f"    {metric.upper()}: {int(base_val)} → {int(tuned_val)} ({diff:+d})")
                else:
                    diff = tuned_val - base_val
                    metric_name = metric.replace('_pneumonia', '').replace('_', ' ').title()
                    print(f"    {metric_name}: {base_val:.6f} → {tuned_val:.6f} ({diff:+.6f})")
