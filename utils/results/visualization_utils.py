import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils.results.constants import MODELS, BASE_COLORS, AUGMENTATION_COLOR_MAP, AUG_IMPACT_PALETTE, DROPOUT_SETTINGS, DEFAULT_NUMERIC_COLS

def plot_metric_comparison(df, metric='test_auc', title_metric='AUC', ylim=(0.8, 1.01)):
    """General function to plot a performance metric comparison."""
    plt.figure(figsize=(16, 8))

    # Map the combined hue tuple to the color
    palette = {level: AUGMENTATION_COLOR_MAP.get(level, '#808080') for level in df['hue_combined'].cat.categories}

    ax = sns.barplot(
        x='dropout_setting',
        y=metric,
        hue='hue_combined', # Use the combined column for hue
        data=df,
        palette=palette,
        order=DROPOUT_SETTINGS, # Ensure consistent dropout order
        errorbar=('ci', 95) # Show confidence interval
    )

    plt.title(f'Test {title_metric} by Dropout Strategy, Model, and Augmentation')
    plt.xlabel('Dropout Strategy')
    plt.ylabel(f'Test {title_metric}')
    if ylim:
        plt.ylim(ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{model} ({'Aug' if aug else 'No Aug'})" for model, aug in sorted(palette.keys())]

    plt.legend(
        handles=handles, 
        labels=new_labels,
        title='Model (Augmentation)', 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),
        ncol=1  # Forces vertical arrangement
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend
    return plt.show()

def plot_false_negative_comparison(df):
    """Plot false negative rates comparison across models and augmentation."""
    plt.figure(figsize=(14, 7))
    plot_df = df[df['false_negative_rate'].notna()].copy()
    if plot_df.empty:
        print("No data available for False Negative Rate plot.")
        return None

    # Define palette based on augmentation status only for this plot
    aug_palette = {False: '#add8e6', True: '#00008b'} # Light Blue, Dark Blue

    ax = sns.barplot(
        x='model_base_name',
        y='false_negative_rate',
        hue='augmented',
        data=plot_df,
        order=MODELS,
        palette=aug_palette,
        errorbar=('ci', 95)
    )

    # Add value labels on bars
    for bar in ax.patches:
        if bar.get_height() > 0: # Avoid labeling zero bars if any
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height(),
                f"{bar.get_height():.2%}",
                ha='center', va='bottom',
                fontsize=9
            )

    plt.title('False Negative Rate by Model Type and Data Augmentation')
    plt.xlabel('Model Type')
    plt.ylabel('False Negative Rate (Lower is Better)')
    max_fnr = plot_df['false_negative_rate'].max()
    plt.ylim(0, max(max_fnr * 1.1, 0.1)) # Ensure ylim starts at 0, reasonable upper bound
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend(title='Augmented')
    return plt.show()

def plot_execution_time_comparison(df):
    """Plot execution time comparison across models and batch sizes."""
    plot_df = df[df['exec_time'].notna()].copy()
    if plot_df.empty:
        print("No data available for Execution Time plot.")
        return None # Return None if no data

    # Group by model_type, augmentation, batch_size, and the combined hue
    grouped_df = plot_df.groupby(['model_base_name', 'augmented', 'batch_size', 'hue_combined'])['exec_time'].mean().reset_index()

    # Map the combined hue tuple to the color
    palette = {level: AUGMENTATION_COLOR_MAP.get(level, '#808080') for level in grouped_df['hue_combined'].cat.categories}

    ax_grid = sns.catplot( # Use ax_grid to capture the FacetGrid object
        x='batch_size',
        y='exec_time',
        hue='hue_combined', # Use multi-index hue
        col='model_base_name', # Facet by model
        data=grouped_df,
        kind='bar',
        palette=palette,
        col_order=MODELS,
        col_wrap=2, # Wrap into 2 columns
        height=5, aspect=1.2,
        legend=False # Turn off default legend
    )

    # Add value labels (adjust coordinates for catplot)
    for ax in ax_grid.axes.flat: # Iterate through axes in the grid
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height(),
                f"{int(bar.get_height())}s",
                ha='center', va='bottom',
                fontsize=9
            )
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylabel('Mean Execution Time (s)') # Set y-label per subplot
        ax.set_xlabel('Batch Size') # Set x-label per subplot

    ax_grid.figure.suptitle('Mean Execution Time by Batch Size, Model, and Augmentation', y=1.03)

    # Create a custom legend
    handles = [plt.Rectangle((0,0),1,1, color=AUGMENTATION_COLOR_MAP[key]) for key in sorted(palette.keys())]
    labels = [f"{model} ({'Aug' if aug else 'No Aug'})" for model, aug in sorted(palette.keys())]
    ax_grid.figure.legend(handles=handles, labels=labels, title='Model (Augmentation)', bbox_to_anchor=(1.02, 0.5), loc='center left')
    ax_grid.figure.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend

    # Return the figure of the catplot
    return ax_grid.figure

def plot_precision_recall_tradeoff(df):
    """Create precision vs recall scatter plot using the new color scheme."""
    plt.figure(figsize=(13, 9))
    plot_df = df[(df['precision_pneumonia'].notna()) & (df['recall_pneumonia'].notna())].copy()
    if plot_df.empty:
        print("No data available for Precision-Recall plot.")
        return None

    # Use hue for model, style for augmentation
    ax = sns.scatterplot(
        data=plot_df,
        x='recall_pneumonia',
        y='precision_pneumonia',
        hue='model_base_name',
        style='augmented', # Use boolean directly for style
        palette=BASE_COLORS, # Use base colors for hue
        s=100, # Marker size
        alpha=0.8,
        hue_order=MODELS
    )

    plt.title('Precision vs. Recall Trade-off (Pneumonia Class)')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.xlim(max(0, plot_df['recall_pneumonia'].min() - 0.05), 1.01) # Adjust xlim based on data
    plt.ylim(max(0, plot_df['precision_pneumonia'].min() - 0.05), 1.01) # Adjust ylim based on data
    plt.grid(True, alpha=0.4)
    plt.legend(title='Model (Style: Augmented)', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout
    return plt.show()

def plot_efficiency_tradeoff(df):
    """Plot efficiency tradeoff (AUC vs. Time) using the new color scheme."""
    plt.figure(figsize=(13, 9))
    plot_df = df[(df['exec_time'].notna()) & (df['test_auc'].notna())].copy()
    if plot_df.empty:
        print("No data available for Efficiency Tradeoff plot.")
        return None

    # Use hue for model, style for augmentation
    ax = sns.scatterplot(
        data=plot_df,
        x='exec_time',
        y='test_auc',
        hue='model_base_name',
        style='augmented', # Use boolean directly for style
        palette=BASE_COLORS, # Use base colors for hue
        s=100, # Marker size
        alpha=0.8,
        hue_order=MODELS
    )

    plt.title('Efficiency Tradeoff: Test AUC vs. Execution Time')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Test AUC')
    plt.ylim(max(0.5, plot_df['test_auc'].min() - 0.02), 1.01) # Reasonable AUC range
    plt.grid(True, alpha=0.4)
    plt.legend(title='Model (Style: Augmented)', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout
    return plt.show()

def plot_aug_impact_by_model(
    model_df, model_col='model_base_name', augment_col='augmented',
    metric_col='test_auc', palette=AUG_IMPACT_PALETTE, figsize=(12, 8)
):
    """
    Plot impact of data augmentation on mean metric by model type.
    """
    # Aggregate mean metric by model type and augmentation status
    aug_by_model = (
        model_df
        .groupby([model_col, augment_col], observed=False)[metric_col]
        .mean()
        .reset_index()
    )

    # Create barplot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x=model_col,
        y=metric_col,
        hue=augment_col,
        data=aug_by_model,
        palette=palette
    )

    # Annotate bars with values
    for bar in ax.patches:
        ax.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2., bar.get_height()),
            ha='center', va='bottom', fontsize=9
        )

    # Labels and layout
    ax.set_title(f"Impact of Data Augmentation on Mean {metric_col.replace('_', ' ').title()} by Model Type")
    ax.set_xlabel('Model Type')
    ax.set_ylabel(f"Mean {metric_col.replace('_', ' ').title()}")
    ax.set_ylim(bottom=max(0, aug_by_model[metric_col].min() - 0.05), top=1.01)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title=augment_col.capitalize())
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return plt.show()

def plot_correlation_matrix(
    model_df, cols=DEFAULT_NUMERIC_COLS, figsize=(10, 8), cmap="coolwarm"
):
    """
    Calculate and plot correlation matrix for given numeric columns in model_df.
    """
    # Determine which columns to use
    cols_to_use = [col for col in cols if col in model_df.columns]

    if len(cols_to_use) <= 1:
        raise ValueError("Not enough numeric columns available to calculate correlation matrix.")

    # Drop rows with missing values in selected columns
    corr_df = model_df[cols_to_use].dropna()

    if len(corr_df) <= 1:
        raise ValueError("Not enough data points to calculate correlation matrix after dropping NaNs.")

    # Compute correlation matrix
    corr_matrix = corr_df.corr()

    # Plot heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    ax.set_title('Correlation Matrix of Performance Metrics and Hyperparameters')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt.show()

def create_best_models_table(df, top_n=10, sort_by='test_auc'):
    """Create a formatted table of the best performing models."""
    if df.empty or sort_by not in df.columns:
        print(f"Cannot create table. DataFrame is empty or missing sort column '{sort_by}'.")
        return pd.DataFrame()

    # Select and copy relevant columns
    cols = ['model_base_name', 'augmented', 'batch_size', 'dropout_setting',
            'test_accuracy', 'test_auc', 'precision_pneumonia', 'recall_pneumonia',
            'f1_pneumonia', 'false_negative_rate', 'exec_time', 'fn'] # Added F1 and raw FN count
    table_df = df[cols].dropna(subset=[sort_by]).copy()

    # Sort
    table_df = table_df.sort_values(sort_by, ascending=False if sort_by != 'false_negative_rate' else True) # Lower FNR is better

    # Format columns for display
    format_map = {
        'test_accuracy': '{:.4f}', 'test_auc': '{:.4f}',
        'precision_pneumonia': '{:.4f}', 'recall_pneumonia': '{:.4f}',
        'f1_pneumonia': '{:.4f}',
        'false_negative_rate': '{:.2%}',
        'exec_time': '{:.1f}s',
        'fn': '{:.0f}' # Raw false negatives
    }
    for col, fmt in format_map.items():
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')

    # Rename columns for better display
    table_df.columns = ['Model', 'Augmented', 'Batch Size', 'Dropout',
                        'Accuracy', 'AUC', 'Precision (Pneu)', 'Recall (Pneu)',
                        'F1 (Pneu)', 'FNR', 'Time', 'FN Count']

    return table_df.head(top_n)
