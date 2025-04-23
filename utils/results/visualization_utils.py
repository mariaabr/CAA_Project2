import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

from utils.results.constants import MODELS, BASE_COLORS, AUGMENTATION_COLOR_MAP, AUG_IMPACT_PALETTE, DROPOUT_SETTINGS, DEFAULT_NUMERIC_COLS

def plot_metric_comparison(df, metric='test_auc', title_metric='AUC', ylim=(0.8, 1.01)):
    """General function to plot a performance metric comparison."""
    plt.figure(figsize=(16, 8))
    
    # Create a custom order for the hue based on the MODELS constant
    custom_hue_order = []
    for model in MODELS:  
        custom_hue_order.append((model, False))  # No Aug
        custom_hue_order.append((model, True))   # Aug
    
    # Filter to only include combinations that exist in the data
    custom_hue_order = [combo for combo in custom_hue_order 
                    if combo in df['hue_combined'].unique()]
    
    # Map the combined hue tuple to the color
    palette = {level: AUGMENTATION_COLOR_MAP.get(level, '#808080') 
            for level in df['hue_combined'].cat.categories}
    
    ax = sns.barplot(
        x='dropout_setting',
        y=metric,
        hue='hue_combined',  
        hue_order=custom_hue_order,  
        data=df,
        palette=palette,
        order=DROPOUT_SETTINGS,  
        errorbar=('ci', 95)  
    )
    
    plt.title(f'Mean Test {title_metric} by Dropout Strategy, Model, and Augmentation')
    plt.xlabel('Dropout Strategy')
    plt.ylabel(f'Mean Test {title_metric}')
    
    if ylim:
        plt.ylim(ylim)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{model} ({'Aug' if aug else 'No Aug'})" 
                for model, aug in custom_hue_order]
    
    # Identify missing model-dropout combinations
    all_models = set(model for model, _ in df['hue_combined'].unique())
    missing_models = {}
    for dropout in DROPOUT_SETTINGS:
        dropout_data = df[df['dropout_setting'] == dropout]
        dropout_models = set(model for model, _ in dropout_data['hue_combined'].unique())
        missing = all_models - dropout_models
        if missing:
            missing_models[dropout] = missing
    
    # Add a note if there are missing combinations
    if missing_models:
        missing_notes = []
        for dropout, models in missing_models.items():
            if models:
                missing_notes.append(
                    f"{', '.join(models)} data not available for {dropout}"
                )
        
        bbox = {"facecolor": "orange", "alpha": 0.2, "pad": 5}
        
        # Get current axes position
        ax_pos = ax.get_position()
        fig = plt.gcf()
        
        # Calculate note position relative to axes
        note_x = ax_pos.x0 + ax_pos.width/2  # Center aligned with plot
        note_y = ax_pos.y0 - 0.1  # Slightly below plot
        
        # Add figure text centered with plot
        plt.figtext(
            note_x,
            note_y,
            "Note: " + "; ".join(missing_notes),
            ha="center",
            fontsize=10,
            bbox=bbox
        )
    
    # Add legend with adjusted position
    plt.legend(
        handles=handles,
        labels=new_labels,
        title='Model (Augmentation)',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1
    )
    
    # Create results dataframe with all bar data
    results = []
    
    # Iterate through dropout settings
    for dropout in DROPOUT_SETTINGS:
        # Get data for this dropout setting
        dropout_data = df[df['dropout_setting'] == dropout]
        
        # For each model/augmentation combination
        for model, aug in custom_hue_order:
            model_aug_data = dropout_data[
                (dropout_data['model_base_name'] == model) & 
                (dropout_data['augmented'] == aug)
            ]
            
            # Only include if we have data
            if not model_aug_data.empty:
                # Calculate statistics
                mean_value = model_aug_data[metric].mean()
                
                # Calculate 95% confidence interval
                import scipy.stats as stats
                n = len(model_aug_data)
                if n > 1:  # Can only calculate CI with more than one sample
                    sem = model_aug_data[metric].sem()
                    ci_95 = stats.t.interval(0.95, n-1, loc=mean_value, scale=sem)
                    lower_ci, upper_ci = ci_95
                else:
                    # If only one sample, use the value itself
                    lower_ci = upper_ci = mean_value
                
                results.append({
                    'Dropout Strategy': dropout,
                    'Model': model,
                    'Augmented': 'Yes' if aug else 'No',
                    f'Mean {title_metric}': mean_value,
                    f'Lower CI (95%)': lower_ci,
                    f'Upper CI (95%)': upper_ci
                })
    
    # Create dataframe from collected results
    result_df = pd.DataFrame(results)
    
    # Display the plot
    plt.show()
    
    # Return the dataframe with the results
    return result_df

def plot_false_negative_comparison(df):
    """Plot false negative rates comparison across models and augmentation."""
    plt.figure(figsize=(14, 7))
    plot_df = df[df['false_negative_rate'].notna()].copy()
    if plot_df.empty:
        print("No data available for False Negative Rate plot.")
        return None
    
    ax = sns.barplot(
        x='model_base_name',
        y='false_negative_rate',
        hue='augmented',
        data=plot_df,
        order=MODELS,
        palette=AUG_IMPACT_PALETTE,
        errorbar=('ci', 95)
    )
    
    # Add value labels on bars with improved visibility
    for bar in ax.patches:
        if bar.get_height() > 0:  # Avoid labeling zero bars if any
            text_x = bar.get_x() + bar.get_width() / 2.
            text_y = bar.get_height()
            
            # Create semi-transparent white background behind text
            ax.text(
                text_x,
                text_y,
                f"{bar.get_height():.2%}",
                ha='center',
                va='bottom',
                fontsize=9,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,  # Semi-transparent background
                    edgecolor='none'
                ),
                zorder=5  # Ensure label appears above error bars
            )
    
    plt.title('Mean False Negative Rate by Model Type and Data Augmentation')
    plt.xlabel('Model Type')
    plt.ylabel('Mean False Negative Rate (Lower is Better)')
    max_fnr = plot_df['false_negative_rate'].max()
    plt.ylim(0, max(max_fnr * 1.1, 0.05)) 
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
        return None
    
    # Create custom hue order based on MODELS constant
    custom_hue_order = []
    for model in MODELS:
        custom_hue_order.append((model, False))  # No Aug
        custom_hue_order.append((model, True))   # Aug
    
    # Filter to only include combinations that exist in the data
    custom_hue_order = [combo for combo in custom_hue_order 
                        if combo in df[['model_base_name', 'augmented']].values]
    
    # Group data and create color mapping
    grouped_df = plot_df.groupby(['model_base_name', 'augmented', 'batch_size', 'hue_combined'], observed=True)['exec_time'].mean().reset_index()
    palette = {level: AUGMENTATION_COLOR_MAP.get(level, '#808080') 
            for level in grouped_df['hue_combined'].cat.categories}
    
    # Create the catplot with custom ordering
    ax_grid = sns.catplot(
        x='batch_size',
        y='exec_time',
        hue='hue_combined',
        hue_order=custom_hue_order,
        col='model_base_name',
        data=grouped_df,
        kind='bar',
        palette=palette,
        col_order=MODELS,
        col_wrap=2,
        height=5,
        aspect=1.2,
        legend=False
    )
    
    # Set custom titles with just the model name
    ax_grid.set_titles("Mean Execution Time by Batch Size for {col_name} Model")
    
    # Add value labels with background
    for ax in ax_grid.axes.flat:
        for bar in ax.patches:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height(),
                    f"{int(bar.get_height())}s",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none'
                    ),
                    zorder=5
                )
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylabel('Mean Execution Time (s)')
        ax.set_xlabel('Batch Size')
    
    # Create custom legend with correct ordering
    handles = [plt.Rectangle((0,0),1,1, color=AUGMENTATION_COLOR_MAP[key]) 
            for key in custom_hue_order]
    labels = [f"{model} ({'Aug' if aug else 'No Aug'})" 
            for model, aug in custom_hue_order]
    
    # Set title and add legend
    ax_grid.figure.suptitle('Mean Execution Time by Batch Size, Model, and Augmentation', y=1.03)
    ax_grid.figure.legend(handles=handles, labels=labels,
                        title='Model (Augmentation)',
                        bbox_to_anchor=(1.02, 0.5),
                        loc='center left')
    
    # Adjust layout
    ax_grid.figure.tight_layout(rect=[0, 0, 0.9, 1])
    
    return plt.show()

def plot_model_performance(df, x_metric, y_metric, title=None, x_label=None, y_label=None, add_regression=True):
    """Generalized plotting function for model performance with customizable metrics."""
    # Prepare figure + data
    fig, ax = plt.subplots(figsize=(13, 9))
    plot_df = df[df[x_metric].notna() & df[y_metric].notna()].copy()
    if plot_df.empty:
        print(f"No data available for {x_metric} vs {y_metric} plot.")
        return
    
    # Define markers: X for non-augmented, ✓ (checkmark) for augmented
    markers = {False: 'X', True: 'o'} 

    # Build the Scatter Plot
    sns.scatterplot(
        data=plot_df,
        x=x_metric,
        y=y_metric,
        hue='model_base_name',
        style='augmented',
        markers=markers,
        palette=BASE_COLORS,
        s=100,
        alpha=0.8,
        hue_order=MODELS,
        ax=ax
    )

    # Make Regression lines (if requested)
    if add_regression:
        for model in plot_df['model_base_name'].unique():
            md = plot_df[plot_df['model_base_name'] == model]
            if len(md) > 1:
                sns.regplot(
                    x=x_metric,
                    y=y_metric,
                    data=md,
                    scatter=False,
                    ci=None,
                    line_kws={'alpha': 0.5, 'linestyle': '--'},
                    color=BASE_COLORS.get(model, 'gray'),
                    ax=ax
                )

    # Create Labels & set limits
    ax.set_title(title or f'{y_metric} vs. {x_metric}')
    ax.set_xlabel(x_label or x_metric.replace('_', ' ').title())
    ax.set_ylabel(y_label or y_metric.replace('_', ' ').title())
    if y_metric in ['test_auc', 'precision_pneumonia', 'recall_pneumonia']:
        ymin = max(0.5, plot_df[y_metric].min() - 0.02)
        ax.set_ylim(ymin, 1.01)
    ax.grid(True, alpha=0.4)

    # Build one legend merging model color + aug marker
    handles = []
    for model in MODELS:
        for aug in [False, True]:
            subset = plot_df[(plot_df['model_base_name'] == model) & (plot_df['augmented'] == aug)]
            if subset.empty:
                continue
            handles.append(
                Line2D([0], [0],
                    marker=markers[aug],
                    color=BASE_COLORS.get(model, 'gray'),
                    label=f"{model}{' (Aug)' if aug else ''}",
                    markersize=10,
                    linestyle='None')
            )

    # Place merged legend outside
    plt.subplots_adjust(right=0.75)
    ax.legend(handles=handles,
            title="Model (Style: Augmented)",
            loc="upper left",
            bbox_to_anchor=(1.02, 1))
    
    return plt.show()

def plot_precision_recall_tradeoff(df, add_regression=True):
    """Precision vs. Recall Trade-off (Pneumonia Class)."""
    return plot_model_performance(
        df=df,
        x_metric='recall_pneumonia',
        y_metric='precision_pneumonia',
        title='Precision vs. Recall Trade-off (Pneumonia Class)',
        x_label='Recall (Sensitivity)',
        y_label='Precision',
        add_regression=add_regression
    )

def plot_efficiency_tradeoff(df, add_regression=True):
    """Efficiency Trade-off: Test AUC vs. Execution Time."""
    return plot_model_performance(
        df=df,
        x_metric='exec_time',
        y_metric='test_auc',
        title='Efficiency Trade-off: Test AUC vs. Execution Time',
        x_label='Execution Time (seconds)',
        y_label='Test AUC',
        add_regression=add_regression
    )

def plot_hyperparameter_impact(df, group_var, metrics=['test_accuracy', 'test_auc', 'false_negative_rate', 'exec_time'], 
                            figsize=None, order=None, palette=None, value_format=None):
    """Plot the impact of a hyperparameter on multiple performance metrics."""
    # Filter out missing group_var values
    plot_df = df[df[group_var].notna()].copy()
    if plot_df.empty:
        print(f"No data available for {group_var} impact plot.")
        return None

    # Default palette for augmentation
    if palette is None and group_var == 'augmented':
        palette = AUG_IMPACT_PALETTE

    # Compute means with observed=True to silence FutureWarning
    impact_data = (
        plot_df
        .groupby(group_var, observed=True)[metrics]
        .mean()
        .reset_index()
    )

    # Figure size default
    if figsize is None:
        figsize = (10, 3 * len(metrics))

    # Ordering
    if order is None:
        order = sorted(impact_data[group_var].unique())

    # Value format defaults
    if value_format is None:
        value_format = {
            'test_accuracy': '{:.4f}',
            'test_auc':      '{:.4f}',
            'false_negative_rate': '{:.2%}',
            'exec_time':     '{:.1f}s',
        }

    # Create subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in impact_data or impact_data[metric].isna().all():
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            continue

        # Unified barplot call with hue to avoid deprecation
        sns.barplot(
            x=group_var,
            y=metric,
            hue=group_var,
            data=impact_data,
            palette=palette,
            order=order,
            ax=ax,
            legend=False
        )

        # Annotate bars
        for p in ax.patches:
            val = p.get_height()
            if pd.isna(val) or val == 0:
                continue
            fmt = value_format.get(metric, '{:.4f}')
            ax.annotate(
                fmt.format(val),
                (p.get_x() + p.get_width() / 2, val),
                ha='center', va='bottom', fontsize=9
            )

        # Titles, labels, grid
        pretty_var    = group_var.replace('_',' ').title()
        pretty_metric = metric.replace('_',' ').title()
        ax.set_title(f'Impact of {pretty_var} on {pretty_metric}')
        ax.set_ylabel(pretty_metric)
        ax.set_xlabel(pretty_var if ax is axes[-1] else '')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Percentage format on false negative rate
        if metric == 'false_negative_rate':
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        # Fixed tick labels for augmentation
        if group_var == 'augmented':
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(['No Augmentation', 'With Augmentation'])
        
        # Increase y-axis limit for breathing room
        max_val = impact_data[metric].max()
        ax.set_ylim(0, max_val * 1.1)

    plt.tight_layout()
    plt.show()

def plot_aug_impact_by_model(model_df, model_col='model_base_name', augment_col='augmented', metric_col='test_auc', palette=AUG_IMPACT_PALETTE, figsize=(12, 8)):
    """Plot impact of data augmentation on mean metric by model type."""
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

def plot_correlation_matrix(model_df, cols=DEFAULT_NUMERIC_COLS, figsize=(10, 8), cmap="coolwarm"):
    """Calculate and plot correlation matrix for given numeric columns in model_df."""
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

    cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad(color='lightgray', alpha=0.25)  # Set color for NaN values

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
        cbar_kws={"shrink": .5},
        vmin=-1,
        vmax=1,
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
