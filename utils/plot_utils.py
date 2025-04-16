import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history, epochs=None):
    """
    Plot training history with accuracy, loss and ROC AUC
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        The history object returned from model.fit()
    epochs : int, optional
        If provided, will limit x-axis to this value. Otherwise uses actual training length.
    """
    # Create figure with 3 subplots (accuracy, loss, AUC)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get actual number of training epochs
    actual_epochs = len(history.history['accuracy'])
    x_limit = epochs if epochs is not None else actual_epochs
    
    # Get maximum loss value for proper y-scaling (with 10% padding)
    max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
    loss_y_limit = max_loss * 1.1
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Training and Validation Accuracy', fontsize=14)
    axes[0].set_xlim([0, x_limit])
    axes[0].set_ylim([0, 1.05])  # Reduced from 1.1 to 1.05
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot with adaptive y-axis limit
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training and Validation Loss', fontsize=14)
    axes[1].set_xlim([0, x_limit])
    axes[1].set_ylim([0, loss_y_limit])  # Dynamic y-limit based on data
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # ROC AUC plot - assuming AUC is always available
    auc_metric = next(metric for metric in history.history.keys() if 'auc' in metric.lower() and not metric.startswith('val'))
    val_auc_metric = f'val_{auc_metric}'
    
    axes[2].plot(history.history[auc_metric], label=f'Training AUC', linewidth=2)
    axes[2].plot(history.history[val_auc_metric], label=f'Validation AUC', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('AUC', fontsize=12)
    axes[2].set_title('Training and Validation AUC', fontsize=14)
    axes[2].set_xlim([0, x_limit])
    axes[2].set_ylim([0, 1.05])  # Reduced from 1.1 to 1.05
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve with AUC area filled"""
    plt.figure(figsize=(8, 6))
    
    plt.fill_between(fpr, tpr, alpha=0.3, color='darkorange', label=f'AUC = {roc_auc:.4f}')
    
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"ROC AUC Score: {roc_auc:.6f}")

def plot_confusion_matrix(cm, class_names=['Normal', 'Pneumonia']):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of the classes, defaults to ['Normal', 'Pneumonia']
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize CM for percentage view alongside raw numbers
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotation text with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()