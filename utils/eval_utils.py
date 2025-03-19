import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def evaluate_model(model, test_images, test_labels):
    """Evaluate model and return metrics"""
    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_images, test_labels)
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels[:, 1], predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, 
                                    target_names=['Normal', 'Pneumonia'])
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results

def compare_models(best_model, selected_model, test_images, test_labels):
    """Compare two models on the same test data"""
    results1 = evaluate_model(best_model, test_images, test_labels)
    results2 = evaluate_model(selected_model, test_images, test_labels)
    
    # Compare metrics
    print("Best Model (in Training) vs Selected Model (through callbacks):")
    print(f"Test accuracy: {results1['test_accuracy']:.6f} vs {results2['test_accuracy']:.6f}")
    print(f"Test AUC: {results1['roc']['auc']:.6f} vs {results2['roc']['auc']:.6f}")
    
    return results1, results2