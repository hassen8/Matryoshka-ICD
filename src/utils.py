import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def plot_and_save_metrics(history_list, baseline_metrics, config, save_folder="./logs"):
    """
    Generates and saves F1, AUC, and P@5 plots from training history.

    Args:
        history_list (list): A list of dictionaries containing epoch-wise training metrics.
        baseline_metrics (dict): A dictionary containing baseline metrics.
        config (ModelConfig): The configuration object containing nesting_dims.
        save_folder (str): The name of the folder to save the plots in.
    """
    history_df = pd.DataFrame(history_list)
    baseline_df = pd.DataFrame([baseline_metrics])

    # Prepend baseline metrics as epoch 0 and shift training epochs
    history_df = pd.concat([baseline_df, history_df], ignore_index=True)

    # Create the directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    plots_folder = os.path.join(save_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)


    #model type
    model_type = config.model_type

    plt.figure(figsize=(15, 8))
    # Plot F1 Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_f1'], label=f'F1 Dim {dim}')

    plt.xticks(np.arange(min(history_df['epoch']), max(history_df['epoch'])+1, 1.0))
    plt.title(f'{model_type}: Validation Micro-F1 Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('Micro-F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_folder}/{model_type}_f1_scores.png')
    plt.close() # Close the figure to free up memory

    plt.figure(figsize=(15, 8))
    # Plot AUC Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_auc'], label=f'AUC Dim {dim}')
    # set x-axis labels to be integers
    plt.xticks(np.arange(min(history_df['epoch']), max(history_df['epoch'])+1, 1.0))
    plt.title(f'{model_type}: Validation ROC-AUC Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_folder}/{model_type}_auc_scores.png')
    plt.close()

    plt.figure(figsize=(15, 8))
    # Plot P@5 Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_p@5'], label=f'P@5 Dim {dim}')

    plt.xticks(np.arange(min(history_df['epoch']), max(history_df['epoch'])+1, 1.0))
    plt.title(f'{model_type}: Validation Precision@5 Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('Precision@5 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_folder}/{model_type}_p@5_scores.png')
    plt.close()

    print(f"Plots saved to folder: {plots_folder}")
    
    save_file(history_list, f'{save_folder}/{model_type}_training_history')    

def convert_numpy_floats(item):
    if isinstance(item, (np.float32, np.float64)):
        return float(item)
    if isinstance(item, (np.int32, np.int64)):
        return int(item)
    return item

def save_file(history, file_name):
    def default(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_floats(elem) for elem in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy_floats(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(f'{file_name}.json', 'w') as f:
        json.dump(history, f, indent=4, default=default)


def plot_test_metrics(test_metrics, config, save_folder="./logs"):
    """
    Comparing the Test Metrics for the baseline and the trained model.
    Generates a single figure with grouped bar charts for F1, AUC, and P@5.
    """
    model_type = config.model_type
    baseline = test_metrics.get('baseline_performance', {})
    trained = test_metrics.get('trained_performance', {})
    
    if not baseline or not trained:
        print("Comparison requires both 'baseline_performance' and 'trained_performance' in test_metrics.")
        return

    os.makedirs(save_folder, exist_ok=True)
    
    # Dimensions
    dims = config.nesting_dims
    
    # Metric types and their keys in the dictionary
    metrics_map = {
        'F1 Score': 'f1',
        'ROC-AUC': 'auc',
        'Precision@5': 'p@5'
    }
    
    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    x = np.arange(len(dims))  # Label locations
    width = 0.35  # Bar width

    for idx, (metric_name, suffix) in enumerate(metrics_map.items()):
        ax = axes[idx]
        
        baseline_vals = []
        trained_vals = []
        
        for dim in dims:
            # Construct keys like "val/dim_64_f1"
            base_key = f"val/dim_{dim}_{suffix}"
            train_key = f"val/dim_{dim}_{suffix}"
            
            baseline_vals.append(baseline.get(base_key, 0.0))
            trained_vals.append(trained.get(train_key, 0.0))
        
        rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#1f77b4', alpha=0.8)
        rects2 = ax.bar(x + width/2, trained_vals, width, label='Trained', color='#ff7f0e', alpha=0.8)
        
        # Add labels, title, and custom x-axis tick labels
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Dim {d}" for d in dims])
        ax.set_xlabel('Matryoshka Dimension')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Helper to label bars with values
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=45, fontsize=8)

        autolabel(rects1)
        autolabel(rects2)
        
    fig.tight_layout()
    save_file(test_metrics, f'{save_folder}/{config.model_type}_test_performance')
    plots_folder = os.path.join(save_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    save_path = f"{plots_folder}/{config.model_type}_test_comparison_all_metrics.png"
    plt.savefig(save_path)
    plt.close()
        
    print(f"Test comparison plot saved to {save_path}")


def load_model(checkpoint_path, device=None):
    """
    Only use thus function after ensuring you are saving only the trained layers, not the whole model with the frozen backbone.
    Loads a saved model checkpoint, handling the initialization of frozen layers correctly.
    """
    from src.models import HMPLMICD
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Reconstruct config object
    config = argparse.Namespace(**config_dict)
    
    # Ensure device is correct
    config.device = device

    # Initialize model (Backbone loaded from pretrained/cache)
    # We pass the model_name from config if available, else use default in class
    model_name = getattr(config, 'model_name', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    model = HMPLMICD(config, pretrained_model_name=model_name)
    
    # Load state dict
    # strict=False allows loading only the trained layers (excluding frozen backbone)
    # The frozen backbone weights remain as initialized (pretrained)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"Model loaded. Missing keys (expected for frozen layers): {len(missing_keys)}")
    # Optional: Check if missing keys are indeed frozen ones
    # frozen_keys = [n for n, p in model.named_parameters() if not p.requires_grad]
    
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('epoch'), checkpoint.get('best_f1')
