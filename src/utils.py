import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_and_save_metrics(history_list, config, save_folder="./logs"):
    """
    Generates and saves F1, AUC, and P@5 plots from training history.

    Args:
        history_list (list): A list of dictionaries containing epoch-wise training metrics.
        config (ModelConfig): The configuration object containing nesting_dims.
        save_folder (str): The name of the folder to save the plots in.
    """
    history_df = pd.DataFrame(history_list)

    # Create the directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 8))
    # Plot F1 Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_f1'], label=f'F1 Dim {dim}')

    plt.title('Validation Micro-F1 Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('Micro-F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'f1_scores.png'))
    plt.close() # Close the figure to free up memory

    plt.figure(figsize=(15, 8))
    # Plot AUC Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_auc'], label=f'AUC Dim {dim}')

    plt.title('Validation ROC-AUC Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'auc_scores.png'))
    plt.close()

    plt.figure(figsize=(15, 8))
    # Plot P@5 Scores
    for dim in config.nesting_dims:
        plt.plot(history_df['epoch'], history_df[f'val/dim_{dim}_p@5'], label=f'P@5 Dim {dim}')

    plt.title('Validation Precision@5 Score per Epoch for each Matryoshka Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('Precision@5 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'p@5_scores.png'))
    plt.close()

    print(f"Plots saved to folder: {save_folder}")
    
    save_file(history_list, 'training_history')    

def save_file(history, file_name):

    with open(f'{file_name}.json', 'w') as f:
        json.dump(history, f, indent=4)