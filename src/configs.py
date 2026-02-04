import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Matryoshka ICD Coding")
    
    # Data paths
    parser.add_argument("--data_path", type=str, required=False, help="Path to the CSV dataset")
    parser.add_argument("--text_column", type=str, default="query", help="Column to be used as text to tokenize")
    parser.add_argument("--label_column", type=str, default="leaf_doc", help="Column to be used as labels")
    parser.add_argument("--model_name", type=str, default="thomas-sounack/BioClinical-ModernBERT-base")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--hierarchical_mode", type=bool, default=False)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    
    # MRL Configs
    parser.add_argument("--nesting_dims", type=int, nargs="+", default=[64, 128, 256, 768])
    
    # Misc
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_csv_dir", type=str, default="./data")
    parser.add_argument("--wandb_api", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="Matriyoshka")
    
    return parser.parse_args()

class ModelConfig:
    def __init__(self, args, num_labels):
        self.args = args
        self.num_labels = num_labels
        self.hidden_size = args.hidden_size
        self.nesting_dims = sorted(args.nesting_dims)
        self.hierarchical_mode = args.hierarchical_mode
        self.dropout_prob = args.dropout_prob
        self.batch_size = args.batch_size
        self.freeze_backbone = args.freeze_backbone
        self.lr = args.lr
        self.epochs = args.epochs
        self.max_len = args.max_len
        self.checkpoint_dir = args.checkpoint_dir
        self.model_name = args.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Mapping hierarchy levels to dimension indices for Step 2
        # Level 0 (Chapters) -> Index 0 (Dim 64)
        # Level 1 (Blocks)   -> Index 1 (Dim 128)
        # Level 2 (3-Digit)  -> Index 2 (Dim 256)
        # Level 3 (Leaf)     -> Index 3 (Dim 768)
        self.level_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3
        }
