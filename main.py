import os
import torch
import torch.optim as optim
from transformers import AutoTokenizer
import wandb
import logging

from src.configs import parse_args, ModelConfig
from src.data import get_data_loaders
from src.models import HMPLMICD
from src.loss import StandardMRLLoss, HierarchicalMRLLoss
from src.trainer import train_model, evaluate_model
from src.utils import plot_and_save_metrics, save_file
# Setup Logging
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 1. Parse Args
    args = parse_args()
    
    # 2. Setup Device and wandb
    device = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb_api_key = os.getenv("WANDB_API_KEY") 
    wandb.login(key=wandb_api_key)
    logger.info(f"Using device: {device}")
    

    # 3. Load Tokenizer & Data
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Preparing Data...")

    train_loader, val_loader, test_loader, label2id, num_labels = get_data_loaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        text_col='query', #To be clarified later on 
        label_col='leaf_doc'# remind me to silve this problem l8r on
    )

    # 4. Initialize Config & Model
    config = ModelConfig(args, num_labels)
    config.device = device
    config.wandb = wandb
    config.logger = logger
    
    model = HMPLMICD(config, pretrained_model_name=args.model_name)
    model.to(device)

    # create the baseline performance
    test_metrics = {}
    test_metrics['baseline_performance'] = evaluate_model(model, test_loader, config)
    
    # 5. Optimizer & Loss
    importance_weights = {64: 1.0, 128: 1.0, 256: 1.0, 768: 1.0} # Can also be moved to args
    criterion = StandardMRLLoss(importance_weights=importance_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 6. Run Training
    logger.info("Starting Training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, config)
    plot_and_save_metrics(history,config)
    
    #7. Run the testset
    test_metrics['trained_performance'] = evaluate_model(model, test_loader, config)
    save_file(test_metrics, 'test_performance')
    
if __name__ == "__main__":
    main()