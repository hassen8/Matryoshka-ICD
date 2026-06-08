import os
import torch
import torch.optim as optim
from transformers import AutoTokenizer
import wandb
import logging

from src.configs import parse_args, ModelConfig
from src.data import get_data_loaders
from src.models import HMPLMICD, StandardAttentionICD, plm_icd
from src.loss import StandardMRLLoss, HierarchicalMRLLoss
from src.trainer import train_model, evaluate_model
from src.utils import plot_and_save_metrics, save_file, plot_test_metrics
from ablation_study.models import get_retrieval_model
from ablation_study.trainer import train_retrieval_model, evaluate_retrieval

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
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb_api_key = os.getenv("WANDB_API_KEY") 
    wandb.login(key=wandb_api_key)
    logger.info(f"Using device: {device}")
    

    # 3. Load Tokenizer & Data
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Determine models to run
    models_to_run = ['laa', 'standard_attn', 'retrieval', 'plm_icd'] if args.model_type == 'all' else [args.model_type]
    
    for current_model_type in models_to_run:
        logger.info(f"\n{'='*50}\nStarting execution for {current_model_type}\n{'='*50}")

        # 1. Fetch data loaders specific to this model type and set up config
        train_loader, val_loader, test_loader, label_data, num_labels, retrieval_descriptions = get_data_loaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            text_col=args.text_column,
            label_col=args.label_column,
            output_dir=args.output_csv_dir,
            dataset_dir=args.dataset_dir,
            model_type=current_model_type
        )

        config = ModelConfig(args, num_labels)
        config.device = device
        config.wandb = wandb
        config.logger = logger
        config.model_type = current_model_type
        
        # 2. Initialize Model
        if config.model_type == 'laa':
            model = HMPLMICD(config, pretrained_model_name=args.model_name)
        elif config.model_type == 'standard_attn':
            model = StandardAttentionICD(config, pretrained_model_name=args.model_name)
        elif config.model_type == 'retrieval':
            model = get_retrieval_model(pretrained_model_name=args.model_name, config=config)
        elif config.model_type == 'plm_icd':
            model = plm_icd(config, pretrained_model_name=args.model_name)
            
        # 3. Dispatch to appropriate trainer
        if config.model_type in ['laa', 'standard_attn', 'plm_icd']:
            model.to(device)

            # create the baseline performance
            test_metrics = {}
            test_metrics['baseline_performance'] = evaluate_model(model, test_loader, config)
            
            # 5. Optimizer & Loss
            importance_weights = {64: 1.0, 128: 1.0, 256: 1.0, 768: 1.0} # Can also be moved to args
            criterion = StandardMRLLoss(importance_weights=importance_weights)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)

            # 6. Run Training
            logger.info(f"Starting Training for {config.model_type}...")
            baseline, history = train_model(model, train_loader, val_loader, criterion, optimizer, config)
            plot_and_save_metrics(history, baseline, config, save_folder=f'logs/{current_model_type}')
            
            #7. Run the testset
            test_metrics['trained_performance'] = evaluate_model(model, test_loader, config)
            plot_test_metrics(test_metrics, config, save_folder=f'logs/{current_model_type}')
        else:
            # 5. Optimizer for Retrieval
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            
            val_dataset = val_loader 
            test_dataset = test_loader 
            label_texts = retrieval_descriptions
            
            # 6. Run Training
            logger.info("Starting Training for retrieval Matryoshka...")
            baseline, history = train_retrieval_model(
                model=model, 
                train_loader=train_loader, 
                val_dataset=val_dataset,
                label_texts=label_texts,
                optimizer=optimizer, 
                config=config, 
                label2id=label_data
            )
            plot_and_save_metrics(history, baseline, config, save_folder=f'logs/{current_model_type}')
            
            # 7. Test
            test_metrics = {}
            test_metrics['baseline_performance'] = baseline
            test_metrics['trained_performance'] = evaluate_retrieval(model, test_dataset, label_texts, config, actual_label2id)
            plot_test_metrics(test_metrics, config, save_folder=f'logs/{current_model_type}')

    
if __name__ == "__main__":
    main()