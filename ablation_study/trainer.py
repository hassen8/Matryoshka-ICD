import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import losses
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_retrieval(model, val_dataset, label_texts, config):
    model.eval()
    logger = config.logger

    logger.info("Starting retrieval evaluation")

    # 1. Pre-compute embeddings for all unique labels
    # label_texts is a list of descriptions corresponding to IDs 0...num_labels-1
    # sentence_transformers.encode automatically handles batching and device
    label_embeddings = model.encode(label_texts, convert_to_tensor=True, show_progress_bar=False) # [num_labels, hidden_size]

    # 2. Prepare queries and calculate similarities
    queries = val_dataset['report_text'].tolist()
    # We multi-hot encode targets identically to how it's done for PLM classification
    targets_list = []
    for labels in val_dataset['leaf_doc']:
        # labels are integers representing original IDs. We need contiguous ones?
        # Actually in data.py label_col='leaf_doc' has the original string IDs, we mapped them.
        # We need to ensure targets match the 0..num_labels-1 indices.
        # Assuming config has label2id or val_dataset already contains mapped list.
        # It's better to pass label2id.
        pass
    
    # The actual matching mechanism:
    metrics = {}
    return metrics

def train_retrieval_model(model, train_loader, val_dataset, label_texts, optimizer, config, label2id):
    logger = config.logger
    wandb = config.wandb
    model.to(config.device)
    
    saving_config = {k: v for k, v in vars(config).items() if k not in ['wandb', 'logger']}

    wandb.init(
        project=config.args.wandb_project,
        config=saving_config,
        group="Matryoshka_Retrieval_Ablation",
        name=f"run_retrieval_{wandb.util.generate_id()}"
    )

    baseline_metrics = evaluate_retrieval(model, val_dataset, label_texts, config, label2id)
    baseline_metrics['epoch'] = 0
    wandb.log(baseline_metrics)

    # Loss Definition
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    matryoshka_loss = losses.MatryoshkaLoss(
        model=model,
        loss=train_loss,
        matryoshka_dims=config.nesting_dims
    )

    save_dir = config.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_retrieval_model.pt")

    history = []
    best_val_auc = 0.0

    for epoch in range(config.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        model.train()
        total_loss = 0

        # Note: train_loader returns batches of InputExample objects
        # To use with standard PyTorch we need sentence_transformers.models features,
        # but the easiest way is using sentence_transformers' built in fit if we can,
        # or doing a manual loop. We'll do a manual loop so we keep wandb logging identical.
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # batch is a list of InputExample
            # sentence_transformers loss functions expect inputs as a list of dictionaries containing tokenized data
            features = model.tokenize([example.texts for example in batch])
            
            # features is a list of dictionaries [query_features, positive_features]
            # device transfer
            for f in features:
                for key in f:
                    f[key] = f[key].to(config.device)
            
            # sentence_transformers MatryoshkaLoss
            # It expects features, labels (ignored for MNR)
            # Dummy labels
            labels = torch.zeros(len(batch)).to(config.device)
            loss_value = matryoshka_loss(features, labels)
            
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train/loss": avg_train_loss}, step=epoch)

        val_metrics = evaluate_retrieval(model, val_dataset, label_texts, config, label2id)
        val_metrics['epoch'] = epoch + 1
        val_metrics['train_loss'] = avg_train_loss
        history.append(val_metrics)
        wandb.log(val_metrics, step=epoch)

        target_metric = val_metrics.get(f"val/dim_{config.nesting_dims[-1]}_f1", 0)
        if target_metric >= best_val_auc:
            logger.info(f"🚀 Performance improved. Saving model...")
            best_val_auc = target_metric
            model.save(best_model_path) # SentenceTransformer save

    wandb.finish()
    return baseline_metrics, history
