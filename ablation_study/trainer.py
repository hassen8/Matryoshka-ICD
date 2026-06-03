import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import losses, util
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_retrieval(model, val_dataset, label_texts, config, label2id):
    model.eval()
    logger = config.logger
    text_col = config.text_col

    logger.info("Starting retrieval evaluation")

    # 1. Pre-compute embeddings for all unique labels
    # label_texts is a list of descriptions corresponding to IDs 0...num_labels-1
    label_embeddings = model.encode(label_texts, convert_to_tensor=True, show_progress_bar=False) # [num_labels, hidden_size]

    # 2. Prepare queries and calculate similarities
    queries = val_dataset[text_col].tolist()
    
    # We multi-hot encode targets identically to how it's done for PLM classification
    targets_list = []
    num_labels = len(label2id)
    
    for labels in val_dataset['leaf_doc']:
        target_vec = np.zeros(num_labels)
        for lbl in labels:
            if lbl in label2id:
                target_vec[label2id[lbl]] = 1.0
        targets_list.append(target_vec)
        
    all_targets = np.vstack(targets_list)
    
    # 3. Get query embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    
    metrics = {}
    logger.info("\n--- Validation Results (Matryoshka Retrieval) ---")
    
    for dim in config.nesting_dims:
        # Truncate to current Matryoshka dimension
        q_emb = query_embeddings[:, :dim]
        l_emb = label_embeddings[:, :dim]
        
        # Calculate Cosine Similarity: [num_queries, num_labels]
        sim_scores = util.cos_sim(q_emb, l_emb).cpu().numpy()
        
        # --- THRESHOLD TUNING ---
        best_threshold = 0.5
        best_f1 = 0.0

        # Sweep thresholds from 0.01 to 0.99 for cosine similarities
        for t in np.arange(0.01, 1.00, 0.01):
            preds_binary_t = (sim_scores > t).astype(int)
            f1_t = f1_score(all_targets, preds_binary_t, average='micro')
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

        micro_f1 = best_f1

        try:
            roc_auc = roc_auc_score(all_targets, sim_scores, average='micro')
        except ValueError:
            roc_auc = 0.0
            
        # Calculate Precision @ 5
        k = 5
        top_k_indices = np.argsort(sim_scores, axis=1)[:, -k:]
        top_k_targets = np.take_along_axis(all_targets, top_k_indices, axis=1)
        p_at_5 = np.mean(np.sum(top_k_targets, axis=1) / k)

        metrics[f"val/dim_{dim}_f1"] = micro_f1
        metrics[f"val/dim_{dim}_auc"] = roc_auc
        metrics[f"val/dim_{dim}_p@5"] = p_at_5

        logger.info(f"Dim {dim}: Micro-F1 = {micro_f1:.4f} (Thresh: {best_threshold:.2f}) | ROC-AUC = {roc_auc:.4f} | P@5 = {p_at_5:.4f}")

        # --- DEBUG STEP: CHECK SIMILARITIES ---
        if dim == config.nesting_dims[-1]:
            logger.info(f"DEBUG: Max Sim: {np.max(sim_scores):.4f}")
            logger.info(f"DEBUG: Mean Sim: {np.mean(sim_scores):.4f}")

    return metrics

def train_retrieval_model(model, train_loader, val_dataset, label_texts, optimizer, config, label2id):
    logger = config.logger
    wandb = config.wandb
    model.to(config.device)
    
    saving_config = {k: v for k, v in vars(config).items() if k not in ['wandb', 'logger']}

    wandb.init(
        project=config.args.wandb_project,
        config=saving_config,
        group="Matryoshka_Query_Based",
        name=f"{config.model_type}_run_{wandb.util.generate_id()}"
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

    save_dir = config.checkpoint_dir+"/matryoshka_retrieval_model"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = save_dir

    history = []
    best_val_auc = 0.0

    for epoch in range(config.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        model.train()
        total_loss = 0

        # train_loader now returns a list of dictionaries containing batched tokenized data natively: [query_features, positive_features]
        # This prevents CPU tokenization bottleneck during training.
        
        for features in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # device transfer
            for f in features:
                for key in f:
                    f[key] = f[key].to(config.device)
            
            # sentence_transformers MatryoshkaLoss
            # It expects features, labels (ignored for MNR)
            # Dummy labels, size matching batch_size
            batch_size = features[0]['input_ids'].size(0)
            labels = torch.zeros(batch_size, dtype=torch.float).to(config.device)
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
