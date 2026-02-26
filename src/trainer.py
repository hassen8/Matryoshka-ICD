import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(model, val_loader, config):
    model.eval()
    logger = config.logger
    model_type = config.model_type

    # print(f"Starting evaluation")
    logger.info(f"Starting evaluation")

    # We want to track metrics for EACH dimension separately
    # structure: {64: {'preds': [], 'targets': []}, 128: ...}
    results_storage = {dim: {'preds': [], 'targets': []} for dim in config.nesting_dims}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            targets = batch['labels'].to(config.device)

            # Forward
            logits_dict = model(input_ids, attention_mask)

            # Store predictions for each dimension
            for dim, logits in logits_dict.items():
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true = targets.cpu().numpy()

                results_storage[dim]['preds'].append(probs)
                results_storage[dim]['targets'].append(y_true)

    metrics = {}
    # print("\n--- Validation Results (Matryoshka) ---")
    logger.info("\n--- Validation Results (Matryoshka) ---")
    for dim in config.nesting_dims:
        # Concatenate all batches
        all_preds = np.vstack(results_storage[dim]['preds'])
        all_targets = np.vstack(results_storage[dim]['targets'])

        # --- REAL-WORLD THRESHOLD TUNING ---
        # Highly imbalanced datasets push probabilities low. 0.5 is a bad assumption.
        # We sweep thresholds to find the optimal global threshold for Micro-F1.
        best_threshold = 0.5
        best_f1 = 0.0

        # Sweep thresholds from 0.01 to 0.50 in steps of 0.01
        for t in np.arange(0.01, 0.51, 0.01):
            preds_binary_t = (all_preds > t).astype(int)
            f1_t = f1_score(all_targets, preds_binary_t, average='micro')
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

        micro_f1 = best_f1

        # Calculate ROC-AUC (weighted or micro)
        try:
            roc_auc = roc_auc_score(all_targets, all_preds, average='micro')
        except ValueError:
            roc_auc = 0.0 # Handle edge cases where a class is never present

        # --- NEW: CALCULATE PRECISION @ 5 ---
        k = 5
        # 1. Get indices of the top k probabilities for each row
        # argsort sorts ascending, so we take the last k columns
        top_k_indices = np.argsort(all_preds, axis=1)[:, -k:]

        # 2. Extract the true targets at those specific indices
        # np.take_along_axis allows us to select values using the indices we just found
        top_k_targets = np.take_along_axis(all_targets, top_k_indices, axis=1)

        # 3. Calculate precision: (number of correct hits in top k) / k
        # We sum across axis 1 to get hits per sample, then mean across all samples
        p_at_5 = np.mean(np.sum(top_k_targets, axis=1) / k)


        metrics[f"val/dim_{dim}_f1"] = micro_f1
        metrics[f"val/dim_{dim}_auc"] = roc_auc
        metrics[f"val/dim_{dim}_p@5"] = p_at_5

        # print(f"Dim {dim}: Micro-F1 = {micro_f1:.4f} (Thresh: {best_threshold:.2f}) | ROC-AUC = {roc_auc:.4f} | P@5 = {p_at_5:.4f}")
        logger.info(f"Dim {dim}: Micro-F1 = {micro_f1:.4f} (Thresh: {best_threshold:.2f}) | ROC-AUC = {roc_auc:.4f} | P@5 = {p_at_5:.4f}")
        
        # --- DEBUG STEP: CHECK PROBABILITIES ---
        if dim == 768: # Just check the largest dim for brevity
            # print(f"DEBUG: Max Prob: {np.max(all_preds):.4f}")
            # print(f"DEBUG: Mean Prob: {np.mean(all_preds):.4f}")
            logger.info(f"DEBUG: Max Prob: {np.max(all_preds):.4f}")
            logger.info(f"DEBUG: Mean Prob: {np.mean(all_preds):.4f}")

    return metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    logger = config.logger
    wandb = config.wandb
    model.to(config.device)
    model_type = config.model_type

    # Filter out non-serializable objects for wandb config
    saving_config = {k: v for k, v in vars(config).items() if k not in ['wandb', 'logger']}

    # Initialize wandb run
    wandb.init(
        project=config.args.wandb_project,
        config= saving_config,
        group="Matryoshka_Query_Based",
        name=f"{model_type}_run_{wandb.util.generate_id()}"
    )
    # record baseline performance
    baseline_metrics = evaluate_model(model, val_loader, config)
    baseline_metrics['epoch'] = 0
    wandb.log(baseline_metrics)

    # 2. Setup Checkpointing
    best_val_auc = 0.0
    save_dir = config.checkpoint_dir + "/matryoshka_plm_model"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_hmplm_model.pt")

    history = [] # To store metrics per epoch

    for epoch in range(config.epochs):
        # print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        logger.info(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            targets = batch['labels'].to(config.device) # Shape: [batch, num_labels]

            optimizer.zero_grad()

            # 1. Forward Pass
            # Returns dict: {64: logits_64, 128: logits_128, ...}
            logits_dict = model(input_ids, attention_mask)

            # 2. Compute Loss (StandardMRLLoss)
            loss = criterion(logits_dict, targets)

            # 3. Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        # print(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train/loss": avg_train_loss}, step=epoch)


        # --- EVALUATION PHASE (Matryoshka Aware) ---
        val_metrics = evaluate_model(model, val_loader, config)
        val_metrics['epoch'] = epoch + 1
        val_metrics['train_loss'] = avg_train_loss
        history.append(val_metrics)

        # Log all validation metrics to wandb
        wandb.log(val_metrics, step=epoch)

        # Use the largest dimension's F1 for early stopping/best model saving
        target_metric = val_metrics[f"val/dim_{config.nesting_dims[-1]}_f1"]
        if epoch == 0:
            best_val_auc = target_metric

        if target_metric > best_val_auc:
            # print(f"🚀 Performance improved ({best_val_auc:.4f} -> {target_metric:.4f}). Saving model...")
            logger.info(f"🚀 Performance improved ({best_val_auc:.4f} -> {target_metric:.4f}). Saving model...")
            best_val_auc = target_metric

            # Save the State Dict (Standard for custom models) 
            # Save only the trained layers (filtering out frozen layers)
            # state_dict = model.state_dict()
            # trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
            # filtered_state_dict = {k: v for k, v in state_dict.items() if k in trainable_params}

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_auc,
                'config': saving_config
            }, best_model_path)

            # Log model artifact to wandb
            artifact = wandb.Artifact(name="best_model", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    return baseline_metrics, history
