import numpy as np


def build_label_hierarchy_map(train_df, val_df, test_df, label2id):
    """
    Build a mapping from label_id (0..num_labels-1) to icd_sequence string.
    Used by cluster matching to look up hierarchy paths for prediction indices.

    Returns: list of length num_labels, where result[label_id] = "520-579|570-579|572|572.3"
    """
    label_to_seq = {}
    for df in [train_df, val_df, test_df]:
        for _, row in df.iterrows():
            for code, seq in zip(row['icd_codes'], row['icd_sequences']):
                if code in label2id and label2id[code] not in label_to_seq:
                    label_to_seq[label2id[code]] = seq
    num_labels = len(label2id)
    result = [None] * num_labels
    for label_id, seq in label_to_seq.items():
        result[label_id] = seq
    return result


def compute_ir_metrics(probs, targets, k_values):
    """
    Compute Information Retrieval metrics from multi-label probability vectors.

    Unlike classification metrics (F1, AUC) which require threshold tuning,
    IR metrics rank codes by probability and evaluate ranking quality.

    Args:
        probs:   [N, C] sigmoid probabilities
        targets: [N, C] binary multi-hot ground truth
        k_values: list of int, cutoffs (e.g. [1, 3, 5, 10, 20, 50, 100])

    Returns:
        dict of {metric_name: {k: float}}
            recall@k:  fraction of ground-truth codes found in top-k
            precision@k: fraction of top-k predictions that are correct
            ndcg@k:  Normalized Discounted Cumulative Gain (position-weighted)
            mrr@k:   Mean Reciprocal Rank of first correct code
            map@k:   Mean Average Precision across all recall levels
    """
    N, C = probs.shape
    # Rank all codes by descending probability
    ranked_indices = np.argsort(-probs, axis=1)  # [N, C]
    max_k = max(k_values)
    top_k = ranked_indices[:, :max_k]  # [N, max_k], only need up to max_k

    # Accumulate per-sample values, then average
    metrics = {
        'recall': {k: [] for k in k_values},
        'precision': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values},
        'mrr': {k: [] for k in k_values},
        'map': {k: [] for k in k_values},
    }

    for i in range(N):
        target_indices = np.where(targets[i] > 0.5)[0]
        n_targets = len(target_indices)
        if n_targets == 0:
            continue  # skip samples with no ground-truth labels

        preds = top_k[i]  # [max_k]

        for k in k_values:
            preds_k = preds[:k]                # top-k predictions
            matches = np.isin(preds_k, target_indices)  # bool [k]
            n_matches = matches.sum()
            actual_k = len(preds_k)

            # Recall@k: fraction of ground-truth found in top-k
            recall = n_matches / n_targets
            metrics['recall'][k].append(recall)

            # Precision@k: fraction of top-k that are correct
            precision = n_matches / actual_k
            metrics['precision'][k].append(precision)

            # NDCG@k: position-weighted relevance, normalized by ideal ranking
            dcg = 0.0
            for rank, hit in enumerate(matches):
                if hit:
                    dcg += 1.0 / np.log2(rank + 2)  # rank 0 → log2(2)=1
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(actual_k, n_targets)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics['ndcg'][k].append(ndcg)

            # MRR@k: reciprocal rank of the first correct code
            mrr = 0.0
            for rank, hit in enumerate(matches):
                if hit:
                    mrr = 1.0 / (rank + 1)
                    break
            metrics['mrr'][k].append(mrr)

            # MAP@k: average of precision at each correct-recall point
            n_correct = 0
            sum_prec = 0.0
            for rank, hit in enumerate(matches):
                if hit:
                    n_correct += 1
                    sum_prec += n_correct / (rank + 1)
            ap = sum_prec / min(actual_k, n_targets) if min(actual_k, n_targets) > 0 else 0.0
            metrics['map'][k].append(ap)

    # Average across samples
    result = {}
    for metric_name, kdict in metrics.items():
        result[metric_name] = {k: float(np.mean(vals)) if vals else 0.0 for k, vals in kdict.items()}

    return result


def compute_cluster_matching(probs, targets, label_hierarchies, k_values):
    """
    Compute hierarchical cluster matching metrics.

    For each hierarchy level i (1=chapter, 2=block, 3=3-digit, 4=leaf, ...),
    checks whether the prefix of a top-k prediction matches any ground-truth
    prefix at that level. This gives partial credit for getting the general
    category right even if the specific leaf is wrong.

    Args:
        probs:   [N, C] sigmoid probabilities
        targets: [N, C] binary multi-hot ground truth
        label_hierarchies: [C] list of pipe-separated hierarchy strings
                           indexed by label_id, e.g. ["520-579|570-579|572|572.3"]
        k_values: list of int cutoffs

    Returns:
        dict of {f'cluster_match@{k}_{level}': float}
    """
    N, C = probs.shape
    ranked_indices = np.argsort(-probs, axis=1)  # [N, C]

    # Pre-compute hierarchy level lists for all label indices
    label_prefixes = []
    for seq_str in label_hierarchies:
        label_prefixes.append(seq_str.split('|') if seq_str else [])

    # Maximum number of hierarchy levels across all codes
    max_depth = max(len(p) for p in label_prefixes if p)

    cluster_metrics = {}

    for k in k_values:
        for level in range(1, max_depth + 1):
            matches = 0
            total = 0
            for i in range(N):
                target_indices = np.where(targets[i] > 0.5)[0]
                pred_indices = ranked_indices[i, :min(k, C)]

                # Build prefix sets for predictions and ground truth at this level
                pred_prefixes = set()
                for idx in pred_indices:
                    pref = label_prefixes[idx]
                    if len(pref) >= level:
                        pred_prefixes.add(tuple(pref[:level]))

                gt_prefixes = set()
                for idx in target_indices:
                    pref = label_prefixes[idx]
                    if len(pref) >= level:
                        gt_prefixes.add(tuple(pref[:level]))

                # Count matches: how many ground-truth prefixes appear in predictions
                if gt_prefixes:
                    total += len(gt_prefixes)
                    matches += len(gt_prefixes & pred_prefixes)

            match_rate = matches / total if total > 0 else 0.0
            cluster_metrics[f'cluster_match@{k}_{level}'] = match_rate

    return cluster_metrics


def compute_all_metrics(probs, targets, label_hierarchies, k_values):
    """
    Convenience function: compute IR metrics + cluster matching,
    returning a flat dict of metric_name -> float.
    """
    ir = compute_ir_metrics(probs, targets, k_values)
    cluster = compute_cluster_matching(probs, targets, label_hierarchies, k_values)

    combined = {}
    for metric, kdict in ir.items():
        for k, v in kdict.items():
            combined[f'{metric}@{k}'] = v
    combined.update(cluster)
    return combined
