# Project Report: MRL-Enhanced ICD Coding System

## 1. Executive Summary

This project implements an automated ICD (International Classification of Diseases) coding system using **Matryoshka Representation Learning (MRL)** and **Label-Aware Attention (LAA)**. The system is designed to predict ICD diagnosis codes from clinical records (MIMIC-CXR radiology reports) with a focus on **efficiency** and **adaptability**. By leveraging MRL, the model learns embeddings that are informative at multiple granularities (dimensions), allowing for flexible deployment scenarios (e.g., using smaller embeddings for faster retrieval/inference without retraining).

## 2. Theoretical Background

### 2.1. Automatic ICD Coding
Automatic ICD coding is a multi-label text classification problem where a medical document $D$ is associated with a set of codes $C = \{c_1, c_2, \dots, c_k\}$ from a very large label space (thousands of possible ICD codes). The challenge lies in the long-tailed distribution of codes and the need to capture specific clinical evidence from long documents.

### 2.2. Label-Aware Attention (LAA)
Standard text classification aggregates document tokens into a single vector (e.g., `[CLS]` token). However, different codes require different parts of the document as evidence. **Label-Aware Attention** addresses this by learning a unique attention mechanism for *each* label.
- For a label $l$, the model computes a specific weighted sum of value vectors from the document tokens.
- This results in $|L|$ distinct document representations, one for each potential label.

### 2.3. Matryoshka Representation Learning (MRL)
Standard representation learning models an object as a fixed-size vector $v \in \mathbb{R}^d$. If downstream tasks require smaller vectors (for storage or speed), the model must be retrained or projected, losing information.
**MRL** trains a single vector $v$ such that its first $k$ dimensions (where $k < d$) are also high-quality representations.
- It enforces a nested structure: $v_{1:64}$ is a valid embedding, $v_{1:128}$ is a better one, and $v_{1:768}$ is the best.
- This is achieved by calculating the loss at multiple "nesting dimensions" (e.g., 64, 128, 256, 768) simultaneously during training.

## 3. System Architecture

### 3.1. Inputs and Data Processing
- **Source**: MIMIC-CXR Dataset (Radiology Reports).
- **Preprocessing**: 
    - Extracts `FINDINGS` and `IMPRESSION` sections from the report files in the MIMIC-CXR
    - Adds it to a dataframe containing the `query` and all other information in the `mimicxr_parsed_ds.jsonl` file per patient.
    - Tokenizes text using a BERT-based tokenizer.
    - Aggregates labels by `reportid` into a multi-hot binary vector.
- **Data Loader**: For standard models, it returns `input_ids`, `attention_mask`, and `labels`. For retrieval models, it extracts anchor-positive text pairs `(query, description)` by matching the target `leaf_icd` to its semantic description from `icd_descr_map.json`.
- Dataset is split into training, validation, and test sets.
- Training treats the problem as a multi-label classification problem (or contrastive learning for the retrieval ablation).

### 3.1.1 Dataset Statistics
Based on the analysis of the provided `mimicxr_parsed_ds.jsonl` file, the duplicates were collapsed to represent a multi-label classification problem, which inturn gives us these unique values:
- **Total Unique Reports**: 5,827
- **Training Set**: 5,749
- **Validation Set**: 37
- **Test Set**: 41

The dataset is considerably small for training the model, though this lays the groundwork for future work when the larger dataset is available. Also for this reason I chose to freeze the backbone encoder and only train the label-aware attention and classifier layers.

### 3.2. Model Architecture (`HMPLMICD`)
The model is defined in `src/models.py` and consists of three main stages:

1.  **Backbone Encoder**:
    - Uses `thomas-sounack/BioClinical-ModernBERT-base` (or PubMedBERT).
    - Encodes specific tokens into contextual embeddings $H \in \mathbb{R}^{L \times 768}$.
    - *Note*: The backbone is frozen by default (`freeze_backbone=True`).

2.  **Label-Aware Attention (`LabelAwareAttention`)**:
    - Projects hidden states $H$ to a latent space using $V$ and computes attention with matrix $W$ (mapping to `num_labels`).
    - **Decoupled Attention**: Crucially, the attention mechanism is restricted to ONLY the base Matryoshka dimension (e.g., 64). This decouples the attention computation from higher dimensions, ensuring that the subset isn't entangled with the full 768-dim space and preserving the Matryoshka gradient flow.
    - Outputs a tensor of shape `[batch, num_labels, hidden_size]`.
    - Each vector $v_l$ represents the document specifically for detecting label $l$.

3.  **Matryoshka Classifier (`MatryoshkaClassifier`)**:
    - A label-specific projection layer with a unique, decoupled bias per label.
    - **Decoupled Classification**: Instead of a shared scalar bias that can squash higher-dimension probabilities, it calculates a dot product over the slicing dimension independently and adds the label's own bias.
    - **Weight Tying**: Instead of separate heads for each dimension, it *slices* the weights of the linear layer. 

    - For a nesting dimension $d$ (e.g., 64):
        - It takes the first $d$ features of the document representation.
        - It takes the first $d$ weights of the classifier.
        -  Computes the detached dot product and adds the decoupled bias.

### 3.3. Loss Function (`StandardMRLLoss`)
The loss function is a weighted sum of Binary Cross-Entropy (BCE) losses calculated at each nesting dimension.
Given nesting dimensions $M = \{64, 128, 256, 768\}$:

$$ \mathcal{L}_{total} = \sum_{m \in M} c_m \cdot \mathcal{L}_{BCE}(\text{Logits}_{m}, Y) $$

where $\text{Logits}_m$ are the predictions using only the first $m$ dimensions of the embeddings.
This forces the model to pack the most critical information into the earliest dimensions.

## 4. Implementation Details

- **Optimization**: AdamW optimizer.
- **Scheduler**: Standard linear/cosine schedulers (implied, though simple optimization used in `trainer.py`).
- **Evaluation**:
    - Metrics: Micro-F1, ROC-AUC, and Precision@5 (P@5).
    - **N.B**: Metrics are calculated *independently* for each nesting dimension during validation.
    - **Threshold Tuning**: Since the dataset is highly imbalanced, probabilities can be pushed low. The evaluation includes a threshold tuning sweep to find the optimal global threshold specifically for Micro-F1, recognizing that 0.5 is often a suboptimal default.    
    - Evaluation tools in `src/utils.py` automatically generate grouped bar charts and validation curves to track these metrics across Matryoshka dimensions during training.
    - This allows verification that the 64-dim representation performs comparably to the 768-dim one (the "Matryoshka" effect).

## 5. Directory Structure
- `src/`: Core source code.
    - `models.py`: Neural network definitions.
    - `loss.py`: Custom MRL loss functions.
    - `trainer.py`: Training and evaluation loops.
    - `data.py`: PyTorch Datasets.
    - `dataset/preprocess.py`: Raw text cleaning and label mapping.
- `logs/`: Stores metric plots (F1, AUC curves).
- `checkpoints/`: Stores model weights (`best_model.pt`).
- `ablation_study/`: Dedicated module for Retrieval-Based Matryoshka ablation, integrating `sentence_transformers`.

## 6. Ablation Studies

To isolate the contributions of LAA and the MRL-classification setup, two ablation studies are provided:

### 6.1. Study A: Standard Attention (`model_type=standard_attn`)
This study compares LAA against a simpler baseline.
- **Architecture**: `Backbone -> Mean Pooling -> Matryoshka Linear Classifier`.
- **Implementation**: Instead of building attention maps for each label, the backbone outputs are mean-pooled down to a single document representation. A linear projection then maps this vector simultaneously across all dimensions to the label space.

### 6.2. Study B: Retrieval-Based Matryoshka (`model_type=retrieval`)
This study compares the multi-label classification paradigm against a Bi-Encoder Dense Retrieval framework.
- **Framework**: Built on `sentence_transformers`.
- **Training Paradigm**: Multi-label records are exploded into `(Clinical Text, ICD Description)` positive pairs. The model minimizes `MultipleNegativesRankingLoss` wrapped inside `MatryoshkaLoss`.
- **Projection Layer**: An optional linear probe (`--use_projection`) can be enabled (and is strictly enforced when the backbone is frozen) to align the generalized text embeddings to the clinical hierarchy.
- **Evaluation**: Performs dynamic vector search (cosine similarity constraint) to evaluate Micro-F1, ROC-AUC, and P@5 locally per Matryoshka dimension.

## 7. Conclusion
By combining Label-Aware Attention (standard for ICD tasks) with Matryoshka Representation Learning, it produces a model that is both accurate and computationally flexible. The resulting embeddings can be truncated to 64 or 128 dimensions for efficient storage or retrieval while maintaining high classification performance. The provided ablation modes further facilitate isolating the effects of the attention mechanism and the learning paradigm.
