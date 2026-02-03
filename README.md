# Matryoshka-ICD: Automated ICD Coding with MRL

**Matryoshka-ICD** implements an automated ICD coding system for clinical reports (e.g., MIMIC-CXR). By combining **Matryoshka Representation Learning (MRL)** with **Label-Aware Attention (LAA)**, the model learns nested embeddings that perform efficiently at multiple dimensions (e.g., 64, 128, 768). This allows for adaptive deployment—scaling inference speed and storage usage without needing to retrain the model.

## key Features

- **Matryoshka Representation Learning (MRL)**: Learns a "Russian Doll" structure in embeddings, ensuring the first $k$ dimensions form a high-quality representation.
- **Label-Aware Attention**: Computes specific attention weights for each label to capture relevant clinical evidence from long documents.
- **Flexible Inference**: Evaluate using 64, 128, 256, or 768 dimensions depending on your computational budget.
- **BioClinical-ModernBERT Backbone**: Leverages state-of-the-art medical language models.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hassen8/Matryoshka-ICD.git
    cd Matryoshka-ICD
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset

The dataset structure is based on the MIMIC-CXR dataset. A preprocessed version of the dataset `mimicxr_parsed_ds.jsonl` was handproduced in the wide format which contained a multiple rows per patient for every applicable and/or adjacent ICD code. This was then collapsed to represent a multi-label classification problem. The preprocessing script at `src/dataset/preprocess.py` was used for this dataset.

In the future, i plan to introduce a script which automatically processes the MIMIC-CXR dataset and produces the wide format dataset. 

`mimicxr_parsed_ds.jsonl` data feilds are as follows:
 `subject_id`: Unique identifier for the patient (e.g., `10000032`).
- `study_id`: Unique identifier for the specific radiology study (e.g., `50414267`).
- `reportid`: ID (p.*) of the target report.
- `docid`: hyphen separated numeric representation of the ICD hierarchy in `icd_hierarchy`, where each ICD code and its higher levels are replaced by numeric identifiers corresponding to nodes in the hierarchical tree. (e.g., `9-16-128`).
- `icd_hierarchy`: pipe separated string representing the hierarchical ICD codes, showing the higher-level classifications it belongs to. (e.g., `R05|R05.9`).
- `query`: A text query, i.e., the sequence of keywords a doctor would search for in a potential search engine. (e.g., `pneumonia | pleural effusion.`).


### Training

To train the model, use `main.py`. You can configure hyperparameters via command-line arguments.

```bash
python main.py \
    --model_name "thomas-sounack/BioClinical-ModernBERT-base" \
    --batch_size 32 \
    --epochs 20 \
    --text_column "query" \
    --label_column "leaf_doc"
```

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | `None` | Path to the CSV dataset (optional if using internal processing). |
| `--model_name` | `thomas-sounack/BioClinical-ModernBERT-base` | HuggingFace model backbone. |
| `--text_column` | `query` | Column name in the dataframe containing the input text. |
| `--label_column` | `leaf_doc` | Column name containing the list of target labels. |
| `--nesting_dims` | `[64, 128, 256, 768]` | Dimensions for Matryoshka learning. |
| `--wandb_project` | `Matriyoshka` | Name of the W&B project for logging. |

## Project Structure

```
├── logs/                 # Stores training logs
├── checkpoints/          # Saved model weights
├── data/                 # Processed datasets (train/val/test CSVs)
├── main.py               # Entry point for training and evaluation
├── project_report.md     # Detailed technical report
├── requirements.txt      # Python dependencies
└── src/
    ├── configs.py        # Configuration and argument parsing
    ├── data.py           # PyTorch Dataset and DataLoader
    ├── loss.py           # MRL Loss implementation
    ├── models.py         # Model architecture (Backbone + LAA + MRL Head)
    ├── trainer.py        # Training loop and evaluation metrics
    ├── utils.py          # Utility functions
    └── dataset/
        └── preprocess.py # Data cleaning and preprocessing scripts
```

## Performance

The system evaluates **Micro-F1**, **ROC-AUC**, and **Precision@5** independently for each nesting dimension (e.g., 64d vs 768d). See `project_report.md` for theoretical background and details.