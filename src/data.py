import torch
import json
from src.dataset.preprocess import preprocess_data
from torch.utils.data import Dataset, DataLoader


class PLM_MultiLabel_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, max_len=512, text_col='report_text', label_col='leaf_doc'):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row[self.text_col])

        # Get list of labels (e.g., [2813, 2748])
        raw_labels = row[self.label_col]

        # --- Create Multi-Hot Vector ---
        # Initialize zero vector [0, 0, ..., 0]
        label_vector = torch.zeros(self.num_labels, dtype=torch.float)

        # Set indices to 1.0
        for label in raw_labels:
            if label in self.label2id:
                mapped_id = self.label2id[label]
                label_vector[mapped_id] = 1.0

        # --- Tokenization ---
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector # Shape: [num_labels] (Floats)
        }


def get_icd_description_map(json_path='/home/hsali/projects/icd/data/icd_descr_map.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        desc_map = {}
        for entry in data:
            hierarchy = entry.get('icd_hierarchy', '')
            if hierarchy:
                # Use the last part of hierarchy as the key
                code = hierarchy.split('|')[-1]
                # Store full description for better semantics
                desc_map[code] = entry.get('description', code)
        return desc_map
    except Exception as e:
        print(f"Error loading description map: {e}")
        return {}


class RetrievalDataset(Dataset):
    def __init__(self, dataframe, icd_descriptions_map, text_col="query", icd_col='leaf_icd'):
        # icd_descriptions_map maps leaf_icd to its description string
        self.data = dataframe
        # We will unroll the multi-label into positive pairs (query, description)
        self.pairs = []
        for index, row in dataframe.iterrows():
            text = str(row[text_col])
            icds = row[icd_col]
            for icd in icds:
                if icd in icd_descriptions_map:
                    self.pairs.append( (text, icd_descriptions_map[icd]) )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from sentence_transformers import InputExample
        query, description = self.pairs[index]
        return InputExample(texts=[query, description])


def get_data_loaders(tokenizer=None, batch_size=32, max_len=512, text_col='query', label_col='leaf_doc', csv_path='', output_dir='', model_type='laa'):
  
    train_df, validate_df, test_df, label2id, num_labels = preprocess_data()
    
    if model_type == 'retrieval':
        # Build description mapping using leaf_icd
        desc_map = get_icd_description_map('/home/hsali/projects/icd/data/icd_descr_map.json')
        
        # Build doc2icd mapping to map leaf_doc back to leaf_icd for evaluation ordered_descriptions
        doc2icd = {}
        for df in [train_df, validate_df, test_df]:
            for _, row in df.iterrows():
                for doc, icd in zip(row.get('leaf_doc', []), row.get('leaf_icd', [])):
                    doc2icd[doc] = icd

        # We need mapping from the contiguous integer ID to description so evaluation indices align
        # or from the raw label to description for training.
        icd_descriptions_map = {lbl: desc_map.get(doc2icd.get(lbl, lbl), f"ICD {lbl}") for lbl in label2id.keys()}
        
        train_ds = RetrievalDataset(dataframe=train_df, icd_descriptions_map=desc_map, text_col=text_col, icd_col='leaf_icd')
        # Validation/Test do not need unrolling for SentenceTransformers evaluation, they need (queries, relevant_docs) dicts
        # We will handle validation in the evaluation logic instead of a DataLoader of InputExamples.
        validate_ds = validate_df
        test_ds = test_df
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x) # collate_fn=list to avoid tensor casting
        
        # We return the list of ALL unique descriptions in order of label2id (index 0 to num_labels-1)
        ordered_descriptions = [icd_descriptions_map[list(label2id.keys())[list(label2id.values()).index(i)]] for i in range(num_labels)]
        
        return train_loader, validate_ds, test_ds, ordered_descriptions, num_labels

    else:
        # Standard PLM Classification
        train_ds = PLM_MultiLabel_Dataset(
            dataframe=train_df,
            tokenizer=tokenizer,
            label2id=label2id,
            max_len=max_len,
            text_col=text_col,
            label_col=label_col,
            )
        
        validate_ds = PLM_MultiLabel_Dataset(
            dataframe=validate_df,
            tokenizer=tokenizer,
            label2id=label2id,
            max_len=max_len,
            text_col=text_col,
            label_col=label_col,
            )
        
        test_ds = PLM_MultiLabel_Dataset(
            dataframe=test_df,
            tokenizer=tokenizer,
            label2id=label2id,
            max_len=max_len,
            text_col=text_col,
            label_col=label_col,
            )
        
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        validate_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
        
        return train_loader, validate_loader, test_loader, label2id, num_labels
    