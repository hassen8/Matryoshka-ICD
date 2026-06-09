import torch
import json
from src.dataset.preprocess import preprocess_data, preprocess_data_v2
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
            desc = entry.get('description', '')
            if hierarchy and desc:
                h_levels = hierarchy.split('|')
                d_levels = desc.split('|')
                for hl, dl in zip(h_levels, d_levels):
                    if hl not in desc_map:
                        desc_map[hl] = dl.strip()
        return desc_map
    except Exception as e:
        print(f"Error loading description map: {e}")
        return {}


def build_icd_descriptions_for_dataset(dataset_dir, label2id):
    import json as _json
    desc_map = get_icd_description_map()

    code_to_seq = {}
    with open(f'{dataset_dir}/final_mimiciv_icd_with_radiology_refs.jsonl', 'r') as f:
        for line in f:
            d = _json.loads(line)
            if d['code_system'] == 'CM' and d['icd_code'] not in code_to_seq:
                code_to_seq[d['icd_code']] = d['icd_sequence'].split('|')

    icd_descriptions_map = {}
    matched = 0
    for code in label2id.keys():
        seq = code_to_seq.get(code, [])
        desc = None
        for level_code in seq:
            if level_code in desc_map:
                desc = desc_map[level_code]
                break
        if desc:
            matched += 1
        else:
            desc = f"ICD-{code}"
        icd_descriptions_map[code] = desc

    print(f"  ICD descriptions: {matched}/{len(label2id)} codes matched ({matched/len(label2id)*100:.1f}%)")
    return icd_descriptions_map


class RetrievalDataset(Dataset):
    def __init__(self, dataframe, icd_descriptions_map, tokenizer=None, max_len=512, text_col="summary_text", icd_col='icd_codes'):
        self.max_len = max_len

        queries = []
        descriptions = []
        print(f"Building retrieval pairs from {len(dataframe)} admissions...")
        for _, row in dataframe.iterrows():
            text = str(row[text_col])
            for icd in row[icd_col]:
                if icd in icd_descriptions_map:
                    queries.append(text)
                    descriptions.append(icd_descriptions_map[icd])

        print(f"  {len(queries)} positive pairs — tokenizing (chunked)...")
        import torch
        chunk_size = 10000
        q_chunks, d_chunks = [], []
        for i in range(0, len(queries), chunk_size):
            qc = queries[i:i+chunk_size]
            dc = descriptions[i:i+chunk_size]
            q_chunks.append(tokenizer(qc, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'))
            d_chunks.append(tokenizer(dc, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'))
            if (i // chunk_size) % 10 == 0:
                print(f"    {i}/{len(queries)} pairs...")
        self.query_features = {k: torch.cat([c[k] for c in q_chunks], dim=0) for k in q_chunks[0].keys()}
        self.desc_features = {k: torch.cat([c[k] for c in d_chunks], dim=0) for k in d_chunks[0].keys()}
        print(f"  Pre-tokenization complete.")

    def __len__(self):
        return len(self.query_features['input_ids'])

    def __getitem__(self, index):
        q_feat = {k: v[index] for k, v in self.query_features.items()}
        d_feat = {k: v[index] for k, v in self.desc_features.items()}
        return [q_feat, d_feat]


def get_data_loaders(tokenizer=None, batch_size=32, max_len=512, text_col='summary_text', label_col='icd_codes', csv_path='', output_dir='/home/hsali/projects/icd/data', dataset_dir='/home/hsali/projects/icd/MIMIC-GEN-RAG', model_type='laa'):
  
    icd_path = f'{dataset_dir}/final_mimiciv_icd_with_radiology_refs.jsonl'
    entity_path = f'{dataset_dir}/entity_summaries_full.jsonl'
    train_df, validate_df, test_df, label2id, num_labels = preprocess_data_v2(
        icd_path=icd_path,
        entity_path=entity_path,
        output_dir=output_dir,
    )
    
    if model_type == 'retrieval':
        icd_descriptions_map = build_icd_descriptions_for_dataset(dataset_dir, label2id)
        
        train_ds = RetrievalDataset(dataframe=train_df, icd_descriptions_map=icd_descriptions_map, tokenizer=tokenizer, max_len=max_len, text_col=text_col, icd_col=label_col)
        validate_ds = validate_df
        test_ds = test_df
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        ordered_descriptions = [icd_descriptions_map.get(list(label2id.keys())[list(label2id.values()).index(i)], f"ICD {i}") for i in range(num_labels)]
        
        return train_loader, validate_ds, test_ds, label2id, num_labels, {
            'retrieval_descriptions': ordered_descriptions,
            'train_df': train_df,
            'validate_df': validate_df,
            'test_df': test_df,
        }

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
        # validate_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
        validate_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, validate_loader, test_loader, label2id, num_labels, {
            'retrieval_descriptions': None,
            'train_df': train_df,
            'validate_df': validate_df,
            'test_df': test_df,
        }
    