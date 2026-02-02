import torch
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


def get_data_loaders(tokenizer, batch_size=32, max_len=512, text_col='query', label_col='leaf_doc', csv_path='', output_dir=''):
  
    train_df, validate_df, test_df, label2id, num_labels = preprocess_data()
    
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
    