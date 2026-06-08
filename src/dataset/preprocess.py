import pandas as pd
from pathlib import Path
from src.utils import save_file

# remove unnecessary data from report_text, keep only the findings and impression sections and remove the section titles and empty lines between them
def clean_report_text(text):
    findings_idx = text.lower().find('findings:')
    impression_idx = text.lower().find('impression:')
    if findings_idx == -1 and impression_idx == -1:
        return text
    elif findings_idx != -1 and impression_idx != -1:
        return text[findings_idx:impression_idx].replace('FINDINGS:', '').strip() + " " + text[impression_idx:].replace('IMPRESSION:', '').strip()
    elif findings_idx != -1:
        return text[findings_idx:]
    else:
        return text[impression_idx:]
    
def aggregate_data(df):
    """
    Groups the dataframe by reportid so that one text = list of all leaf_docs.
    """
    # Group by reportid and aggregate leaf_doc into a list
    # We also keep the report_text (taking the first one since duplicates are identical)
    df_grouped = df.groupby('reportid').agg({
        'report_text': 'first',
        'query':'first',
        'split': 'first',
        'leaf_doc': list,  # This creates a list of labels like [2813, 2748, 2805...]
        'leaf_icd': list
    }).reset_index()

    return df_grouped
    
def preprocess_data(mimic_folder='/datasets/MIMIC-CXR/files', file='/datasets/MIMIC-CXR/mimicxr_parsed_ds.jsonl', output_dir='/home/hsali/projects/icd/data'):
    mimic_folder = Path(mimic_folder)
    file = Path(file)
    output_dir = Path(output_dir)

    #read jsonl file
    dataset = pd.read_json(file, lines=True)

    # get report text and also add icd leaf heirarchy
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        report_id = row['reportid']
        icd_leaf = row['icd_hierarchy'].split('|')[-1]
        icd_leaf_doc = row['docid'].split('-')[-1]
        folder, subfolder, subject_id = report_id.split('.')
        filepath = Path( f'{mimic_folder}/{folder}/{subfolder}/{subject_id}.txt' )
        with open(filepath, 'r') as f:
            report_text = f.read()
        dataset.at[i, 'report_text'] = report_text
        dataset.at[i, 'leaf_icd'] = icd_leaf
        dataset.at[i, 'leaf_doc'] = icd_leaf_doc

    # clean the report text
    dataset['report_text'] = dataset['report_text'].apply(clean_report_text)
    dataset['query'] = dataset['query'].str.replace('|',', ')

    # save leaf icd's
    labels = list(set(dataset['leaf_icd'].sort_values().tolist()))
    docids = list(set(dataset['leaf_doc'].sort_values().tolist()))

    with open(f'{output_dir}/ALL_LABELS.txt', 'w') as f:
        f.write('\n'.join(labels))

    with open(f'{output_dir}/docids.txt', 'w') as f:
        f.write('\n'.join(labels))


    # # Get all the labels
    # dataset = pd.read_csv('/content/drive/MyDrive/drive_bys/ds.csv')
    # # labels = pd.read_csv('ALL_LABELS.txt', header=None)
    # # docids = pd.read_csv('docids.txt', header=None)
     # # save leaf icd's
    # # labels = list(set(dataset['icd_leaf'].sort_values().tolist()))
    # docids = list(set(dataset['leaf_doc'].sort_values().tolist()))
    # # labels = labels[0].sort_values().tolist()
    # # docids = docids[0].sort_values().tolist()

    # create the mappings
    label2id = {label: idx for idx, label in enumerate(docids)}
    id2label = {idx: label for idx, label in enumerate(docids)}
    num_labels = len(docids)



    # Assuming your raw dataframe is called 'df_raw'
    df_grouped = aggregate_data(dataset)

    # --- CRITICAL STEP: Create a contiguous Label Map ---
    # Your leaf_docs are IDs like 2813, 2748. If the max ID is 5000,
    # but you only have 100 classes, a vector of size 5000 is wasteful.
    # We map your dataset IDs to range 0..N_labels-1
    all_unique_labels = sorted(dataset['leaf_doc'].unique())
    label2id = {label: idx for idx, label in enumerate(all_unique_labels)}
    num_labels = len(all_unique_labels)

    print(f"Total unique labels: {num_labels}")

    # split the train validate and test datasets
    train_df = df_grouped[df_grouped['split']=='train'].reset_index(drop=True)
    validate_df = df_grouped[df_grouped['split']=='validate'].reset_index(drop=True)
    test_df = df_grouped[df_grouped['split']=='test'].reset_index(drop=True)
    
    #save cleaned dataframes
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    validate_df.to_csv(f'{output_dir}/validate.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    #save other files
    save_file(label2id, f'{output_dir}/label2id')     
    save_file(id2label, f'{output_dir}/id2label')
         
    return train_df, validate_df, test_df, label2id, num_labels
    # len(train_df),len(validate_df), len(test_df)

def preprocess_data_v2(
    icd_path='/home/hsali/projects/icd/MIMIC-GEN-RAG/final_mimiciv_icd_with_radiology_refs.jsonl',
    entity_path='/home/hsali/projects/icd/MIMIC-GEN-RAG/entity_summaries_full.jsonl',
    output_dir='/home/hsali/projects/icd/data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    min_code_frequency=0,
):
    import json
    import random
    import numpy as np

    output_dir = Path(output_dir)

    # 1. Load entity summaries: index by (subject_id, hadm_id)
    print("Loading entity summaries...")
    entity_map = {}
    with open(entity_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            key = (row['subject_id'], row['hadm_id'])
            entity_map[key] = row['summary_text']
    print(f"  Loaded {len(entity_map)} entity summaries")

    # 2. Load ICD codes, filter CM only, group by (subject_id, hadm_id)
    print("Loading ICD codes...")
    icd_map = {}
    total_icd_rows = 0
    cm_rows = 0
    with open(icd_path, 'r') as f:
        for line in f:
            total_icd_rows += 1
            row = json.loads(line)
            if row['code_system'] != 'CM':
                continue
            cm_rows += 1
            key = (row['subject_id'], row['hadm_id'])
            if key not in icd_map:
                icd_map[key] = []
            icd_map[key].append({
                'icd_code': row['icd_code'],
                'seq_num': row['seq_num'],
            })
    print(f"  Total ICD rows: {total_icd_rows}, CM rows: {cm_rows}")
    print(f"  Unique admissions with CM codes: {len(icd_map)}")

    # 3. Inner join: keep only admissions in both
    common_keys = set(entity_map.keys()) & set(icd_map.keys())
    print(f"  Admissions with both text and codes: {len(common_keys)}")

    # 4. Build per-admission records: sort codes by seq_num
    records = []
    for key in common_keys:
        codes = sorted(icd_map[key], key=lambda x: x['seq_num'])
        records.append({
            'subject_id': key[0],
            'hadm_id': key[1],
            'summary_text': entity_map[key],
            'icd_codes': [c['icd_code'] for c in codes],
            'seq_nums': [c['seq_num'] for c in codes],
        })

    df = pd.DataFrame(records)
    print(f"  Total merged records: {len(df)}")

    # 5. Filter low-frequency codes
    if min_code_frequency > 0:
        from collections import Counter
        code_freq = Counter()
        for codes_list in df['icd_codes']:
            code_freq.update(codes_list)
        valid_codes = {c for c, f in code_freq.items() if f >= min_code_frequency}
        removed_codes = set(code_freq.keys()) - valid_codes
        print(f"  Filtering codes with freq < {min_code_frequency}: removing {len(removed_codes)} rare codes")
        df['icd_codes'] = df['icd_codes'].apply(
            lambda clist: [c for c in clist if c in valid_codes]
        )
        # Remove admissions that end up with zero labels
        before = len(df)
        df = df[df['icd_codes'].apply(len) > 0].reset_index(drop=True)
        print(f"  Removed {before - len(df)} admissions with zero labels after filtering")

    # 6. Stratified multi-label split: ensure max code coverage across all splits
    from collections import defaultdict

    code_to_adms = defaultdict(set)
    for _, row in df.iterrows():
        for code in row['icd_codes']:
            code_to_adms[code].add(row['hadm_id'])

    all_adms = set(df['hadm_id'].unique())
    rng = np.random.RandomState(random_seed)

    code_freq = {c: len(adms) for c, adms in code_to_adms.items()}
    sorted_codes = sorted(code_freq, key=lambda c: code_freq[c])

    train_ids = set()
    val_ids = set()
    test_ids = set()
    assigned_set = set()

    def get_unassigned(code, max_needed=3):
        result = []
        for a in code_to_adms[code]:
            if a not in assigned_set:
                result.append(a)
                if len(result) >= max_needed:
                    break
        return result

    for code in sorted_codes:
        freq = code_freq[code]
        available = get_unassigned(code, 3)
        rng.shuffle(available)

        if freq >= 3 and len(available) >= 3:
            train_ids.add(available[0])
            val_ids.add(available[1])
            test_ids.add(available[2])
            assigned_set.update(available[:3])
        elif freq >= 2 and len(available) >= 2:
            train_ids.add(available[0])
            val_ids.add(available[1])
            assigned_set.update(available[:2])
        elif len(available) >= 1:
            train_ids.add(available[0])
            assigned_set.add(available[0])

    print(f"  Phase 1 coverage assignments — Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Phase 2: fill remaining admissions to match target ratios
    remaining = list(all_adms - assigned_set)
    rng.shuffle(remaining)

    n_total = len(all_adms)
    n_train_target = int(n_total * train_ratio)
    n_val_target = int(n_total * val_ratio)

    n_train_remaining = max(0, n_train_target - len(train_ids))
    n_val_remaining = max(0, n_val_target - len(val_ids))

    train_ids.update(remaining[:n_train_remaining])
    val_ids.update(remaining[n_train_remaining:n_train_remaining + n_val_remaining])
    test_ids.update(remaining[n_train_remaining + n_val_remaining:])

    df['split'] = df['hadm_id'].apply(
        lambda x: 'train' if x in train_ids else ('validate' if x in val_ids else 'test')
    )

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    validate_df = df[df['split'] == 'validate'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Val: {len(validate_df)}, Test: {len(test_df)}")

    # Report split coverage
    def get_codes(df_sub):
        codes = set()
        for clist in df_sub['icd_codes']:
            codes.update(clist)
        return codes
    train_cov = get_codes(train_df)
    val_cov = get_codes(validate_df)
    test_cov = get_codes(test_df)
    print(f"  Code coverage — Train: {len(train_cov)}, Val: {len(val_cov)}, Test: {len(test_cov)}")
    print(f"  Test-only codes (not in train): {len(test_cov - train_cov)}")
    print(f"  Val-only codes (not in train):  {len(val_cov - train_cov)}")

    # 7. Build label2id from all unique icd_codes (already filtered)
    all_codes = set()
    for codes_list in df['icd_codes']:
        all_codes.update(codes_list)
    all_codes = sorted(all_codes)
    label2id = {code: idx for idx, code in enumerate(all_codes)}
    id2label = {idx: code for idx, code in enumerate(all_codes)}
    num_labels = len(all_codes)
    print(f"  Unique ICD codes (CM): {num_labels}")

    # 8. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    validate_df.to_csv(f'{output_dir}/validate.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    save_file(label2id, f'{output_dir}/label2id')
    save_file(id2label, f'{output_dir}/id2label')
    print(f"  Saved to {output_dir}/")

    return train_df, validate_df, test_df, label2id, num_labels


if __name__ == "__main__":
    train_df, validate_df, test_df, label2id, num_labels = preprocess_data_v2()
    print(f"Train: {len(train_df)}, Val: {len(validate_df)}, Test: {len(test_df)}, Labels: {num_labels}")

