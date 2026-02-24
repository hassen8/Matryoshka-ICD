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

if __name__ == "__main__":
    train_df, validate_df, test_df, label2id, num_labels = preprocess_data()
    print(len(train_df), len(validate_df), len(test_df))

