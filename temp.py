def get_hierarchy_level(code):
    """
    Returns the hierarchy level (0-3) for an ICD-9 code.
    
    Level Mapping:
    0: Chapter/Section Range (e.g., '390-459')
    1: Category (Root)       (e.g., '410', 'V10', 'E800')
    2: Subcategory           (e.g., '410.0', 'E800.1')
    3: Subclassification     (e.g., '410.01')
    """
    # Normalize input
    code = str(code).strip().upper()
    
    # LEVEL 0: Detect Ranges (Chapters or Sections)
    if '-' in code: 
        return 0
        
    # Remove the dot to count pure digits/characters
    clean_code = code.replace('.', '')
    length = len(clean_code)
    
    # Determine the "Root" length
    # Standard ICD-9 and V-codes have a 3-character root (e.g., 410, V01)
    # E-codes (External causes) have a 4-character root (e.g., E812)
    is_e_code = clean_code.startswith('E')
    root_len = 4 if is_e_code else 3
    
    # LEVEL 1: Category (Root Code)
    if length == root_len:
        return 1
        
    # LEVEL 2: Subcategory (Root + 1 digit)
    elif length == root_len + 1:
        return 2
        
    # LEVEL 3: Subclassification (Root + 2 digits)
    elif length >= root_len + 2:
        return 3
        
    return 0

# Dataframe transformation logic
def expand_hierarchy(df):
    """
    Takes wide format dataframe with list of leaf codes.
    Returns long format with all ancestors added.
    """
    # Pseudo-code for brevity
    # 1. Load ICD-9 taxonomy graph (networkx or dictionary)
    # 2. For each patient row:
    #    current_codes = row['codes']
    #    new_codes = set(current_codes)
    #    for c in current_codes:
    #        new_codes.update(get_ancestors(c))
    #    row['expanded_codes'] = list(new_codes)
    pass


# Assume `label_map` is a dict {code_string: index}
num_labels = len(label_map)
label_level_tensor = torch.zeros(num_labels, dtype=torch.long)

for code, idx in label_map.items():
    level = get_hierarchy_level(code)
    label_level_tensor[idx] = level

# This tensor is passed to the HierarchicalMRLLoss during training.