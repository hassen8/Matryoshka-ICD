import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models

def get_retrieval_model_linear(pretrained_model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', config=None):
    """
    Constructs a SentenceTransformer model for retrieval-based Matryoshka learning.
    It builds the model from components to allow inserting a projection layer easily.
    """
    # 1. Base Transformer
    word_embedding_model = models.Transformer(pretrained_model_name, max_seq_length=config.max_len if config else 512)
    
    if config and config.freeze_backbone:
        for param in word_embedding_model.auto_model.parameters():
            param.requires_grad = False
            
    # 2. Pooling (Mean Pooling)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    modules = [word_embedding_model, pooling_model]
    
    # 3. Optional Projection Layer
    if config and config.use_projection:
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=pooling_model.get_sentence_embedding_dimension(),
                                   activation_function=nn.Identity()) # Linear probe
        modules.append(dense_model)
        
    model = SentenceTransformer(modules=modules)
    return model


def get_retrieval_model(pretrained_model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', config=None):
    # 1. Base Transformer
    word_embedding_model = models.Transformer(pretrained_model_name, max_seq_length=config.max_len if config else 512)
    
    # Freeze the backbone to match HMPLMICD
    if config and config.freeze_backbone:
        for param in word_embedding_model.auto_model.parameters():
            param.requires_grad = False
            
    # 2. Pooling (Mean Pooling)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    
    modules = [word_embedding_model, pooling_model]
    
    # 3. Non-Linear Projection Layer (To fairly match LAA's capacity)
    if config and config.use_projection:
        # First layer with Tanh (matches LAA's W1 + Tanh)
        dense_model_1 = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                     out_features=pooling_model.get_sentence_embedding_dimension(),
                                     activation_function=nn.Tanh())
        
        # Second linear layer (matches the final projection flexibility)
        dense_model_2 = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                     out_features=pooling_model.get_sentence_embedding_dimension(),
                                     activation_function=nn.Identity())
        
        modules.extend([dense_model_1, dense_model_2])
        
    model = SentenceTransformer(modules=modules)
    return model

