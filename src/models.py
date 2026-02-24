import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class LabelAwareAttention(nn.Module):
    """
    Computes label-specific representations from the document encoding.
    Output shape:
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # Projection matrix U from the paper (maps hidden states to attention latent space)
        # Equation: Z = tanh(V * H^T)
        self.W1 = nn.Linear(config.hidden_size, config.hidden_size)

        # Projection matrix to Label Space
        # Equation: A = softmax(W * Z)
        self.W2 = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [batch_size, seq_len, hidden_size]

        # 1. Project to latent attention space
        # Z shape: [batch, seq_len, hidden_size]
        Z = torch.tanh(self.W1(hidden_states))

        # 2. Compute attention logits for each label
        # attn_logits shape: [batch, seq_len, num_labels]
        attn_logits = self.W2(Z)

        # 3. Apply Masking
        # We must mask out padding tokens so they don't contribute to the average
        if attention_mask is not None:
            # attention_mask is 1 for tokens, 0 for pad.
            # We treat 0 as -inf for softmax.
            # Shape expansion: [batch, seq_len, 1] for broadcasting
            extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
            attn_logits = attn_logits + extended_mask

        # 4. Softmax over the sequence dimension (dim 1)
        # We transpose to [batch, num_labels, seq_len] first for clarity
        attn_logits = attn_logits.transpose(1, 2)
        alpha = F.softmax(attn_logits, dim=2) # [batch, num_labels, seq_len]

        if self.training:
            alpha = self.dropout(alpha)

        # 5. Compute Weighted Sum
        # D = Alpha * H
        # [batch, num_labels, seq_len] x [batch, seq_len, hidden_size]
        # -> [batch, num_labels, hidden_size]
        doc_representations = torch.bmm(alpha, hidden_states)

        return doc_representations

class MatryoshkaClassifier(nn.Module):
    """
    Implements Efficient Matryoshka Representation Learning (MRL-E).
    Uses a single linear layer but slices weights for nested dimensions.
    """
    def __init__(self, config):
        super().__init__()
        self.nesting_dims = config.nesting_dims
        self.hidden_size = config.hidden_size

        # Single projection layer.
        # Note: In PLM-ICD, we predict presence/absence (binary).
        # Since 'doc_representations' is already,
        # we need to project 'Hidden' to scalar 1.
        # This acts as w_i^T * d_i + b_i shared across all labels (common in LAA).
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, doc_representations):
        # doc_representations: [batch, num_labels, hidden_size]

        logits_dict = {}

        for dim in self.nesting_dims:
            # 1. Slice the Input Representation
            # We take the first 'dim' features from the last dimension
            sliced_rep = doc_representations[:, :, :dim] # [batch, num_labels, dim]

            # 2. Slice the Classifier Weights (Weight Tying)
            # Efficient MRL: We reuse the first 'dim' weights of the full classifier
            # Full weight: [1, hidden_size] -> Sliced: [1, dim]
            sliced_weight = self.classifier.weight[:, :dim]
            sliced_bias = self.classifier.bias

            # 3. Linear Projection
            # Functional linear call: input @ weight.T + bias
            # [batch, num_labels, dim] @ [dim, 1] +  -> [batch, num_labels, 1]
            logits = F.linear(sliced_rep, sliced_weight, sliced_bias)

            # Remove the last singleton dimension
            logits_dict[dim] = logits.squeeze(-1) # [batch, num_labels]

        return logits_dict

class HMPLMICD(nn.Module):
    def __init__(self, config, pretrained_model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'):
        super().__init__()
        self.config = config

        # Load Pretrained Backbone
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        if config.freeze_backbone:
          print(f"Freezing backbone: {pretrained_model_name}")
          for param in self.bert.parameters():
                param.requires_grad = False

        # Attach Custom Modules
        self.attention = LabelAwareAttention(config)
        self.classifier = MatryoshkaClassifier(config)

    def forward(self, input_ids, attention_mask):
        # 1. PLM Encoding
        # input_ids: [batch, seq_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [batch, seq_len, hidden]

        # Note: For real MIMIC-CXR usage, you would implement segment pooling here.
        # This involves reshaping input to [batch * segments, seq_len], encoding,
        # and then reshaping back/concatenating.
        # We omit the boilerplate reshape code for brevity but acknowledge its necessity.

        # 2. Label-Aware Attention
        # Produces [batch, num_labels, hidden]
        doc_reps = self.attention(last_hidden_state, attention_mask)

        # 3. Matryoshka Classification
        # Returns Dict[dim_int -> logits_tensor]
        logits_dict = self.classifier(doc_reps)

        return logits_dict

class StandardAttentionICD(nn.Module):
    def __init__(self, config, pretrained_model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'):
        super().__init__()
        self.config = config

        # Load Pretrained Backbone
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        if config.freeze_backbone:
            print(f"Freezing backbone: {pretrained_model_name}")
            for param in self.bert.parameters():
                param.requires_grad = False

        # Drop Label-Aware Attention; use standard projection
        # This acts as the replacement for the LAA classifier logic, simply replicating 
        # the projection over all labels from a mean-pooled sentence representation.
        self.num_labels = config.num_labels
        self.nesting_dims = config.nesting_dims
        self.hidden_size = config.hidden_size
        
        # In PLM-ICD with LAA, each label gets a unique attention vector, and a shared linear(hidden->1)
        # projects it. For standard pooling, we have *one* document vector, and project it to *num_labels*.
        # To support Matryoshka naturally, we do a Linear(hidden -> num_labels).
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [batch, seq_len, hidden]

        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        # embeddings: [batch, hidden]

        logits_dict = {}

        for dim in self.nesting_dims:
            # Slice representation
            sliced_rep = embeddings[:, :dim] # [batch, dim]

            # Slice weights (Weight Tying)
            # Full weight: [num_labels, hidden_size] -> Sliced: [num_labels, dim]
            sliced_weight = self.classifier.weight[:, :dim]
            sliced_bias = self.classifier.bias

            # Project
            # [batch, dim] @ [dim, num_labels] + bias -> [batch, num_labels]
            logits = F.linear(sliced_rep, sliced_weight, sliced_bias)

            logits_dict[dim] = logits # [batch, num_labels]

        return logits_dict