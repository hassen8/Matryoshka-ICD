import torch.nn as nn
import torch.nn.functional as F

class StandardMRLLoss(nn.Module):
    def __init__(self, importance_weights=None):
        super().__init__()
        # Optional weights c_m for each dimension (e.g., weigh 64 more heavily)
        self.importance_weights = importance_weights

    def forward(self, logits_dict, targets):
        # targets: [batch, num_labels] (Binary Multi-hot)
        total_loss = 0

        for dim, logits in logits_dict.items():
            # Standard Multi-Label BCE Loss
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            # Apply importance weight c_m
            w = 1.0
            if self.importance_weights and dim in self.importance_weights:
                w = self.importance_weights[dim]

            total_loss += w * loss

        return total_loss
    
class HierarchicalMRLLoss(nn.Module):
    def __init__(self, config, label_level_tensor):
        """
        label_level_tensor: A Tensor of shape [num_labels] where each value is
                            the hierarchy level {0, 1, 2, 3} of that label.
        """
        super().__init__()
        self.config = config
        self.register_buffer('label_levels', label_level_tensor)

    def forward(self, logits_dict, targets):
        total_loss = 0

        for dim_idx, dim in enumerate(self.config.nesting_dims):
            logits = logits_dict[dim] # [batch, num_labels]

            # Identify the Hierarchy Level this dimension is responsible for.
            # Using the level_map: {0:0, 1:1, 2:2, 3:3}
            # If dim is 64 (idx 0), it handles Level 0.
            # If dim is 128 (idx 1), it handles Level 1.

            # We define a "Target Level" for this dimension.
            # Any label with level <= target_level is included.
            # Any label with level > target_level is MASKED (excluded).

            # Find the hierarchy level associated with this dimension index
            target_level = -1
            for lvl, d_idx in self.config.level_map.items():
                if self.config.nesting_dims[d_idx] == dim:
                    target_level = lvl
                    break

            if target_level == -1:
                # Should not happen if config is correct
                continue

            # Create Mask: 1 for allowed labels, 0 for disallowed (too fine)
            # Shape: [num_labels]
            mask = (self.label_levels <= target_level).float()

            # Expand mask to batch size: [batch, num_labels]
            batch_mask = mask.unsqueeze(0).expand_as(logits)

            # Compute element-wise BCE (no reduction yet)
            loss_element = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

            # Apply Mask
            masked_loss = loss_element * batch_mask

            # Average over valid elements only
            # Sum over all, divide by sum of mask
            loss_scalar = masked_loss.sum() / (batch_mask.sum() + 1e-9)

            total_loss += loss_scalar

        return total_loss