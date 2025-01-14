import torch
import torch.nn.functional as F


def tuplet_loss(anchor_emb, positive_emb, negative_embs, margin=0.2, k=2):
    # Calculate positive distances
    positive_dist = F.pairwise_distance(anchor_emb, positive_emb)
    
    # Calculate negative distances
    if negative_embs.numel() > 0:  # Ensure negatives are not empty
        neg_dist = torch.cdist(anchor_emb, negative_embs.view(-1, anchor_emb.size(1))).view(anchor_emb.size(0), -1)
        # Select the k hardest negatives
        top_k_negatives, _ = torch.topk(neg_dist, k, dim=1, largest=False)
        hardest_negative_dist = torch.mean(top_k_negatives, dim=1)
    else:
        # Default to a high distance when no negatives exist
        hardest_negative_dist = torch.tensor(float('inf')).to(anchor_emb.device)
    
    # Calculate the loss
    loss = torch.mean(F.relu(positive_dist - hardest_negative_dist + margin))
    
    return loss