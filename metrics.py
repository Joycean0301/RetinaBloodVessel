import torch

# ─────────────────────────── METRICS ────────────────────────────
def dice_score(pred_logits: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-7) -> float:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + eps)).item()


def iou_score(pred_logits: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-7) -> float:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + eps)).item()