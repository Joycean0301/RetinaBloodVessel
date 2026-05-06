import csv
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from config import PLOT_DIR, EPOCHS, BACKBONE, LOG_DIR

# ──────────────────────── VISUALIZATION ─────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor [3,H,W] → uint8 numpy [H,W,3]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # [H,W,3]
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def read_training_log(log_path: str) -> list[dict]:
    """Read epoch metrics saved by training.py."""
    history = []
    with open(log_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                'epoch': int(row['epoch']),
                'lr': float(row['lr']),
                'elapsed_sec': float(row['elapsed_sec']),
                'tr_loss': float(row['tr_loss']),
                'tr_dice': float(row['tr_dice']),
                'tr_iou': float(row['tr_iou']),
                'va_loss': float(row['va_loss']),
                'va_dice': float(row['va_dice']),
                'va_iou': float(row['va_iou']),
                'best_val_dice': float(row['best_val_dice']),
                'is_best': int(row['is_best']),
            })
    return history


def _default_log_path(model_name: str, log_dir: str = LOG_DIR) -> str:
    return os.path.join(log_dir, f'{model_name}_training_log.csv')


def _plot_metric(ax, history: list[dict], train_key: str, val_key: str, title: str, ylabel: str):
    epochs = [h['epoch'] for h in history]
    ax.plot(epochs, [h[train_key] for h in history], marker='o', label='Train')
    ax.plot(epochs, [h[val_key] for h in history], marker='o', label='Val')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()


def save_prediction_mask_frame(model_name: str, epoch: int, sample_pred: torch.Tensor):
    """Save only the predicted mask for building training-progress GIFs."""
    pred_prob = torch.sigmoid(sample_pred).squeeze().cpu().numpy()
    pred_bin = (pred_prob > 0.5).astype(np.float32)

    gif_dir = os.path.join(PLOT_DIR, 'gif')
    os.makedirs(gif_dir, exist_ok=True)

    out_path = os.path.join(gif_dir, f'{model_name}_{epoch}.png')
    plt.imsave(out_path, pred_bin, cmap='gray', vmin=0, vmax=1)
    print(f"    GIF frame saved → {out_path}")


def save_epoch_plot(model_name: str, epoch: int, log_path: str,
                    sample_img: torch.Tensor, sample_mask: torch.Tensor,
                    sample_pred: torch.Tensor, backbone_name: str = BACKBONE):
    """
    Lưu ảnh progress mỗi epoch.
    Metrics được đọc từ CSV log do training.py ghi ra.
    """
    history = read_training_log(log_path)
    if not history:
        raise ValueError(f'Empty training log: {log_path}')

    last = history[-1]
    best = max(history, key=lambda h: h['va_dice'])

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{model_name} - Epoch {epoch}/{EPOCHS} - Backbone: {backbone_name}',
                 fontsize=14, fontweight='bold')

    _plot_metric(axes[0, 0], history, 'tr_loss', 'va_loss', 'Loss', 'Loss')
    _plot_metric(axes[0, 1], history, 'tr_dice', 'va_dice', 'Dice', 'Dice')
    _plot_metric(axes[0, 2], history, 'tr_iou', 'va_iou', 'IoU', 'IoU')

    axes[0, 3].axis('off')
    axes[0, 3].text(
        0.05, 0.95,
        '\n'.join([
            f"Epoch: {last['epoch']}",
            f"LR: {last['lr']:.6g}",
            f"Time: {last['elapsed_sec']:.1f}s",
            '',
            f"Train Loss: {last['tr_loss']:.4f}",
            f"Val Loss: {last['va_loss']:.4f}",
            f"Train Dice: {last['tr_dice']:.4f}",
            f"Val Dice: {last['va_dice']:.4f}",
            f"Train IoU: {last['tr_iou']:.4f}",
            f"Val IoU: {last['va_iou']:.4f}",
            '',
            f"Best Val Dice: {best['va_dice']:.4f}",
            f"Best Epoch: {best['epoch']}",
        ]),
        transform=axes[0, 3].transAxes,
        va='top',
        family='monospace',
        fontsize=10,
    )

    img_np = denormalize(sample_img)
    mask_np = sample_mask.squeeze().cpu().numpy()
    pred_prob = torch.sigmoid(sample_pred).squeeze().cpu().numpy()
    pred_bin = (pred_prob > 0.5).astype(np.float32)

    overlay = img_np.copy().astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_np * 0.55, 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] + pred_bin * 0.55, 0, 1)

    image_panels = [
        (axes[1, 0], img_np, 'Input Image', None),
        (axes[1, 1], mask_np, 'Ground Truth Mask', 'gray'),
        (axes[1, 2], pred_bin, 'Predicted Mask', 'gray'),
        (axes[1, 3], overlay, 'Overlay', None),
    ]

    for ax, data, title, cmap in image_panels:
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    model_plot_dir = os.path.join(PLOT_DIR, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    out_path = os.path.join(model_plot_dir, f'epoch_{epoch:04d}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    📊 Plot saved → {out_path}")


def save_final_comparison(model_names: list[str] | None = None, log_dir: str = LOG_DIR):
    """So sánh Loss / Dice / IoU của các model bằng dữ liệu CSV log."""
    if not os.path.isdir(log_dir):
        raise ValueError(f'Training log directory not found: {log_dir}')

    if model_names is None:
        model_names = sorted(
            filename.replace('_training_log.csv', '')
            for filename in os.listdir(log_dir)
            if filename.endswith('_training_log.csv')
        )

    histories = {}
    for model_name in model_names:
        log_path = _default_log_path(model_name, log_dir)
        if os.path.exists(log_path):
            histories[model_name] = read_training_log(log_path)

    if not histories:
        raise ValueError(f'No training logs found in: {log_dir}')

    metrics = [
        ('va_loss', 'Val Loss'),
        ('va_dice', 'Val Dice'),
        ('va_iou', 'Val IoU'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Model Comparison - All Epochs', fontsize=14, fontweight='bold')

    for ax, (key, label) in zip(axes, metrics):
        for model_name, history in histories.items():
            epochs = [h['epoch'] for h in history]
            values = [h[key] for h in history]
            ax.plot(epochs, values, marker='o', label=model_name)

        ax.set_title(label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    plt.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, 'comparison_all_models.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  📊 Final comparison plot → {out_path}")
