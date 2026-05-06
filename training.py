import os
import random
import time
import csv
import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss

from config import BACKBONE, BATCH_SIZE_BY_MODEL, DATA_DIR, EPOCHS, LR, NUM_WORKERS, SAVE_DIR, DEVICE, IMG_SIZE, SEED, LOG_DIR
from dataset import RetinaDataset
from metrics import dice_score, iou_score 
from model import build_model
from visualization import save_epoch_plot, save_prediction_mask_frame


# ───────────────────────── REPRODUCIBILITY ──────────────────────
def fix_seed(seed: int = SEED):
    """Fix random seeds so training is reproducible between runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _training_config(model_name: str, batch_size: int, backbone_label: str) -> dict:
    return {
        'model_name': model_name,
        'backbone': backbone_label,
        'epochs': EPOCHS,
        'learning_rate': LR,
        'batch_size': batch_size,
        'seed': SEED,
        'data_dir': DATA_DIR,
        'img_size': list(IMG_SIZE),
        'num_workers': NUM_WORKERS,
        'device': str(DEVICE),
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        'loss': 'DiceLoss(binary, from_logits=True)',
    }


def _init_training_logs(model_name: str, metadata: dict) -> tuple[str, str]:
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_path = os.path.join(LOG_DIR, f'{model_name}_training_log.csv')
    json_path = os.path.join(LOG_DIR, f'{model_name}_training_config.json')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'lr', 'elapsed_sec',
            'tr_loss', 'tr_dice', 'tr_iou',
            'va_loss', 'va_dice', 'va_iou',
            'best_val_dice', 'is_best',
        ])
        writer.writeheader()

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def _append_epoch_log(csv_path: str, row: dict):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'lr', 'elapsed_sec',
            'tr_loss', 'tr_dice', 'tr_iou',
            'va_loss', 'va_dice', 'va_iou',
            'best_val_dice', 'is_best',
        ])
        writer.writerow(row)


def _write_training_summary(json_path: str, metadata: dict, history: list, best_epoch: int, best_dice: float):
    summary = dict(metadata)
    summary.update({
        'best_epoch': best_epoch,
        'best_val_dice': best_dice,
        'history': history,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ──────────────────────── TRAIN / EVAL LOOPS ────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        total_iou  += iou_score(logits,  masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        total_loss += criterion(logits, masks).item()
        total_dice += dice_score(logits, masks)
        total_iou  += iou_score(logits,  masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────── MAIN ───────────────────────────────
def train_model(model_name: str):
    fix_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    batch_size = BATCH_SIZE_BY_MODEL.get(model_name, 1)  # default batch size if model not in config
    backbone_label = BACKBONE

    print(f"\n{'='*60}")
    print(f"  Training  : {model_name}  |  Backbone: {backbone_label}")
    print(f"  Device    : {DEVICE}")
    print(f"  Epochs    : {EPOCHS}  |  LR: {LR}  |  Batch: {batch_size}")
    print(f"{'='*60}")

    # ── Datasets & Loaders ──
    train_ds = RetinaDataset(DATA_DIR, split='train', img_size=IMG_SIZE, augment=True)
    test_ds  = RetinaDataset(DATA_DIR, split='test',  img_size=IMG_SIZE, augment=False)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)

    print(f"  Train samples : {len(train_ds)}  |  Test samples : {len(test_ds)}")

    # ── Fixed sample từ val set để visualize mỗi epoch ──
    # Luôn dùng ảnh index=0 để thấy rõ sự thay đổi theo epoch
    fixed_img, fixed_mask = test_ds[0]          # [3,H,W], [1,H,W]

    # ── Model, Loss, Optimizer ──
    model     = build_model(model_name, BACKBONE).to(DEVICE)
    criterion = DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    metadata = _training_config(model_name, batch_size, backbone_label)
    metadata.update({
        'train_samples': len(train_ds),
        'test_samples': len(test_ds),
        'train_batches': len(train_loader),
        'test_batches': len(test_loader),
        'model_params_total': sum(p.numel() for p in model.parameters()),
        'model_params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    log_csv_path, log_json_path = _init_training_logs(model_name, metadata)
    print(f"  Training log  : {log_csv_path}")
    print(f"  Config log    : {log_json_path}")

    best_dice  = 0.0
    best_epoch = 0
    history    = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_dice, va_iou = evaluate(model, test_loader, criterion, DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        elapsed = time.time() - t0
        is_best = va_dice > best_dice
        if is_best:
            best_dice = va_dice
            best_epoch = epoch

        epoch_log = dict(epoch=epoch,
                         lr=current_lr,
                         elapsed_sec=elapsed,
                         tr_loss=tr_loss, tr_dice=tr_dice, tr_iou=tr_iou,
                         va_loss=va_loss, va_dice=va_dice, va_iou=va_iou,
                         best_val_dice=best_dice,
                         is_best=int(is_best))
        history.append(epoch_log)
        _append_epoch_log(log_csv_path, epoch_log)

        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  {elapsed:.1f}s"
              f"  | Train Loss={tr_loss:.4f}  Dice={tr_dice:.4f}  IoU={tr_iou:.4f}"
              f"  | Val Loss={va_loss:.4f}  Dice={va_dice:.4f}  IoU={va_iou:.4f}")

        # ── Visualize progress mỗi epoch ──
        with torch.no_grad():
            model.eval()
            inp = fixed_img.unsqueeze(0).to(DEVICE)   # [1,3,H,W]
            pred_logits = model(inp).squeeze(0)        # [1,H,W]
            model.train()

        save_epoch_plot(
            model_name  = model_name,
            backbone_name = backbone_label,
            epoch       = epoch,
            log_path    = log_csv_path,
            sample_img  = fixed_img,       # [3,H,W]  cpu tensor
            sample_mask = fixed_mask,      # [1,H,W]  cpu tensor
            sample_pred = pred_logits.cpu()  # [1,H,W]  cpu tensor
        )
        save_prediction_mask_frame(
            model_name=model_name,
            epoch=epoch,
            sample_pred=pred_logits.cpu()
        )

        # ── Save best checkpoint ──
        if is_best:
            ckpt_path = os.path.join(SAVE_DIR, f'{model_name}_best.pth')
            torch.save({
                'epoch'          : epoch,
                'model_name'     : model_name,
                'backbone'       : backbone_label,
                'state_dict'     : model.state_dict(),
                'optimizer'      : optimizer.state_dict(),
                'scheduler'      : scheduler.state_dict(),
                'best_dice'      : best_dice,
                'history'        : history,
                'training_config': metadata,
            }, ckpt_path)
            print(f"    ✓ Saved best checkpoint → {ckpt_path}  (Val Dice={best_dice:.4f})")

    _write_training_summary(log_json_path, metadata, history, best_epoch, best_dice)
    print(f"\n  {model_name} — Best Val Dice: {best_dice:.4f}")
    return history
