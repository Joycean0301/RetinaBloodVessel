"""
Inference script — load a saved checkpoint and predict on test images.

Usage:
    python predict.py --model Unet --img path/to/image.png
    python predict.py --model Unet3Plus --img data/test/image/0.png
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from config import BACKBONE, IMG_SIZE, SAVE_DIR, DEVICE
from model import build_model

# ─── CONFIG (must match training) ─────────────────────────────
THRESHOLD  = 0.5


def load_checkpoint(model_name: str) -> torch.nn.Module:
    ckpt_path = os.path.join(SAVE_DIR, f'{model_name}_best.pth')
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    backbone = ckpt.get('backbone', BACKBONE)

    model = build_model(model_name, backbone).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded {model_name} checkpoint (epoch {ckpt['epoch']}, "
          f"backbone={backbone}, best dice={ckpt['best_dice']:.4f})")
    return model


def preprocess(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    t   = TF.to_tensor(img)
    t   = TF.normalize(t, mean=[0.485, 0.456, 0.406],
                          std =[0.229, 0.224, 0.225])
    return t.unsqueeze(0)          # [1, 3, H, W]


@torch.no_grad()
def predict(model: torch.nn.Module, img_path: str, save_path: str | None = None):
    tensor = preprocess(img_path).to(DEVICE)
    logits = model(tensor)                           # [1, 1, H, W]
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask   = (prob > THRESHOLD).astype(np.uint8) * 255

    result = Image.fromarray(mask)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        result.save(save_path)
        print(f"Saved prediction → {save_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Retina Vessel Prediction')
    parser.add_argument('--model', required=True,
                        choices=['Unet', 'UnetPlusPlus', 'Unet3Plus', 'PSPNet'], help='Model name')
    parser.add_argument('--img',   required=True, help='Path to input image')
    parser.add_argument('--out',   default=None,  help='Output mask path (optional)')
    args = parser.parse_args()

    model = load_checkpoint(args.model)

    out_path = args.out or f'predictions/{args.model}_{os.path.basename(args.img)}'
    predict(model, args.img, save_path=out_path)


if __name__ == '__main__':
    main()
