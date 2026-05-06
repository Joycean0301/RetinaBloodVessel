import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from config import CLASS_VALUES_MASK

# ──────────────────────────── DATASET ───────────────────────────
class RetinaDataset(Dataset):
    """
    Expects folder structure:
        data/RetinaBloodVessel/
            train/image/*.png  &  train/mask/*.png
            test/image/*.png   &  test/mask/*.png
    """
    def __init__(self, root: str, split: str = 'train',
                 img_size: tuple = (256, 256), augment: bool = True):
        self.img_dir  = os.path.join(root, split, 'image')
        self.mask_dir = os.path.join(root, split, 'mask')
        self.img_size = img_size
        self.augment  = augment and (split == 'train')

        self.ids = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        assert len(self.ids) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.ids)

    def _augment(self, img: Image.Image, mask: Image.Image):
        """Simple geometric augmentation (train only)."""
        if torch.rand(1) > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if torch.rand(1) > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        angle = float(torch.empty(1).uniform_(-15, 15))
        img  = TF.rotate(img,  angle)
        mask = TF.rotate(mask, angle)
        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]

        img  = Image.open(os.path.join(self.img_dir,  name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, name)).convert('L')

        img  = img.resize(self.img_size,  Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t  = TF.to_tensor(img)                      # [3, H, W]  float [0,1]
        img_t  = TF.normalize(img_t,
                              mean=[0.485, 0.456, 0.406],
                              std =[0.229, 0.224, 0.225])

        mask_np = np.array(mask, dtype=np.uint8)        # [H, W]
        # Binary mask: pixel == 255 → 1, else 0
        bin_mask = np.zeros_like(mask_np, dtype=np.float32)
        for v in CLASS_VALUES_MASK:
            bin_mask[mask_np == v] = 1.0
        mask_t = torch.from_numpy(bin_mask).unsqueeze(0) # [1, H, W]

        return img_t, mask_t