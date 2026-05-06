import torch

# ─────────────────────────── CONFIG ────────────────────────────
BACKBONE         = 'resnet34'  # có thể đổi sang 'resnet101' hoặc 'vgg16'
BATCH_SIZE_BY_MODEL = {
    'Unet'        : 6,
    'UnetPlusPlus': 2,
    'Unet3Plus'   : 1,
}
EPOCHS           = 2
LR               = 1e-4
SEED             = 42
CLASS_VALUES_MASK = [255]
MODELS_TO_TRAIN  = list(BATCH_SIZE_BY_MODEL.keys())  # ← tự động lấy tên model từ config batch_size_by_model

DATA_DIR    = 'data'
SAVE_DIR    = 'checkpoints'
PLOT_DIR    = 'plots'          # ← thư mục lưu ảnh visualize
LOG_DIR     = 'logs'           # ← thư mục lưu thông số/metrics training
IMG_SIZE    = (512, 512)
NUM_WORKERS = 2
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
