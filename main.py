import os

from config import LOG_DIR, MODELS_TO_TRAIN, PLOT_DIR, SAVE_DIR
from training import train_model
from visualization import save_final_comparison

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS_TO_TRAIN:
        all_results[model_name] = train_model(model_name)

    # ── Final comparison plot ──
    save_final_comparison(MODELS_TO_TRAIN)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'Best Val Dice':>14}  {'Best Val IoU':>12}")
    print(f"  {'-'*40}")
    for model_name, hist in all_results.items():
        best = max(hist, key=lambda x: x['va_dice'])
        print(f"  {model_name:<20}  {best['va_dice']:>14.4f}  {best['va_iou']:>12.4f}")
    print(f"{'='*60}\n")
    print(f"  Plots saved in  : {PLOT_DIR}/")
    print(f"  Logs saved in   : {LOG_DIR}/")
    for model_name in MODELS_TO_TRAIN:
        print(f"    ├── {model_name}/epoch_XXXX.png")
    print(f"    └── comparison_all_models.png")


if __name__ == '__main__':
    main()
