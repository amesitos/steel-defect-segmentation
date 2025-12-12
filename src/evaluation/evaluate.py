# evaluate.py

import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader

from models.unet_resnet18 import build_model
from data.dataset import SteelDefectDataset
from src.log_config import setup_logger

# ------------------------------
# Métricas
# ------------------------------
def iou_score(pred, mask, eps=1e-6):
    """IoU entre dos máscaras 2D (H x W)."""
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()

    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection

    return (intersection + eps) / (union + eps)

def dice_score(pred, mask, eps=1e-6):
    """Dice Coefficient para dos máscaras 2D."""
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()

    intersection = (pred * mask).sum()

    return (2 * intersection + eps) / (pred.sum() + mask.sum() + eps)



# ------------------------------
# Evaluación
# ------------------------------
def evaluate_model():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Evaluando en: {DEVICE}")

    setup_logger("outputs/logs/eval.log")
    logging.info("===== INICIO DE EVALUACIÓN =====")

    # -----------------------------------
    # Cargar split de validación
    # -----------------------------------
    val_df = pd.read_csv("data/val_split.csv")
    val_dataset = SteelDefectDataset(val_df, "data/train_images", transform=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # -----------------------------------
    # Modelo
    # -----------------------------------
    model = build_model().to(DEVICE)
    checkpoint_path = "outputs/checkpoints/best_model.pth"

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    logging.info(f"Modelo cargado desde: {checkpoint_path}")

    # -----------------------------------
    # Métricas acumuladas
    # -----------------------------------
    NUM_CLASSES = 4
    iou_totals = np.zeros(NUM_CLASSES)
    dice_totals = np.zeros(NUM_CLASSES)
    count = 0

    logging.info(f"Evaluando {len(val_dataset)} imágenes...")

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)
            preds = torch.sigmoid(preds)

            # 4 clases
            for c in range(NUM_CLASSES):
                iou_value = iou_score(preds[0, c], masks[0, c]).item()
                dice_value = dice_score(preds[0, c], masks[0, c]).item()

                iou_totals[c] += iou_value
                dice_totals[c] += dice_value

            count += 1

    # -----------------------------------
    # Resultados finales
    # -----------------------------------
    iou_final = iou_totals / count
    dice_final = dice_totals / count
    miou = iou_final.mean()

    print("\n===== RESULTADOS =====")
    for i in range(NUM_CLASSES):
        print(f"Clase {i+1}: IoU={iou_final[i]:.4f} | Dice={dice_final[i]:.4f}")

    print(f"\n📌 mIoU (promedio): {miou:.4f}")

    # Guardar CSV
    df_metrics = pd.DataFrame({
        "Clase": [1,2,3,4],
        "IoU": iou_final,
        "Dice": dice_final
    })

    df_metrics.to_csv("outputs/logs/validation_metrics.csv", index=False)
    logging.info("Métricas guardadas en outputs/logs/validation_metrics.csv")

    logging.info("===== EVALUACIÓN COMPLETADA =====")


if __name__ == "__main__":
    evaluate_model()
