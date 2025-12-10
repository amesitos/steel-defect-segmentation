# validate.py

import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader
from models.unet_resnet18 import build_model
from src.dataset import SteelDefectDataset
from src.log_config import setup_logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dice_coef(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def validate(ckpt_path="outputs/checkpoints/model_epoch_10.pth"):

    setup_logger("outputs/logs/validate.log")
    logging.info("Validación iniciada.")

    test_df = pd.read_csv("data/test_split.csv")

    dataset = SteelDefectDataset(test_df, "data/train_images")
    loader  = DataLoader(dataset, batch_size=4, shuffle=False)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    logging.info(f"Modelo cargado desde {ckpt_path}")

    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(loader):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            preds = model(img)

            # ================================
            # Aquí va el cálculo por clase 👇
            # ================================
            for c in range(4):
                pred_mask = (preds[:, c] > 0.3).float().cpu().numpy()
                true_mask = mask[:, c].cpu().numpy()
                d = dice_coef(pred_mask, true_mask)
                logging.info(f"Batch {batch_idx} | Dice Clase {c+1}: {d:.4f}")
            # ================================

    logging.info("Validación completada.")


if __name__ == "__main__":
    validate()
