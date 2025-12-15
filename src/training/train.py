import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
import csv
import time

from models.unet_resnet18 import build_model
from src.data.dataset import SteelDefectDataset
from src.log_config import setup_logger

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_prediction(model, dataloader, epoch, device="cpu"):
    model.eval()

    # Tomamos SOLO 1 batch del validation set
    images, masks = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = preds.cpu().numpy()

    img = images[0].cpu().numpy().squeeze()
    true_mask = masks[0].cpu().numpy()
    pred_mask = preds[0]

    # Crear figura
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Imagen Original")

    axs[1].imshow(np.max(true_mask, axis=0), cmap="viridis")
    axs[1].set_title("Máscara Real")

    axs[2].imshow(np.max(pred_mask, axis=0), cmap="viridis")
    axs[2].set_title("Predicción del Modelo")

    for ax in axs:
        ax.axis("off")

    save_path = f"outputs/predictions/epoch_{epoch}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()

    logging.info(f"🖼 Imagen de predicción guardada en: {save_path}")


# Selección de dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Usando dispositivo: {DEVICE}")


def train():

    # ===============================
    # Logger
    # ===============================
    setup_logger("outputs/logs/train.log")
    logging.info("Entrenamiento iniciado.")

    # ===============================
    # Cargar splits
    # ===============================
    train_df = pd.read_csv("data/train_split.csv")
    val_df   = pd.read_csv("data/val_split.csv")

    logging.info(f"Train images: {train_df['ImageId'].nunique()}")
    logging.info(f"Val images: {val_df['ImageId'].nunique()}")

    # ===============================
    # CSV de pérdidas
    # ===============================
    loss_log_path = "outputs/logs/loss_history.csv"
    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # ===============================
    # Transformaciones
    # ===============================
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    val_transform = A.Compose([])

    # ===============================
    # Dataset y DataLoader
    # ===============================
    train_dataset = SteelDefectDataset(train_df, "data/train_images", transform=train_transform)
    val_dataset   = SteelDefectDataset(val_df, "data/train_images", transform=val_transform)

    # En CPU recomendamos batch_size=2
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=0)

    # ===============================
    # Crear modelo (SIN compile())
    # ===============================
    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bce  = smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="multilabel")

    def loss_fn(pred, mask):
        return bce(pred, mask) + dice(pred, mask)

    # ===============================
    # Entrenamiento
    # ===============================
    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        logging.info(f"====== EPOCH {epoch}/{epochs} ======")

        model.train()
        train_loss = 0
        start_time = time.time()

        for images, masks in tqdm(train_loader, desc=f"Entrenando Epoch {epoch}"):

            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            # SIN autocast, SIN compile
            preds = model(images)
            loss = loss_fn(preds, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        # ===============================
        # VALIDACIÓN
        # ===============================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validando"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds = model(images)
                val_loss += loss_fn(preds, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Val Loss: {avg_val_loss:.4f}")

        # ===============================
        # Visualización por epoch
        # ===============================
        visualize_prediction(model, val_loader, epoch, device=DEVICE)
            
        # ===============================
        # Guardar CSV
        # ===============================
        with open(loss_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_val_loss])

        # ===============================
        # Guardar mejor modelo
        # ===============================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")
            logging.info("🎉 Nuevo mejor modelo guardado.")

        # Guardar checkpoint normal
        ckpt_path = f"outputs/checkpoints/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)

        elapsed = time.time() - start_time
        logging.info(f"⏱ Tiempo de epoch: {elapsed:.2f}s")

    logging.info("Entrenamiento COMPLETADO.")


if __name__ == "__main__":
    train()
