# visualize_predictions_per_epoch.py

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.unet_resnet18 import build_model
from data.utils_rle import rle_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Usando dispositivo para predicción: {DEVICE}")

# Colores BGR para cada clase
CLASS_COLORS = [
    (255, 0, 0),    # Clase 1 → Azul
    (0, 255, 0),    # Clase 2 → Verde
    (0, 255, 255),  # Clase 3 → Amarillo
    (0, 0, 255)     # Clase 4 → Rojo
]


def overlay_mask(image_gray, pred_mask):
    """Overlay transparente de predicción coloreada."""
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()

    for c in range(4):
        mask = (pred_mask[c] > 0.5).astype(np.uint8)
        color = CLASS_COLORS[c]

        colored = np.zeros_like(image_color)
        colored[mask == 1] = color

        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

    return overlay


def visualize_epoch(model_path, image_id, df, save_dir):

    print(f"📸 Procesando {image_id} usando modelo: {model_path}")

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    image_path = f"data/train_images/{image_id}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))

    # Preparar tensor
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0].cpu().numpy()

    # Máscara real
    true_mask = np.zeros((4, 128, 128), dtype=np.float32)

    rows = df[df["ImageId"] == image_id]

    for _, row in rows.iterrows():

        encoded = row["EncodedPixels"]

        # CORRECCIÓN ClassId corrupto
        try:
            class_id = int(row["ClassId"]) - 1
        except:
            print("⚠ Saltando fila corrupta.")
            continue

        if not (0 <= class_id <= 3):
            continue

        if isinstance(encoded, str):
            decoded = rle_decode(encoded, shape=(256, 1600))
            decoded = cv2.resize(decoded, (128, 128))
            true_mask[class_id] = decoded

    overlay = overlay_mask(img_resized, pred)

    # --- Plot bonito ---
    fig, axes = plt.subplots(5, 3, figsize=(14, 18))
    fig.suptitle(f"Progreso - {os.path.basename(model_path)}", fontsize=18)

    classes = ["Clase 1", "Clase 2", "Clase 3", "Clase 4"]

    for i in range(4):
        axes[i, 0].imshow(img_resized, cmap="gray")
        axes[i, 0].set_title("Imagen Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask[i], cmap="gray")
        axes[i, 1].set_title(f"GT {classes[i]}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred[i], cmap="jet")
        axes[i, 2].set_title(f"Pred {classes[i]}")
        axes[i, 2].axis("off")

    axes[4, 0].imshow(overlay)
    axes[4, 0].set_title("Overlay")
    axes[4, 0].axis("off")

    axes[4, 1].axis("off")
    axes[4, 2].axis("off")

    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{image_id}_epoch_progress.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✔ Guardado: {out_path}")


def run_progress_visualization(num_images=3):
    df = pd.read_csv("data/val_split.csv")
    df.columns = ["ImageId", "EncodedPixels", "ClassId"]  # Orden correcto

    image_ids = df["ImageId"].unique()[:num_images]

    model_dir = "outputs/checkpoints/"
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])

    print(f"🔎 Detectados {len(model_files)} modelos para analizar.")

    for model_name in model_files:
        model_path = os.path.join(model_dir, model_name)
        save_dir = f"outputs/progress/{model_name.replace('.pth','')}"

        for image_id in image_ids:
            visualize_epoch(model_path, image_id, df, save_dir)


if __name__ == "__main__":
    run_progress_visualization(num_images=3)
