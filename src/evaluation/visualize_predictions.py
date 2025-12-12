# visualize_predictions.py

import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.unet_resnet18 import build_model
from data.dataset import SteelDefectDataset
from data.utils_rle import rle_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Usando dispositivo para predicción: {DEVICE}")

SAVE_DIR = "outputs/predictions/"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------
# COLORES PARA EL OVERLAY (BGR porque usamos OpenCV)
# --------------------------------------------
CLASS_COLORS = [
    (255, 0, 0),    # Clase 1 → Azul
    (0, 255, 0),    # Clase 2 → Verde
    (0, 255, 255),  # Clase 3 → Amarillo
    (0, 0, 255)     # Clase 4 → Rojo
]


def create_overlay(image_gray, pred_mask):
    """
    Crea overlay coloreado encima de la imagen original.
    """
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()

    for i in range(4):
        class_pred = (pred_mask[i] > 0.5).astype(np.uint8)  # Umbral
        color = CLASS_COLORS[i]

        # Colorear píxeles donde la clase está presente
        mask_colored = np.zeros_like(image_color)
        mask_colored[class_pred == 1] = color

        # Mezclar con transparencia
        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.5, 0)

    return overlay


def show_prediction(image, true_mask, pred_mask, overlay, image_id, save=True):

    fig, axes = plt.subplots(5, 3, figsize=(14, 16))
    fig.suptitle(f"Predicciones para {image_id}", fontsize=18)

    classes = ["Clase 1", "Clase 2", "Clase 3", "Clase 4"]

    for i in range(4):
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Imagen Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask[i], cmap="gray")
        axes[i, 1].set_title(f"GT {classes[i]}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask[i], cmap="jet")
        axes[i, 2].set_title(f"Pred {classes[i]}")
        axes[i, 2].axis("off")

    # Última fila: Overlay final
    axes[4, 0].imshow(overlay)
    axes[4, 0].set_title("Overlay Predicción")
    axes[4, 0].axis("off")

    axes[4, 1].axis("off")
    axes[4, 2].axis("off")

    plt.tight_layout()

    if save:
        plt.savefig(f"{SAVE_DIR}/{image_id}_visualization.png", dpi=150)
        plt.close()
    else:
        plt.show()


def run_visualization(num_samples=5):

    df = pd.read_csv("data/val_split.csv")
    image_ids = df["ImageId"].unique()

    print(f"📸 Generando {num_samples} visualizaciones...")

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=DEVICE))
    model.eval()

    for i in range(num_samples):

        image_id = image_ids[i]
        image_path = f"data/train_images/{image_id}"

        # Imagen original
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))

        img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Predicción
        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor))[0].cpu().numpy()

        # Construir la máscara real
        rows = df[df["ImageId"] == image_id]
        true_mask = np.zeros((4, 128, 128), dtype=np.float32)

        for _, row in rows.iterrows():

            encoded = row["EncodedPixels"]
            try:
                class_id = int(row["ClassId"]) - 1
            except:
                print("⚠ Advertencia: ClassId incorrecto detectado, corrigiendo...")
                # En el dataset original, la columna correcta está en 'ClassId' como string tipo "3"
                # Detectamos si hay espacios (entonces NO es un ID)
                val = str(row["ClassId"]).strip()
                if " " in val:
                    # ERROR → usamos el VALOR REAL del split
                    # 🔥 El valor REAL siempre está en df original, pero para predicción
                    # asumimos que todos los renglones con máscaras pertenecen a 1-4.
                    # Usamos row["EncodedPixels"] para detectar clases válidas:
                    # Como estamos visualizando, solo omitimos esta máscara corrupta.
                    class_id = None 
                else:
                    class_id = int(val) - 1

            if class_id is None or not (0 <= class_id <= 3):
                # Si no logramos determinar la clase, simplemente saltamos ese row
                continue

            if isinstance(encoded, str):
                decoded = rle_decode(encoded, shape=(256, 1600))
                decoded = cv2.resize(decoded, (128, 128))
                true_mask[class_id] = decoded

        # Crear overlay coloreado
        overlay = create_overlay(img_resized, pred)

        # Mostrar y guardar resultado
        show_prediction(img_resized, true_mask, pred, overlay, image_id)

        print(f"✔ Guardado: outputs/predictions/{image_id}_visualization.png")


if __name__ == "__main__":
    run_visualization(num_samples=5)
