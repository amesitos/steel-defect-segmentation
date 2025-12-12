import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from data.utils_rle import rle_decode


class SteelDefectDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        # Asegurar columnas correctas
        df.columns = ["ImageId", "ClassId", "EncodedPixels"]

        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # NUEVO: tamaño reducido para CPU
        self.img_size = (128, 128)

    def __len__(self):
        return len(self.df["ImageId"].unique())

    def __getitem__(self, idx):

        image_id = self.df["ImageId"].unique()[idx]
        image_path = f"{self.img_dir}/{image_id}"

        # ===========================
        # CARGAR IMAGEN
        # ===========================
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        # Reducir tamaño aquí ↓
        img = cv2.resize(img, self.img_size)

        # ===========================
        # CREAR MÁSCARA MULTICLASE
        # ===========================
        mask = np.zeros((*self.img_size, 4), dtype=np.uint8)

        rows = self.df[self.df["ImageId"] == image_id]

        for _, row in rows.iterrows():

            try:
                class_id = int(row["ClassId"]) - 1
            except:
                continue

            encoded = row["EncodedPixels"]

            # Si no hay máscara → continuar
            if not isinstance(encoded, str) or encoded.strip() == "":
                continue

            # Decodificar usando tamaño ORIGINAL del dataset (256×1600)
            decoded = rle_decode(encoded, shape=(256, 1600))

            # Reducir tamaño de la máscara a 128×128
            decoded = cv2.resize(decoded, self.img_size, interpolation=cv2.INTER_NEAREST)

            mask[:, :, class_id] = decoded

        # ===========================
        # TRANSFORMACIONES
        # ===========================
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # ===========================
        # NORMALIZAR Y FORMATEAR
        # ===========================
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)  # → (1, 128, 128)

        mask = np.transpose(mask, (2, 0, 1)).astype("float32")  # → (4, 128, 128)

        return torch.tensor(img), torch.tensor(mask)
