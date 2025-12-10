import pandas as pd
from src.dataset import SteelDefectDataset

df = pd.read_csv("data/train_split.csv")
ds = SteelDefectDataset(df, "data/train_images")

img, mask = ds[0]

print("Imagen shape:", img.shape)
print("Máscara shape:", mask.shape)
