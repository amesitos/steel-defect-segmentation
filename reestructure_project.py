import os
import shutil

# -------------------------------
# Rutas correctas del proyecto
# -------------------------------
STRUCTURE = {
    "data": ["data/train_images", "data/test_images"],
    "src": [
        "src/dataset.py",
        "src/log_config.py",
        "src/metrics.py",
        "src/utils_rle.py",
        "src/predict.py",
        "src/validate.py",
    ],
    "models": ["models/unet_resnet18.py"],
    "scripts": [
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/visualize_predictions.py",
        "scripts/visualize_predictions_per_epoch.py",
    ],
    "notebooks": [
        "notebooks/01_EDA.ipynb",
        "notebooks/02_Training.ipynb",
        "notebooks/03_LossCurve.ipynb",
        "notebooks/04_Predictions.ipynb",
        "notebooks/05_Evaluation.ipynb",
    ],
    "outputs": [
        "outputs/logs",
        "outputs/plots",
        "outputs/predictions",
        "outputs/progress",
        "outputs/checkpoints",
    ],
}

# -------------------------------
# Carpetas basura a eliminar
# -------------------------------
TRASH = [
    "__pycache__",
    ".ipynb_checkpoints",
    "temp",
    "tmp",
    "outputs/predictions/*.tmp",
]


# -------------------------------
# Funciones
# -------------------------------

def ensure_directories():
    print("📁 Creando estructura profesional...")
    for folder in STRUCTURE:
        os.makedirs(folder, exist_ok=True)
    for group in STRUCTURE.values():
        for path in group:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)


def move_files():
    print("📦 Moviendo archivos a su ubicación profesional...")
    for folder, paths in STRUCTURE.items():
        for path in paths:
            filename = os.path.basename(path)
            if os.path.exists(filename):
                shutil.move(filename, path)
                print(f"✔ Movido: {filename} → {path}")


def clean_trash():
    print("🧹 Eliminando basura...")
    for item in TRASH:
        if "*" in item:
            continue
        if os.path.exists(item):
            shutil.rmtree(item, ignore_errors=True)
            print(f"✖ Eliminado: {item}")


def main():
    ensure_directories()
    move_files()
    clean_trash()
    print("\n✨ Proyecto reestructurado profesionalmente.\n")


if __name__ == "__main__":
    main()
