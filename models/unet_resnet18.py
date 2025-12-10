# models/unet_resnet50.py

import segmentation_models_pytorch as smp
import torch.nn as nn

def build_model():
    """
    Construye una UNet con backbone ResNet50 preentrenado.
    Output = 4 canales (clases 1,2,3,4).
    """
    model = smp.Unet(
        encoder_name="resnet18",          # backbone
        encoder_weights="imagenet",       # preentrenado
        in_channels=1,                    # imagen en escala de grises
        classes=4,                        # 4 clases de defectos
        activation=None,                  # Usamos logits
        encoder_depth=4,                 # más liviano
        decoder_channels=[128, 64, 32, 16]
    )
    return model
