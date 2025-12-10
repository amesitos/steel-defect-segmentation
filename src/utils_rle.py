import numpy as np

def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decodifica una máscara RLE a una matriz binaria.
    """
    if mask_rle is None or mask_rle == "" or not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T


def rle_encode(mask):
    """
    Codifica una máscara binaria en formato RLE.
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
