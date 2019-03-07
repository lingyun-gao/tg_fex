import numpy as np
from skimage.transform import resize

from config import FEX_VERSION, INPUT_WH


class Transformer():
    model_version = FEX_VERSION

    def __init__(self):
        mean = np.array([104.00698793, 116.66876762, 122.67891434])
        self.mean_value = mean[:, np.newaxis, np.newaxis]
        self.img_shape = (INPUT_WH, INPUT_WH)

    def transform(self, img):
        # Takes as input [0, 255] img
        img = np.array(img)
        img = resize(img, self.img_shape, preserve_range=True)
        img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
        img = img[::-1, :, :]
        img -= self.mean_value
        return img

    def init_batch(self, batch_size):
        return np.zeros((batch_size, 3, INPUT_WH, INPUT_WH))
