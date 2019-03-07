import os
import time
from io import StringIO
# from StringIO import StringIO

import io
import numpy as np
import requests
from PIL import Image


def sparkey_exists(root):
    return os.path.exists(root + '.spi') \
        and os.path.exists(root + '.spi')


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def avg_image_border(img):
    left = np.asarray(img)[0, :]
    right = np.asarray(img)[-1, :]
    top = np.asarray(img)[:, 0]
    bottom = np.asarray(img)[:, -1]
    average = np.concatenate((left, right, top, bottom)).mean(axis=0)
    return tuple(map(int, average.tolist()))


def pad_image(img, pad):
    w, h = img.size
    d = max(w, h)
    padding = pad if pad is not None else avg_image_border(img)
    new_im = Image.new('RGB', (d, d), padding)
    new_im.paste(img, ((d - w) / 2, (d - h) / 2))

    return new_im


def open_image_from_path(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def open_pad_image_from_path(img_path, pad=None):
    img = open_image_from_path(img_path)
    return pad_image(img, pad)


def get_image_from_url(img_url):
    n_tries = 0

    while n_tries < 3:
        try:
            r = requests.get(img_url, verify=False)

            img = Image.open(StringIO(r.content))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            return img

        except requests.exceptions.ConnectionError:
            time.sleep(1)
            n_tries += 1

        except Exception:
            break

    print('Couldn\'t download image with url: %s' % img_url)


def get_crop_image_from_url(img_url, crop=None):
    img = get_image_from_url(img_url)

    if img:
        if crop is not None:
            w, h = img.size
            x1, y1 = int(w * crop[0]), int(h * crop[1])
            x2, y2 = int(w * crop[2]), int(h * crop[3])
            img = img.crop((x1, y1, x2, y2))

        return img

def get_pad_image_from_url(img_url, pad=None):
    img = get_image_from_url(img_url)

    if img:
        return pad_image(img, pad)


def get_crop_pad_image_from_url(img_url, crop=None, pad=None):
    img = get_crop_image_from_url(img_url, crop)

    if img:
        return pad_image(img, pad)


def open_crop_pad_image_from_path(img_path, bbox, pad=None):
    img = open_image_from_path(img_path)
    img = img.crop(tuple(bbox))

    return pad_image(img, pad)


def get_image_from_blob(blob):
    image = Image.open(io.BytesIO(blob))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def get_pad_image_from_blob(blob, pad=None):
    img = get_image_from_blob(blob)
    return pad_image(img, pad)


def get_crop_image_from_blob(blob, crop=None):
    """ crop is percentage of full image: x1, y1, x2, y2
    """

    img = get_image_from_blob(blob)

    if crop is not None:
        w, h = img.size
        x1, y1 = int(w * crop[0]), int(h * crop[1])
        x2, y2 = int(w * crop[2]), int(h * crop[3])
        img = img.crop((x1, y1, x2, y2))

    return img


def get_crop_pad_image_from_blob(blob, crop=None, pad=None):
    img = get_crop_image_from_blob(blob, crop)

    return pad_image(img, pad)


def batch_generator(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]
