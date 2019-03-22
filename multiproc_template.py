import numpy as np
import multiprocessing as mp

from __init__ import FeatureExtractorModelVersioned as FeatureExtractorBase
from tg_utils import get_pad_image_from_url

import logging
logger = logging.getLogger(__name__)

# TODO: can be 64 or 128 on GPU machine
batch_size = 2

_model = FeatureExtractorModelVersioned(batch_size=batch_size)
_transformer = _model.t


def transform_img_url_worker(load):
    img_idx = load['img_idx']
    img_url = load['img_url']

    if img_url is None:
        return

    try:
        img = get_pad_image_from_url(img_url)  # pad image into square

        new_img = _transformer.transform(img)

        return img_idx, new_img

    except Exception as exception:
        logger.error(exception)


class Transformer():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pool = mp.Pool(mp.cpu_count())

    def transform_blobs(self, blobs):
        if len(blobs) > self.batch_size:
            raise Exception("Number of blobs must be less than batch size (%d)" % self.batch_size)

        jobs = []
        for img_idx, img_url in enumerate(blobs):
            job = {
                'img_idx': img_idx,
                'img_url': img_url,
            }
            jobs.append(job)

        batch_out = _transformer.init_batch(self.batch_size)

        logger.debug("Transforming batch of size %d..." % len(blobs))
        for idx, load in enumerate(self.pool.map(transform_img_url_worker, jobs)):
            if load is None:
                logger.error('No image content at idx: %d' % idx)
                continue

            img_idx, new_img = load
            batch_out[img_idx] = new_img

        return batch_out

    def __del__(self):
        self.pool.terminate()


class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, net_batch_size, *args, **kwargs):
        self.model = FeatureExtractorBase.latest(
            batch_size=None)

        self.batch_size = net_batch_size

    def forward(self, batch_in):
        if batch_in.shape[0] != self.batch_size:
            raise Exception("Input batch size must be %d" % self.batch_size)

        forward_batch = self.model.forward_batch(batch_in, return_tags=True)

        forwards = []
        for blob in forward_batch:
            forwards.append({
                'embedding': blob[self.model.pca_layer].tolist(),
                'tags': blob['tags']
            })

        return forwards
