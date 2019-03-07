import numpy as np
import multiprocessing as mp

from fex import FeatureExtractorModel as FeatureExtractorBase
from fex import Transformer
from tg_utils import get_crop_pad_image_from_blob, get_pad_image_from_blob

IM_WH = 244
_transformer = Transformer((IM_WH, IM_WH))

# TODO: reverting back to Theano until new TF models are tested
# fex_base_model_version = FeatureExtractorBase.latest_model_version()
# _transformer = tg_deployed_models.Transformer.version(fex_base_model_version)

import logging
logger = logging.getLogger(__name__)


def transform_img_cont_worker(load):
    img_idx = load['img_idx']
    img_cont = load['img_cont']
    crop = load['crop']

    if img_cont is None:
        return

    try:
        if crop:
            img = get_crop_pad_image_from_blob(blob=img_cont, crop=crop)
        else:
            img = get_pad_image_from_blob(img_cont)  # pad image into square

        new_img = _transformer.transform(img)

        return img_idx, new_img

    except Exception as exception:
        logger.error(exception)


class Transformer():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pool = mp.Pool(mp.cpu_count())

    def transform_blobs(self, blobs, crops=None):
        if len(blobs) > self.batch_size:
            raise Exception("Number of blobs must be less than batch size (%d)" % self.batch_size)

        if crops:
            if len(crops) != len(blobs):
                raise Exception("Unequal number of crops and blobs.")

        jobs = []
        for img_idx, img_cont in enumerate(blobs):
            job = {
                'img_idx': img_idx,
                'img_cont': img_cont,
                'crop': crops[img_idx] if crops else None
            }
            jobs.append(job)

        # batch_out = _transformer.init_batch(self.batch_size)
        batch_out = np.zeros((self.batch_size, 3, IM_WH, IM_WH), dtype=np.float32)

        logger.debug("Transforming batch of size %d..." % len(blobs))
        for idx, load in enumerate(self.pool.map(transform_img_cont_worker, jobs)):
            if load is None:
                logger.error('No image content at idx: %d' % idx)
                continue

            img_idx, new_img = load
            batch_out[img_idx] = new_img

        return batch_out

    def __del__(self):
        self.pool.terminate()

# TODO: reverting back to Theano until new TF models are tested
# class FeatureExtractor(FeatureExtractorBase):
#     def __init__(self, net_batch_size, *args, **kwargs):
#         self.model = FeatureExtractorBase.latest(
#             batch_size=None)
#
#         self.batch_size = net_batch_size
#
#     def forward(self, batch_in):
#         if batch_in.shape[0] != self.batch_size:
#             raise Exception("Input batch size must be %d" % self.batch_size)
#
#         forward_batch = self.model.forward_batch(batch_in, return_tags=True)
#
#         forwards = []
#         for blob in forward_batch:
#             forwards.append({
#                 'embedding': blob[self.model.pca_layer].tolist(),
#                 'tags': blob['tags']
#             })
#
#         return forwards

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, net_batch_size, use_gpu, *args, **kwargs):

        super(FeatureExtractor, self).__init__(
            batch_size=net_batch_size,
            debug=True,  # setting this to False might reduce latency
            gpu=use_gpu)

        self.batch_size = net_batch_size

    def forward(self, batch_in):
        if batch_in.shape[0] != self.batch_size:
            raise Exception("Input batch size must be %d" % self.batch_size)

        forwards = []
        forward_pca_batch = self.forward_pca_batch(batch_in)

        embeddings = forward_pca_batch['embeddings']
        all_tags = forward_pca_batch['tags']

        for blob, tags in zip(embeddings, all_tags):
            forwards.append({
                'embedding': blob[self.pca_layer].tolist(),
                'tags': tags
            })

        return forwards