import os
import urllib

import numpy as np

from config import FEX_VERSION, INPUT_WH
from transformer import Transformer

EMBEDDING_LAYER = 'common_layer'
PCAY_LAYER = 'common_layer_pca'


def get_and_set_model_paths():
    home_path = os.path.expanduser('~')

    models_tmp_dir = os.path.join(
        home_path, 'tg-models-tmp')
    if not os.path.exists(models_tmp_dir):
        print('creating temp directory %s ...' % models_tmp_dir)
        os.makedirs(models_tmp_dir)

    fex_version_ = 'feature-extractor-%s' % FEX_VERSION
    fex_dir = '%s/%s' % (models_tmp_dir, fex_version_)
    if not os.path.exists(fex_dir):
        print('creating feature extractor model directory %s ...' % fex_dir)
        os.makedirs(fex_dir)

    url_root = 'https://s3.amazonaws.com/threadgenius-models'
    fn2urls = {
        'tf_params_pruned_pca.npy':
            '%s/feature-extractor-v1.7/tf_params_pruned_pca.npy' % url_root,
    }
    for fn in fn2urls:
        local_path = os.path.join(fex_dir, fn)
        if not os.path.exists(local_path):
            print('copying %s from s3 ...' % fn)
            urllib.urlretrieve(fn2urls[fn], local_path)

        if fn == 'tf_params_pruned_pca.npy':
            model_path = local_path

    return model_path


class FeatureExtractorModelVersioned(object):
    model_version = FEX_VERSION
    model_description = """model_version %s:
TensorFlow implementation of original feature extractor.
Model has 1024-dim 'common_layer' and a 128-dim
'common_layer_pca' that was surgeried in. Model and PCA
parameters are same as those in v1.7p. Finally, about a
dozen labels are removed compared to v1.7p since they had
too few training samples.""" % FEX_VERSION

    def __init__(self,
                 batch_size=None):

        from model import Model

        self.img_wh = INPUT_WH
        self.batch_size = batch_size

        model_path = get_and_set_model_paths()

        self.model = Model(batch_size=batch_size)
        self.model.load(model_path)

        self.embedding_layer = EMBEDDING_LAYER
        self.pca_layer = PCAY_LAYER
        self.prob_layers = ['%s_prob' %
                            mod_name for mod_name, _ in self.model.modules_specs]
        self.output_layers = [self.embedding_layer, self.pca_layer] + self.prob_layers

        self.idx2name = self.model.idx2name
        self.maps = self.model.maps
        self.sess = self.model.sess

        print('Done initializing feature extractor model.')

        self.t = Transformer()

    def init_batch(self, batch_size):
        return np.zeros((batch_size, self.img_wh, self.img_wh, 3))

    def forward_batch(self,
                      batch_in,
                      layers=[],
                      return_tags=True,
                      return_matrix=False,
                      debug=False):
        '''
        Forward pass a batch of images.

        Returns:
            List of blobs containing layer values for each batch item
            - return_tags: if True, includes text tags for each batch
            - return_matrix: if True, returns full matrix
            - debug: if True, prints tags
        '''
        if not isinstance(layers, list):
            raise Exception('"layers" must be a list')

        if len(layers) == 0:
            layers = self.output_layers

        if return_tags or debug:
            for ln in self.prob_layers:
                if ln not in layers:
                    layers.append(ln)

        forward_pass = self.sess.run(
            [self.model.net[l] for l in layers],
            feed_dict={
                self.model.inputs: batch_in,
            }
        )
        named_forward_pass = dict(zip(layers, forward_pass))
        embeddings = [{l: named_forward_pass[l][b]
                       for l in named_forward_pass}
                      for b in range(len(batch_in))]

        if return_tags or debug:
            tags = []
            for idx, blob in enumerate(embeddings):
                tags.append(self.classify_embedding(blob, debug=debug))

        if return_matrix:
            if return_tags:
                named_forward_pass['tags'] = tags
            return named_forward_pass
        else:
            if return_tags:
                for tag, blob in zip(tags, embeddings):
                    blob['tags'] = tag
            return embeddings

    def forward_pca_batch(self,
                          batch_in,
                          layers=[],
                          return_tags=True,
                          debug=False):
        '''
        Forward pass and PCA a batch of images.

        Returns:
            List of blobs containing layer values for each batch item
            with PCA layer included.
            - return_tags: if True, includes text tags for each batch
            - debug: if True, prints tags
        '''

        if self.pca_layer not in layers:
            layers.append(self.pca_layer)

        return self.forward_batch(batch_in,
                                  layers=layers,
                                  return_tags=return_tags,
                                  debug=debug)

    def transform_forward_img(self,
                              img,
                              layers=[],
                              return_tags=False,
                              debug=False):
        '''
        Forward pass a _single_ image.

        Returns:
            Blob containing layer values for image
        '''
        batch_in = self.t.transform(img)[np.newaxis].astype(np.float32)
        embeddings = self.forward_batch(batch_in,
                                        layers=layers,
                                        return_tags=return_tags,
                                        debug=debug)
        return embeddings[0]  # single image forward pass

    def transform_forward_pca_img(self,
                                  img,
                                  debug=False):
        '''
        Forward pass and PCA a _single_ image.

        Returns:
            Blob containing layer values for image with PCA layer
            included.
        '''
        batch_in = self.t.transform(img)[np.newaxis].astype(np.float32)
        embeddings = self.forward_pca_batch(batch_in,
                                            debug=debug)
        return embeddings[0]  # single image forward pass

    def classify_img(self,
                     img,
                     debug=False):
        embedding = self.transform_forward_img(img,
                                               return_tags=True,
                                               debug=debug)
        return embedding['tags']

    def classify_embedding(self,
                           embedding,
                           debug=False):
        labels = []

        for layer_name in self.prob_layers:
            mod_name = layer_name.replace('_prob', '')
            prob = embedding[layer_name]
            mod_topK = sorted(enumerate(prob), key=lambda (_, v): v, reverse=True)[:6]

            if debug:
                print('[%s]:' % mod_name)

            for i, p in mod_topK:
                conf = float(p)

                if conf < 0.1:
                    continue

                label_name = self.idx2name[mod_name][i] \
                                 .replace('**', '').replace('::', ' ')

                label = {
                    'type': mod_name,
                    'name': label_name,
                    'confidence': conf
                }

                if debug:
                    print('%0.4f - %s' % (conf, label_name))

                labels.append(label)

        sorted_labels = sorted(
            labels, key=lambda x: x['confidence'], reverse=True)

        return sorted_labels
