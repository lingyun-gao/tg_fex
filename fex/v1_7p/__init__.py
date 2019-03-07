import os
import urllib

import numpy as np
from sklearn.externals import joblib

from config import FEX_VERSION, INPUT_WH
from transformer import Transformer

FEX_ITERATION = 60000
EMBEDDING_LAYER = 'common_layer'


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
        'model_iter_%d.pkl' % FEX_ITERATION:
            '%s/feature-extractor-v1.7p/model_iter_60000.pkl' % url_root,
        'canonical_labels.txt':
            '%s/canonical_labels.txt' % url_root,
        'look-product-pca.model':
            '%s/feature-extractor-v1.7p/look-product-pca.model' % url_root,
    }
    for fn in fn2urls:
        local_path = os.path.join(fex_dir, fn)
        if not os.path.exists(local_path):
            print('copying %s from s3 ...' % fn)
            urllib.urlretrieve(fn2urls[fn], local_path)

        if fn == 'look-product-pca.model':
            pca_path = local_path

    return fex_dir, pca_path


class FeatureExtractorModelVersioned(object):
    model_version = FEX_VERSION
    model_description = """model_version %s:
Theano implementation of original feature extractor.
Model has 1024-dim 'common_layer'. A separate PCA model
is trained to get 128-dim embeddings.""" % FEX_VERSION

    def __init__(self, batch_size=1,
                 include_embedding_layer=True,
                 additional_layers=[],
                 debug=False,
                 gpu=True):

        import theano
        from lasagne.layers import get_output
        from data import ReadData as MultiModReadData
        from model import Model as MultiModLasagneModel

        self.img_wh = INPUT_WH
        self.batch_size = batch_size
        self.debug = debug

        fex_dir, pca_path = get_and_set_model_paths()
        rd = MultiModReadData(fex_dir)

        self.model = MultiModLasagneModel(batch_size=batch_size,
                                          rgb=3,
                                          img_size=self.img_wh,
                                          data=rd,
                                          gpu=gpu)
        self.model.set_model_paths(fex_dir, [1, 1, 1, 1])
        self.model.load(FEX_ITERATION)

        self.pca = joblib.load(pca_path)

        self.embedding_layer = EMBEDDING_LAYER
        self.pca_layer = 'pca'

        self.idx2name = self.model.idx2name
        self.layers = []
        if include_embedding_layer:
            self.layers.append(self.embedding_layer)
        if additional_layers:
            for al in additional_layers:
                if al not in self.layers:
                    self.layers.append(al)

        self.prob_layers = ['%s-prob' %
                            mod_name for mod_name, _ in self.model.modules_specs]
        for pl in self.prob_layers:
            if pl not in self.layers:
                self.layers.append(pl)

        outputs = get_output([self.model.net[layer] for layer in self.layers],
                             deterministic=True)

        print('Compiling theano forward pass function...')
        self._forward_fn = theano.function(
            inputs=[self.model.input_var],
            outputs=outputs)

        print('Done initializing feature extractor model.')

        self.t = Transformer()

    def init_batch(self, batch_size):
        return np.zeros((batch_size, 3, self.img_wh, self.img_wh))

    def forward_batch(self, batch_in, return_matrix=False, debug=False, return_probs=False):
        '''
        Forward pass a batch of images.
        Returns: List of dictionaries of {layer name: layer vector}.
                 Optionally return top probabilities.
        '''
        forward_pass = self._forward_fn(batch_in)
        named_forward_pass = zip(self.layers, forward_pass)
        embeddings = [{l: embedding[b]
                       for l, embedding in named_forward_pass}
                      for b in range(len(batch_in))]

        return_blob = dict(named_forward_pass) if return_matrix else embeddings

        tags = []
        for blob in embeddings:
            tags.append(self.classify_embedding(blob))

        output = {
            'blob': return_blob,
            'tags': tags
        }

        if (debug or return_probs) and self.debug:
            # K = 16 for both Caffe and Lasagne
            topK_dicts = []
            for batch_idx, blob in enumerate(embeddings):
                if debug:
                    print('Classifying batch element #%d...' % batch_idx)

                # Get tags
                topK_dict = {}
                for name in self.prob_layers:
                    mod_name = name.replace('-prob', '')
                    prob = blob[name]
                    mod_topK = sorted(enumerate(prob), key=lambda (_, v): v, reverse=True)[:4]

                    if debug:
                        mod_topK_labelled = [('[%s] %s' % (name[0].upper(), self.idx2name[mod_name][i]), float(p))
                                             for i, p in mod_topK]

                        print('[%s]:' % name)
                        for label_name, p in mod_topK_labelled:
                            print('%0.4f - %s' % (p, label_name))

                if return_probs:
                    topK_dicts.append(topK_dict)

                if debug:
                    print()

            if return_probs:
                output['top-k-dict'] = topK_dicts

        return output

    def forward_pca_batch(self, batch_in, debug=False, return_probs=False):
        '''
        Forward pass and PCA a batch of images.
        Returns: List of dictionaries of {layer name: layer vectors},
                 with PCA layer included.
                 Optionally return top probabilities.
        '''
        forward_batch = self.forward_batch(
            batch_in, return_matrix=True, debug=debug, return_probs=return_probs)

        embeddings_dict = forward_batch['blob']
        tags = forward_batch['tags']

        embeddings_dict[self.pca_layer] = self.pca.transform(
            embeddings_dict[self.embedding_layer])

        embeddings = [{l: embedding[b]
                       for l, embedding in embeddings_dict.iteritems()}
                      for b in range(len(batch_in))]

        output = {
            'embeddings': embeddings,
            'tags': tags,
        }

        if return_probs:
            output['top-k-dict'] = forward_batch['top-k-dict']

        return output

    def transform_forward_img(self, img, debug=False):
        '''
        Forward pass a *single* image.
        Returns: Dictionary of {layer name: layer vector}.
        '''
        batch_in = self.t.transform(img)[np.newaxis].astype(np.float32)
        embeddings = self.forward_batch(batch_in, debug=debug)['blob']
        return embeddings[0]  # single image forward pass

    def transform_forward_pca_img(self, img, debug=False):
        '''
        Forward pass and PCA a *single* image.
        Returns: Dictionary of {layer name: layer vectors},
                 with PCA layer included.
        '''
        embeddings = self.transform_forward_img(img, debug)
        embeddings[self.pca_layer] = self.pca.transform(
            embeddings[self.embedding_layer].reshape(1, -1))[0]
        return embeddings

    def classify_img(self, img, debug=False):
        embedding = self.transform_forward_img(img, debug)
        return self.classify_embedding(embedding)

    def classify_embedding(self, embedding):
        labels = []

        for layer_name in self.prob_layers:
            mod_name = layer_name.replace('-prob', '')
            prob = embedding[layer_name]
            mod_topK = sorted(enumerate(prob), key=lambda (_, v): v, reverse=True)[:6]

            for i, p in mod_topK:
                conf = float(p)

                if conf < 0.1:
                    continue

                label_name = self.idx2name[mod_name][i].replace('**', '').replace('::', ' ')
                label_type = layer_name.replace('-prob', '')

                label = {
                    'type': label_type,
                    'name': label_name,
                    'confidence': conf
                }

                labels.append(label)

        sorted_labels = sorted(
            labels, key=lambda x: x['confidence'], reverse=True)

        return sorted_labels
