import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, batch_size):
        from .fashion_net import fashion_net
        from .lyst_external_helpers import LystExternal

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_config)

        self.inputs = tf.placeholder(tf.float32,
                                     shape=(batch_size, 224, 224, 3))
        self.net = fashion_net(self.inputs)

        self.maps = LystExternal
        self.modules_specs = LystExternal.modules_specs
        self.idx2name = LystExternal.idx2name

    def load(self, params_fn):
        params = np.load(params_fn).item()
        for key in params:
            with tf.variable_scope(key, reuse=True):
                for subkey in params[key]:
                    var = tf.get_variable(subkey)
                    self.sess.run(var.assign(params[key][subkey]))

        print('Loaded feature extractor params.')
