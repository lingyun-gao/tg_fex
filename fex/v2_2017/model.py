import tensorflow as tf
slim = tf.contrib.slim


class Model(object):

    def __init__(self, batch_size):
        from .fashion_net import fashion_net
        from .fashion_open_images_helpers import FashionOpenImages

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_config)

        self.inputs = tf.placeholder(tf.float32,
                                     shape=(batch_size, 224, 224, 3))
        self.net = fashion_net(self.inputs)

        self.variables_to_restore = slim.get_variables_to_restore()

        self.maps = FashionOpenImages
        self.modules_specs = FashionOpenImages.modules_specs
        self.idx2name = FashionOpenImages.idx2name

    def load(self, checkpoint_path):
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path,
            self.variables_to_restore
        )

        self.sess.run(init_assign_op, init_feed_dict)

        print('Loaded feature extractor params.')
