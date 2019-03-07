import tensorflow as tf
slim = tf.contrib.slim

from .fashion_net_base import fashion_net_base
from .lyst_external_helpers import LystExternal
from .open_images_helpers import OpenImages


MODULES_SPECS = {
    'fashion': LystExternal.modules_specs,
    'open_images': OpenImages.modules_specs,
}


def fashion_net(inputs,
                modules_specs=MODULES_SPECS,
                is_training=False,
                dropout_keep_prob=0.6):
  """Defines the fashion_open_images_net architecture.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether is training or not.

  Returns:
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  print('> modules_specs:', MODULES_SPECS)

  common_layer, end_points = fashion_net_base(
      inputs,
      common_layer_activation_fn=tf.nn.relu,
      is_training=is_training)

  with tf.variable_scope('fashion_net_base'):
    end_point = 'common_layer_dimred'
    common_layer_dimred = slim.fully_connected(
        common_layer, 128,
        activation_fn=None, scope=end_point)
    end_point_dropout = 'common_layer_dimred_dropout'
    common_layer_dimred = slim.dropout(
        common_layer_dimred, dropout_keep_prob,
        is_training=is_training, scope=end_point_dropout)
    end_points[end_point] = common_layer_dimred

  with tf.variable_scope('fashion_open_images_net'):
    # fashion net branch
    with tf.variable_scope('fashion_net'):
      end_point = 'fashion_common_layer'
      fashion_common_layer = slim.fully_connected(
          common_layer_dimred, 128,
          activation_fn=None, scope=end_point)
      end_point_dropout = 'fashion_common_layer_dropout'
      fashion_common_layer = slim.dropout(
          fashion_common_layer, dropout_keep_prob,
          is_training=is_training, scope=end_point_dropout)
      end_points[end_point] = fashion_common_layer

      with tf.variable_scope('modules'):
        for mod_name, mod_size in MODULES_SPECS['fashion']:
          end_point = '%s_logits' % mod_name
          net = slim.fully_connected(fashion_common_layer, mod_size,
                                     activation_fn=None, scope=end_point)
          end_points[end_point] = net

          end_point = '%s_prob' % mod_name
          net = slim.softmax(net, scope=end_point)
          end_points[end_point] = net

    # open images net branch
    with tf.variable_scope('open_images_net'):
      end_point = 'open_images_common_layer'
      open_images_common_layer = slim.fully_connected(
          common_layer_dimred, 128,
          activation_fn=None, scope=end_point)
      end_point_dropout = 'open_images_common_layer_dropout'
      open_images_common_layer = slim.dropout(
          open_images_common_layer, dropout_keep_prob,
          is_training=is_training, scope=end_point_dropout)
      end_points[end_point] = open_images_common_layer

      with tf.variable_scope('modules'):
        for mod_name, mod_size in MODULES_SPECS['open_images']:
          end_point = '%s_logits' % mod_name
          net = slim.fully_connected(open_images_common_layer, mod_size,
                                     activation_fn=None, scope=end_point)
          end_points[end_point] = net

          end_point = '%s_prob' % mod_name
          net = slim.softmax(net, scope=end_point)
          end_points[end_point] = net

  return end_points


fashion_net.default_image_size = 224
