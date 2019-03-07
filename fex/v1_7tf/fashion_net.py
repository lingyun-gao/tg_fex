import tensorflow as tf
slim = tf.contrib.slim

from .fashion_net_base import fashion_net_base
from .lyst_external_helpers import LystExternal


def fashion_net(inputs,
                modules_specs=LystExternal.modules_specs,
                is_training=False):
  """Defines the fashion_net architecture.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether is training or not.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  print('> modules_specs: %s' % modules_specs)

  common_layer, end_points = fashion_net_base(
      inputs,
      is_training=is_training)

  # # # here we surgically add layers to allow for pca # #

  with tf.variable_scope('fashion_net_base'):
    end_point = 'common_layer_centered'
    net = slim.bias_add(common_layer, scope=end_point)

    end_point = 'common_layer_pca'
    net = slim.fully_connected(net, 128,
                               activation_fn=None, scope=end_point)
    end_points[end_point] = net

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  with tf.variable_scope('fashion_net'):
    with tf.variable_scope('modules'):
      for mod_name, mod_size in modules_specs:
        end_point = '%s_logits' % mod_name
        net = slim.fully_connected(common_layer, mod_size,
                                   activation_fn=None, scope=end_point)
        end_points[end_point] = net

        end_point = '%s_prob' % mod_name
        net = slim.softmax(net, scope=end_point)
        end_points[end_point] = net

  return end_points

fashion_net.default_image_size = 224
