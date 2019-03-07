from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from .tg_layers import pad_conv2d, pad_max_pool2d, pad_avg_pool2d


def build_inception_module(name, net, nfilters):
  with tf.variable_scope(name):
    with tf.variable_scope('branch_0'):
      branch_0 = slim.conv2d(net, nfilters[0], kernel_size=1, scope='1x1')
    with tf.variable_scope('branch_1'):
      branch_1 = slim.conv2d(net, nfilters[1], kernel_size=1, scope='3x3_reduce')
      branch_1 = pad_conv2d(branch_1, nfilters[2], kernel_size=3,
                            padding=1, scope='3x3')
    with tf.variable_scope('branch_2'):
      branch_2 = slim.conv2d(net, nfilters[3], kernel_size=1, scope='5x5_reduce')
      branch_2 = pad_conv2d(branch_2, nfilters[4], kernel_size=5,
                            padding=2, scope='5x5')
    with tf.variable_scope('branch_3'):
      branch_3 = pad_max_pool2d(net, kernel_size=3, stride=1,
                                padding=1, scope='pool')
      branch_3 = slim.conv2d(branch_3, nfilters[5], kernel_size=1, scope='pool_proj')
    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
  return net


def fashion_net_base(inputs,
                     final_endpoint='common_layer',
                     common_layer_activation_fn=None,
                     is_training=True,
                     dropout_keep_prob=0.6,
                     reuse=False,
                     scope=None):
  """Defines the fashion_net base architecture.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'fashion_net_base', [inputs], reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d],  # NOTE: pad_conv2d's slim.conv2d also inherits this arg_scope
        stride=1,
    ):

      end_point = 'conv1_7x7_s2'
      net = pad_conv2d(inputs, 64, kernel_size=7, stride=2,
                       padding=3, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool1_3x3_s2'
      net = slim.max_pool2d(net, kernel_size=3, stride=2,
                            padding='SAME', scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool1_norm1'
      net = tf.nn.lrn(net, 2, 1.0, 2e-05, 0.75)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'conv2_3x3_reduce'
      net = slim.conv2d(net, 64, kernel_size=1, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'conv2_3x3'
      net = pad_conv2d(net, 192, kernel_size=3,
                       padding=1, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'conv2_norm2'
      net = tf.nn.lrn(net, 2, 1.0, 2e-05, 0.75)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool2_3x3_s2'
      net = slim.max_pool2d(net, kernel_size=3, stride=2, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points


      end_point = 'inception_3a'
      nfilters = [64, 96, 128, 16, 32, 32]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_3b'
      nfilters = [128, 128, 192, 32, 96, 64]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool3_3x3_s2'
      net = slim.max_pool2d(net, kernel_size=3, stride=2, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points


      end_point = 'inception_4a'
      nfilters = [192, 96, 208, 16, 48, 64]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_4b'
      nfilters = [160, 112, 224, 24, 64, 64]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_4c'
      nfilters = [128, 128, 256, 24, 64, 64]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_4d'
      nfilters = [112, 144, 288, 32, 64, 64]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_4e'
      nfilters = [256, 160, 320, 32, 128, 128]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool4_3x3_s2'
      net = slim.max_pool2d(net, kernel_size=3, stride=2, scope=end_point)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points


      end_point = 'inception_5a'
      nfilters = [256, 160, 320, 32, 128, 128]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'inception_5b'
      nfilters = [384, 192, 384, 48, 128, 128]
      net = build_inception_module(end_point, net, nfilters)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points

      end_point = 'pool5_6x6_s1'
      net = pad_avg_pool2d(net, kernel_size=6, stride=6, scope=end_point)
      net = tf.squeeze(net, [1, 2], name='spatial_squeeze')
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points


      end_point = 'common_layer'
      net = slim.fully_connected(net, 1024,
                                 activation_fn=common_layer_activation_fn,
                                 scope=end_point)
      end_point_dropout = 'common_layer_dropout'
      net = slim.dropout(net, dropout_keep_prob,
                         is_training=is_training, scope=end_point_dropout)
      end_points[end_point] = net
      if final_endpoint == end_point: return net, end_points


  raise ValueError('Unknown final endpoint %s' % final_endpoint)
