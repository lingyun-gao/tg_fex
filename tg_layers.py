import tensorflow as tf
slim = tf.contrib.slim

VALID_PADDINGS = set(['SAME', 'VALID'])


# NOTES: even though pad_* functions aren't decorated by add_arg_scope, the
#        convolutions within are within the arg_scope
#
#        layer arguments are copied from:
#        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py


def apply_pad(inputs, padding):
  if isinstance(padding, int):
    pad_conf = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
    inputs = tf.pad(inputs, pad_conf, mode='CONSTANT')
    padding = 'VALID'
  elif not isinstance(padding, str) or padding not in VALID_PADDINGS:
    raise Exception('Encountered invalid padding "%s" in convolution' % padding)

  return inputs, padding


def pad_conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME',
               **kwargs):

  inputs, padding = apply_pad(inputs, padding)

  return slim.conv2d(inputs, num_outputs, kernel_size,
                     stride=stride, padding=padding, **kwargs)


def pad_max_pool2d(inputs, kernel_size, stride=2, padding='VALID',
                   **kwargs):

  inputs, padding = apply_pad(inputs, padding)

  return slim.max_pool2d(inputs, kernel_size,
                         stride=stride, padding=padding, **kwargs)


def pad_avg_pool2d(inputs, kernel_size, stride=2, padding='VALID',
                   **kwargs):

  inputs, padding = apply_pad(inputs, padding)

  return slim.avg_pool2d(inputs, kernel_size,
                         stride=stride, padding=padding, **kwargs)
