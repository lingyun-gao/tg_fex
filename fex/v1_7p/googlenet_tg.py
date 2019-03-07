# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, linear


def conditional_import(gpu):
    global ConvLayer, InceptionFirstPoolLayer
    if gpu:
        from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
        from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
        InceptionFirstPoolLayer = PoolLayerDNN
    else:
        from lasagne.layers import Conv2DLayer
        # flip convolutional filters during testing since DNN filters are correlations

        class ConvLayer(Conv2DLayer):
            def __init__(self, *args, **kwargs):
                super(ConvLayer, self).__init__(flip_filters=False, *args, **kwargs)
        InceptionFirstPoolLayer = PoolLayer


def init(init_net, layer):
    if init_net is None:
        return {}
    else:
        return {'W': init_net[layer].W,
                'b': init_net[layer].b}


def build_inception_module(name, input_layer, nfilters, init_net):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = InceptionFirstPoolLayer(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1,
                                 **init(init_net, '%s/pool_proj' % name))

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1,
                           **init(init_net, '%s/1x1' % name))

    net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1,
                                  **init(init_net, '%s/3x3_reduce' % name))
    net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1,
                           **init(init_net, '%s/3x3' % name))

    net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1,
                                  **init(init_net, '%s/5x5_reduce' % name))
    net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2,
                           **init(init_net, '%s/5x5' % name))

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
    ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model(input_var, input_size, modules_specs, init_net=None, gpu=True):
    conditional_import(gpu)

    net = {}
    net['input'] = InputLayer(shape=(None, 3, input_size, input_size),
                              input_var=input_var)
    net['conv1/7x7_s2'] = ConvLayer(net['input'], 64, 7, stride=2, pad=3,
                                    **init(init_net, 'conv1/7x7_s2'))
    net['pool1/3x3_s2'] = PoolLayer(net['conv1/7x7_s2'],
                                    pool_size=3,
                                    stride=2,
                                    ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(net['pool1/norm1'], 64, 1,
                                        **init(init_net, 'conv2/3x3_reduce'))
    net['conv2/3x3'] = ConvLayer(net['conv2/3x3_reduce'], 192, 3, pad=1,
                                 **init(init_net, 'conv2/3x3'))
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32],
                                      init_net))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96],
                                      init_net))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48],
                                      init_net))

    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64],
                                      init_net))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64],
                                      init_net))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64],
                                      init_net))

    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128],
                                      init_net))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128],
                                      init_net))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128],
                                      init_net))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])

    net['common_layer'] = DenseLayer(net['pool5/7x7_s1'],
                                     num_units=1024,
                                     nonlinearity=linear,
                                     **init(init_net, 'common_layer'))
    net['common_layer/drop_fc'] = DropoutLayer(net['common_layer'],
                                               p=0.4)

    # OUTPUT MODULES
    for mod_name, mod_size in modules_specs:
            net['%s-classifier' % mod_name] = DenseLayer(
                net['common_layer/drop_fc'],
                num_units=mod_size,
                nonlinearity=linear,
                **init(init_net, '%s-classifier' % mod_name))
            net['%s-prob' % mod_name] = NonlinearityLayer(
                net['%s-classifier' % mod_name],
                nonlinearity=softmax)

    # PATTERN OUTPUT MODULES
    pattern_modules = set(['color', 'pattern'])
    pattern_modules_specs = [(mod_name, mod_size) for mod_name, mod_size in modules_specs
                             if mod_name in pattern_modules]
    net['pattern_fc'] = DenseLayer(net['common_layer/drop_fc'],
                                   num_units=128)
    net['pattern_fc/drop'] = DropoutLayer(net['pattern_fc'],
                                          p=0.4)
    net['pattern_embed'] = DenseLayer(net['pattern_fc/drop'],
                                      num_units=64,
                                      nonlinearity=linear)
    net['pattern_embed/drop'] = DropoutLayer(net['pattern_embed'],
                                             p=0.4)

    for mod_name, mod_size in pattern_modules_specs:
            net['pattern_%s-classifier' % mod_name] = DenseLayer(
                net['pattern_embed/drop'],
                num_units=mod_size,
                nonlinearity=linear)
            net['pattern_%s-prob' % mod_name] = NonlinearityLayer(
                net['pattern_%s-classifier' % mod_name],
                nonlinearity=softmax)

    # dummy layer to join multi-outputs (for saving and loading)
    net['outputs'] = ConcatLayer([net['%s-prob' % m_n] for
                                  m_n, _ in modules_specs] +
                                 [net['pattern_%s-prob' % m_n] for
                                  m_n, _ in pattern_modules_specs])
    return net


def build_pattern_model(input_var, input_size, modules_specs):
    net = {}
    net['input'] = InputLayer(shape=(None, input_size),
                              input_var=input_var)
    net['input/drop'] = DropoutLayer(net['input'],
                                     p=0.4)
    net['pattern_fc'] = DenseLayer(net['input/drop'],
                                   num_units=128)
    net['pattern_fc/drop'] = DropoutLayer(net['pattern_fc'],
                                          p=0.4)
    net['pattern_embed'] = DenseLayer(net['pattern_fc/drop'],
                                      num_units=64,
                                      nonlinearity=linear)
    net['pattern_embed/drop'] = DropoutLayer(net['pattern_embed'],
                                             p=0.4)

    for mod_name, mod_size in modules_specs:
            net['%s-classifier' % mod_name] = DenseLayer(
                net['pattern_embed/drop'],
                num_units=mod_size,
                nonlinearity=linear)
            net['%s-prob' % mod_name] = NonlinearityLayer(
                net['%s-classifier' % mod_name],
                nonlinearity=softmax)

    # dummy layer to make loading easier
    net['outputs'] = ConcatLayer([net['%s-prob' % m_n] for
                                 m_n, _ in modules_specs])

    return net
