import os
import cPickle as pickle
from os.path import join, exists
import theano
import theano.tensor as T
from googlenet_tg import build_model

import lasagne
from lasagne.layers import get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.objectives import categorical_crossentropy, \
    categorical_accuracy


def convolved_accuracy(prediction, target, top_k):
    probs = prediction * T.gt(target, 0)
    if top_k == 1:
        top_probs = T.max(probs, axis=-1)
    else:
        # top-k
        top_probs = T.sort(probs, axis=-1)
        top_probs = top_probs[[slice(None) for _ in range(top_probs.ndim - 1)] +
                              [slice(-top_k, None)]].sum(axis=-1)
    return top_probs


class Model(object):
    def __init__(self, batch_size, rgb, img_size, data, modules_weights=None, loss_type='sum', gpu=True):
        self.modules_specs = data.modules_specs
        self.inv_n_labels = 1.0 / data.n_labels
        self.prop_modules = dict([(mod_name, float(mod_size)/data.n_labels) for
                                 mod_name, mod_size in data.modules_specs])
        self.loss_type = loss_type

        # weights on each module
        if modules_weights is not None:
            self.modules_weights = modules_weights
        else:
            modules_weights = dict([(mod_name, 1) for mod_name, _ in self.modules_specs])
        # self.modules_weights = []
        # for mod_name, _ in self.modules_specs:
        #     self.modules_weights.append(modules_weights[mod_name])

        # classifier
        self.input_var = T.tensor4('input_var')
        self.target_vars = {}
        self.idx_vars = {}
        for mod_name, _ in self.modules_specs:
            self.target_vars[mod_name] = T.dmatrix('target_%s_var' % mod_name)
            self.idx_vars[mod_name] = T.bvector('idx_%s_var' % mod_name)

        self.net = build_model(self.input_var, img_size, self.modules_specs, None, gpu)

        # used in the predict_all() class method
        self.predict_all_f = None
        self.idx2name = data.idx2name
        # used in load(), dump() class methods
        self.model_prefix = None

    def function(self, train=False, momentum=0.9):
        deterministic = False if train else True
        learning_rate = T.scalar(dtype=theano.config.floatX)
        mode = theano.compile.get_default_mode()
        mode = mode.including('conv_gemm')

        forward = get_output([self.net['%s-prob' % mod_name] for mod_name, _ in self.modules_specs],
                             deterministic=deterministic)
        probs_full = {}
        for (mod_name, _), forward_i in zip(self.modules_specs, forward):
            probs_full[mod_name] = forward_i

        masked_preds = {}
        masked_targets = {}
        losses = {}
        for mod_name, _ in self.modules_specs:
            masked_preds[mod_name] = probs_full[mod_name][self.idx_vars[mod_name]]
            masked_targets[mod_name] = self.target_vars[mod_name][self.idx_vars[mod_name]]
            losses[mod_name] = categorical_crossentropy(
                masked_preds[mod_name],
                masked_targets[mod_name])

        for mod_name, _ in self.modules_specs:
            if self.loss_type == 'sum':
                losses[mod_name] = losses[mod_name].sum()
            elif self.loss_type == 'mean':
                losses[mod_name] = losses[mod_name].mean()
            elif self.loss_type == 'prop-mean':
                losses[mod_name] = self.prop_modules[mod_name] * losses[mod_name].mean()
            else:
                raise Exception('encountered unknown loss-type: %s' % self.loss_type)
        loss = T.sum([losses[mod_name] * self.modules_weights[mod_name] for
                      mod_name, _ in self.modules_specs])
        if self.loss_type == 'sum':
            loss *= self.inv_n_labels
        elif self.loss_type == 'mean':
            loss *= 0.1
        elif self.loss_type == 'prop-mean':
            pass
            # loss *= 1.0

        all_params = get_all_params(self.net['outputs'], trainable=True)
        if train:
            outputs = [loss] + \
                [losses[mod_name] for mod_name, _ in self.modules_specs]
            updates = lasagne.updates.nesterov_momentum(loss,
                                                        all_params,
                                                        learning_rate,
                                                        momentum)
        else:
            accuracies = {1: {}, 5: {}, 10: {}}
            conv_accuracies = {1: {}, 5: {}, 10: {}}
            for mod_name, _ in self.modules_specs:
                for n_acc in [1, 5, 10]:
                    accuracies[n_acc][mod_name] = categorical_accuracy(
                        masked_preds[mod_name],
                        masked_targets[mod_name],
                        n_acc).mean()
                    conv_accuracies[n_acc][mod_name] = convolved_accuracy(
                        masked_preds[mod_name],
                        masked_targets[mod_name],
                        n_acc).mean()

            outputs = [loss] + \
                [losses[mod_name] for mod_name, _ in self.modules_specs]
            for mod_name, _ in self.modules_specs:
                for acc_type in [accuracies, conv_accuracies]:
                    for n_acc in [1, 5, 10]:
                        outputs += [acc_type[n_acc][mod_name]]
            updates = []

        print('Compiling %s function...' % ('train' if train else 'test'))
        inputs = [self.input_var]
        for mod_name, _ in self.modules_specs:
            inputs += [self.target_vars[mod_name], self.idx_vars[mod_name]]
        inputs += [learning_rate]
        f = theano.function(inputs=inputs,
                            outputs=outputs,
                            updates=updates,
                            mode=mode,
                            allow_input_downcast=True,
                            on_unused_input='ignore')

        return f

    def get_predict_function(self, mod_name=None):
        if mod_name in [mn for mn, _ in self.modules_specs]:
            prediction = get_output(self.net['%s-prob' % mod_name],
                                    deterministic=True)
            return theano.function(inputs=[self.input_var],
                                   outputs=prediction)
        else:
            print('Provide a module name:', [mn for mn, _ in self.modules_specs])

    def get_all_predict_function(self):
        predictions = get_output([self.net['%s-prob' % mod_name] for mod_name, _ in self.modules_specs],
                                 deterministic=True)
        return theano.function(inputs=[self.input_var],
                               outputs=predictions)

    def predict_all(self, image, N=5):
        if image.shape[0] != 1:
            print('this function works only for one image')
            return
        if self.predict_all_f is None:
            self.predict_all_f = self.get_all_predict_function()
        forward = self.predict_all_f(image)
        top_probs = {}
        for (mod_name, _), forward_i in zip(self.modules_specs, forward):
            top_mod_probs = sorted(enumerate(forward_i[0]), key=lambda (_, v): v, reverse=True)[:10]
            top_probs[mod_name] = [(prob, self.idx2name[mod_name][idx]) for idx, prob in top_mod_probs]
        return top_probs

    def set_model_paths(self, model_prefix, modules_weights):
        if all([w == 1 for w in modules_weights]):
            self.model_prefix = model_prefix
        else:
            self.model_prefix = join(model_prefix, '_'.join(map(str, modules_weights)))
            if not exists(self.model_prefix):
                os.mkdir(self.model_prefix)

    def get_model_fn(self, iteration):
        if self.model_prefix is None:
            raise Exception('class variable self.model_prefix is not set; try set_model_paths()')
        return join(self.model_prefix, 'model_iter_%d.pkl' % iteration)

    def dump(self, iteration):
        model_fn = self.get_model_fn(iteration)
        param_values = get_all_param_values(self.net['outputs'])
        pickle.dump(param_values, open(model_fn, 'wb'))
        print('Saved model to:', model_fn)

    def load(self, iteration):
        model_fn = self.get_model_fn(iteration)
        if exists(model_fn):
            param_values = pickle.load(open(model_fn, 'rb'))
            set_all_param_values(self.net['outputs'], param_values)
            print('Set values from:', model_fn)
        else:
            print('No saved model found at %s, nothing to load' % model_fn)
            return
