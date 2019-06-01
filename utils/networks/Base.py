from __future__ import print_function
import sys
import importlib
from utils import caffe_path
sys.path.insert(0, caffe_path)
import caffe


class BaseNetwork(object):
    def __init__(self, name, mode='train', input_source='file', input_layer_from_file='',
                 input_names=[], output_names=dict(train=[], test=[]), **kwargs):
        if mode != 'train' and mode != 'test':
            raise ValueError('Network mode should be \'train\' or \'test\'!')
        if input_source != 'file' and input_source != 'python':
            raise ValueError('Network input_source should be \'file\' or \'python\'!')
        self.net_spec = caffe.NetSpec()
        self.name = name
        self.mode = mode
        self.input_source = input_source
        self.input_layer_from_file = input_layer_from_file
        self.input_names  = input_names
        self.output_names = output_names
        self.init_args(kwargs)
        return

    def init_args(self, net_kwargs):
        """
        """
        if self.input_source == 'python':
            input_shape = [dict(dim=shape) for shape in net_kwargs['input_shape']]
            self.input_layer_name = 'L_Input'
            self.input_layer_args = dict(input_shape=input_shape)
        else:
            self.input_layer_name = self.input_layer_from_file
            self.input_layer_args = dict(param_str=net_kwargs['param_str'])
        return

    def build(self):
        layer_utils = importlib.import_module('utils.layers')

        # Build the input layer
        input_layer = getattr(layer_utils, self.input_layer_name)
        input_blobs = input_layer(**self.input_layer_args)
        for index, blob in enumerate(input_blobs):
            self.net_spec[self.input_names[index]] = blob

        # Build the core network architecture
        self.build_network()

        # Build the output layers
        if self.mode == 'train':
            self.build_outputs_for_train()
        else:
            self.build_outputs_for_test()
        return self.input_names[:len(input_blobs)], self.output_names[self.mode]

    def to_str(self):
        return str(self.net_spec.to_proto())

    def build_network(self):
        """
        """
        raise NotImplementedError('build_network is not implemented!')
        return

    def build_outputs_for_train(self):
        """
        """
        raise NotImplementedError('build_outputs_for_train is not implemented!')
        return

    def build_outputs_for_test(self):
        """
        """
        raise NotImplementedError('build_outputs_for_test is not implemented!')
        return
