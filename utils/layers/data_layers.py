from __future__ import print_function
import sys
import os
import yaml
import cv2
import numpy as np
from collections import defaultdict
from utils import root_path, caffe_path
sys.path.insert(0, caffe_path)
import caffe


def blob_load_image_with_resize(sample, index, flags, width, height,
                                mean=0.0, std=1.0):
    image_path = sample[index]
    image = cv2.imread(image_path, flags)
    image = cv2.resize(image, (width, height))
    image = (image - mean) / std
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    return image


def blob_load_discrete_aus(sample, index):
    aus = sample[index]
    return aus

def blob_load_discrete_ids(sample, index):
    ids = sample[index]
    return ids


def make_positive_pairs(subject_samples):
    positive_pairs = list()
    for v in subject_samples.values():
        np.random.shuffle(v)
        num_pairs = len(v) // 2
        subject_pairs = [(v[2*i], v[2*i+1]) for i in xrange(num_pairs)]
        positive_pairs.append(subject_pairs)
    positive_pairs = reduce(lambda x,y: x+y, positive_pairs)
    np.random.shuffle(positive_pairs)
    return positive_pairs


class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if not hasattr(self, 'name'):
            self.name = 'DataLayer'
        print('{} setting up!'.format(self.name))
        layer_params = yaml.load(self.param_str)
        self.processed_num = np.inf
        self.epoch = -1
        self.init_params(layer_params)
        file_dict = {k: v for k, v in layer_params.iteritems() if k.endswith('_file')}
        self.prepare_data_from_files(file_dict)
        assert(hasattr(self, 'batch_size') and hasattr(self, 'sample_list'))
        self.num_samples = len(self.sample_list)
        self.index_list  = np.arange(self.num_samples)
        self.post_epoch()
        self.blob_info, self.blob_func = self.init_top_blobs(top)
        return

    def forward(self, bottom, top):
        blobs = self.make_batch()
        for name, index in self.blob_info.iteritems():
            blob = blobs[name]
            top[index].reshape(*blob.shape)
            top[index].data[...] = blob.astype(np.float32, copy=False)
        return

    def backward(self, bottom, top):
        return

    def reshape(self, bottom, top):
        return

    def make_batch(self):
        """ Note that self.processed_num and self.num_sampels are used in train.py
            for showing the training progress
        """
        # Get current batch of samples
        cur_batch = np.min([self.batch_size,
                            self.num_samples - self.processed_num])
        index_begin = self.processed_num
        cur_inds = self.index_list[index_begin:index_begin+cur_batch]
        cur_samples = [self.sample_list[i] for i in cur_inds]

        # Get current batch of data
        blobs = dict()
        for name, (func, kwargs) in self.blob_func.iteritems():
            blobs[name] = map(lambda s: func(s, **kwargs), cur_samples)
        blobs = self.post_batch(blobs)
        self.processed_num += cur_batch
        self.post_epoch()
        return blobs

    def init_params(self, layer_params):
        """
        """
        raise NotImplementedError('init_params is not implemented!')
        return

    def prepare_data_from_files(self, file_dict):
        """
        """
        raise NotImplementedError('prepare_data_from_files is not implemented!')
        return

    def init_top_blobs(self, top):
        """
        """
        raise NotImplementedError('init_top_blobs is not implemented!')
        return

    def post_batch(self, blobs):
        """
        """
        raise NotImplementedError('post_batch is not implemented!')
        return

    def post_epoch(self):
        """
        """
        if self.processed_num >= self.num_samples:
            self.epoch += 1
            self.processed_num = 0
            np.random.shuffle(self.index_list)
        return

    @staticmethod
    def get_input_struct(AUs):
        """
        """
        raise NotImplementedError('get_input_struct is not implemented!')
        return


class ImageAUsDataLayer(DataLayer):
    def setup(self, bottom, top):
        """ Parameters:
                Required-> batch_size, height, width, data_file
                Optional-> grayscale
        """
        self.name = 'ImageAUsDataLayer'
        super(ImageAUsDataLayer, self).setup(bottom, top)
        return

    def init_params(self, layer_params):
        self.batch_size = layer_params['batch_size']
        self.height = layer_params['height']
        self.width  = layer_params['width']
        if 'grayscale' in layer_params and layer_params['grayscale']:
            self.imread_flags = cv2.IMREAD_GRAYSCALE
            self.ch = 1
        else:
            self.imread_flags = cv2.IMREAD_COLOR
            self.ch = 3
        return

    def prepare_data_from_files(self, file_dict):
        self.sample_list = list()
        for file_key, file_val in file_dict.iteritems():
            if file_key == 'data_file':
                with open(os.path.join(root_path, file_val), 'r') as f:
                    line = f.readline()  # The first line contains column names
                    tokens = line.strip().split(',')
                    self.au_names = tokens[1:]
                    while True:
                        line = f.readline()
                        if line == '': break
                        tokens = line.strip().split(',')
                        sample = (tokens[0], [int(x) for x in tokens[1:]])
                        self.sample_list.append(sample)
            else:
                raise ValueError('{} is not supported!'.format(file_key))
        return

    def init_top_blobs(self, top):
        blob_info = dict(data=0, labels=1)

        d_kwargs = dict(index=0, flags=self.imread_flags, std=255.0,
                        width=self.width, height=self.height)
        l_kwargs = dict(index=1)
        blob_func = dict(data=(blob_load_image_with_resize, d_kwargs),
                         labels=(blob_load_discrete_aus, l_kwargs))

        top[0].reshape(1, self.ch, self.height, self.width)
        top[1].reshape(1, len(self.au_names))
        return blob_info, blob_func

    def post_batch(self, blobs):
        blobs['data']   = np.array(blobs['data'], dtype=np.float32)
        blobs['data']   = np.transpose(blobs['data'], (0, 3, 1, 2))
        blobs['labels'] = np.array(blobs['labels'], dtype=np.float32)
        return blobs

    @staticmethod
    def get_input_struct(AUs):
        input_struct =  ['image_path']
        input_struct += ['AU{}'.format(au) for au in AUs]
        return input_struct


class ImageIdAUsDataLayer(DataLayer):
    def setup(self, bottom, top):
        """ Parameters:
                Required-> batch_size, height, width, data_file
                Optional-> grayscale
        """
        self.name = 'ImageIdAUsDataLayer'
        super(ImageIdAUsDataLayer, self).setup(bottom, top)
        return

    def init_params(self, layer_params):
        self.batch_size = layer_params['batch_size']
        self.height = layer_params['height']
        self.width  = layer_params['width']
        if 'grayscale' in layer_params and layer_params['grayscale']:
            self.imread_flags = cv2.IMREAD_GRAYSCALE
            self.ch = 1
        else:
            self.imread_flags = cv2.IMREAD_COLOR
            self.ch = 3
        return

    def prepare_data_from_files(self, file_dict):
        self.sample_list = list()
        for file_key, file_val in file_dict.iteritems():
            if file_key == 'data_file':
                with open(os.path.join(root_path, file_val), 'r') as f:
                    line = f.readline()  # The first line contains column names
                    tokens = line.strip().split(',')
                    self.au_names = tokens[2:]
                    self.id_index = defaultdict(lambda: len(self.id_index))
                    while True:
                        line = f.readline()
                        if line == '': break
                        tokens = line.strip().split(',')
                        id_labels = self.id_index[tokens[1]]
                        au_labels = [int(x) for x in tokens[2:]]
                        sample = (tokens[0], id_labels, au_labels)
                        self.sample_list.append(sample)
            else:
                raise ValueError('{} is not supported!'.format(file_key))
        return

    def init_top_blobs(self, top):
        blob_info = dict(data=0, id_labels=1, au_labels=2)
        d_kwargs = dict(index=0, flags=self.imread_flags, std=255.0,
                        width=self.width, height=self.height)
        idl_kwargs = dict(index=1)
        aul_kwargs = dict(index=2)

        blob_func = dict(data=(blob_load_image_with_resize, d_kwargs),
                         id_labels=(blob_load_discrete_ids, idl_kwargs),
                         au_labels=(blob_load_discrete_aus, aul_kwargs))

        top[0].reshape(1, self.ch, self.height, self.width)
        top[1].reshape(1, 1)
        top[2].reshape(1, len(self.au_names))
        return blob_info, blob_func

    def post_batch(self, blobs):
        blobs['data']      = np.array(blobs['data'], dtype=np.float32)
        blobs['data']      = np.transpose(blobs['data'], (0, 3, 1, 2))
        blobs['id_labels'] = np.array(blobs['id_labels'], dtype=np.float32)
        blobs['au_labels'] = np.array(blobs['au_labels'], dtype=np.float32)
        return blobs

    @staticmethod
    def get_input_struct(AUs):
        input_struct =  ['image_path', 'Id']
        input_struct += ['AU{}'.format(au) for au in AUs]
        return input_struct


class ImageIdAUsTripletDataLayer(DataLayer):
    def setup(self, bottom, top):
        """ Parameters:
                Required-> batch_size, height, width, data_file
                Optional-> grayscale
        """
        self.name = 'ImageIdAUsTripletDataLayer'
        super(ImageIdAUsTripletDataLayer, self).setup(bottom, top)
        return

    def init_params(self, layer_params):
        self.batch_size = layer_params['batch_size']
        self.height = layer_params['height']
        self.width  = layer_params['width']
        if 'grayscale' in layer_params and layer_params['grayscale']:
            self.imread_flags = cv2.IMREAD_GRAYSCALE
            self.ch = 1
        else:
            self.imread_flags = cv2.IMREAD_COLOR
            self.ch = 3
        return

    def prepare_data_from_files(self, file_dict):
        self.sample_list = list()
        self.sample_list_id_only = list()
        self.subject_samples = defaultdict(lambda: [])
        self.subject_samples_id_only = defaultdict(lambda: [])
        for file_key, file_val in file_dict.iteritems():
            if file_key == 'data_file':
                with open(os.path.join(root_path, file_val), 'r') as f:
                    line = f.readline()  # The first line contains column names
                    tokens = line.strip().split(',')
                    self.au_names = tokens[2:]
                    self.id_index = defaultdict(lambda: len(self.id_index))
                    while True:
                        line = f.readline()
                        if line == '': break
                        tokens = line.strip().split(',')
                        id_labels = self.id_index[tokens[1]]
                        au_labels = [int(x) for x in tokens[2:]]
                        sample = (tokens[0], id_labels, au_labels)
                        have_au_labels = (np.sum(au_labels) != -1*len(self.au_names))
                        if have_au_labels:
                            count = len(self.sample_list)
                            self.sample_list.append(sample)
                            self.subject_samples[id_labels].append(count)
                        else:
                            count = len(self.sample_list_id_only)
                            self.sample_list_id_only.append(sample)
                            self.subject_samples_id_only[id_labels].append(count)
            else:
                raise ValueError('{} is not supported!'.format(file_key))
        print('#Samples with au label: {}'.format(len(self.sample_list)))
        print('#Ids: {}, #AUs: {}'.format(len(self.id_index), len(self.au_names)))
        self.pos_pairs = list()
        self.pos_pairs_id_only = list()
        self.processed_pos_pairs_num = np.inf
        self.processed_pos_pairs_num_id_only = np.inf
        return

    def init_top_blobs(self, top):
        blob_info = dict(data=0, id_labels=1, au_labels=2)
        d_kwargs = dict(index=0, flags=self.imread_flags, std=255.0,
                        width=self.width, height=self.height)
        idl_kwargs = dict(index=1)
        aul_kwargs = dict(index=2)

        blob_func = dict(data=(blob_load_image_with_resize, d_kwargs),
                         id_labels=(blob_load_discrete_ids, idl_kwargs),
                         au_labels=(blob_load_discrete_aus, aul_kwargs))

        top[0].reshape(1, self.ch, self.height, self.width)
        top[1].reshape(1, 1)
        top[2].reshape(1, len(self.au_names))
        return blob_info, blob_func

    def post_batch(self, blobs):
        blobs['data']      = np.array(blobs['data'], dtype=np.float32)
        blobs['data']      = np.transpose(blobs['data'], (0, 3, 1, 2))
        blobs['id_labels'] = np.array(blobs['id_labels'], dtype=np.float32)
        blobs['au_labels'] = np.array(blobs['au_labels'], dtype=np.float32)
        return blobs

    def make_batch(self):
        """ Triplet batches, first half is the samples with au labels,
            and the second half is the samples with id labels only
        """
        num_pos_pairs = self.batch_size // 2 // 2
        first_begin  = self.processed_pos_pairs_num
        second_begin = self.processed_pos_pairs_num_id_only
        first_half  = np.min([len(self.pos_pairs)-first_begin,
                              num_pos_pairs])
        second_half = np.min([len(self.pos_pairs_id_only)-second_begin,
                              num_pos_pairs])
        first_inds  = self.pos_pairs[first_begin:first_begin+first_half]
        second_inds = self.pos_pairs_id_only[second_begin:second_begin+second_half]
        first_inds  = reduce(lambda x,y: x+y, first_inds)
        second_inds = reduce(lambda x,y: x+y, second_inds)
        first_samples  = map(lambda x: self.sample_list[x], first_inds)
        second_samples = map(lambda x: self.sample_list_id_only[x], second_inds)

        blobs = dict()
        for name, (func, kwargs) in self.blob_func.iteritems():
            blobs[name] = (map(lambda s: func(s, **kwargs), first_samples) +
                           map(lambda s: func(s, **kwargs), second_samples))
        blobs = self.post_batch(blobs)
        self.processed_num += first_half * 2
        self.processed_pos_pairs_num += first_half
        self.processed_pos_pairs_num_id_only += second_half
        self.post_epoch()
        return blobs

    def post_epoch(self):
        if self.processed_pos_pairs_num >= len(self.pos_pairs):
            self.epoch += 1
            self.processed_num = 0
            self.processed_pos_pairs_num = 0
            self.pos_pairs = make_positive_pairs(self.subject_samples)
        if self.processed_pos_pairs_num_id_only >= len(self.pos_pairs_id_only):
            self.processed_pos_pairs_num_id_only = 0
            self.pos_pairs_id_only = make_positive_pairs(self.subject_samples_id_only)
        return

    @staticmethod
    def get_input_struct(AUs):
        input_struct =  ['image_path', 'Id']
        input_struct += ['AU{}'.format(au) for au in AUs]
        return input_struct
