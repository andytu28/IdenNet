from __future__ import print_function
import sys
from utils import caffe_path
from utils.networks import BaseNetwork
from utils.layers import *
sys.path.insert(0, caffe_path)
import caffe
from caffe import params as p


def MFM(net_spec, input_blob, layer_index):
    n = net_spec
    slice1 = 'slice{}_1'.format(layer_index)
    slice2 = 'slice{}_2'.format(layer_index)
    n[slice1], n[slice2] = L_Slice(input_blob, axis=1, ntop=2)
    output = L_Eltwise(n[slice1], n[slice2], eltwise_op=P.Eltwise.MAX,
                       coeff=[1.0, 1.0])
    return output


class IdenNet(BaseNetwork):
    def __init__(self, num_output=12, mode='train', input_source='file', **kwargs):
        input_names  = ['data', 'id_labels', 'au_labels']
        output_names = dict(train=['id_loss', 'au_loss'], test=['au_prob'])
        if mode == 'train':
            super(IdenNet, self).__init__('IdenNet', mode=mode, input_source=input_source,
                                          input_layer_from_file='L_ImageIdAUsTripletDataLayer',
                                          input_names=input_names, output_names=output_names,
                                          **kwargs)
        else:
            super(IdenNet, self).__init__('IdenNet', mode=mode, input_source=input_source,
                                          input_layer_from_file='L_ImageIdAUsDataLayer',
                                          input_names=input_names, output_names=output_names,
                                          **kwargs)
        self.num_output = num_output
        return

    def build_network(self):
        n = self.net_spec

        lr_mult    = [1, 1]
        decay_mult = [1, 1]
        gauss_001  = dict(type='gaussian', std=0.01)
        const_0    = dict(type='constant', value=0.0)
        xavier     = dict(type='xavier')

        # Layer 1
        n.conv1 = L_Conv(n.data, num_output=96, kernel_size=9, pad=0, group=1, stride=1,
                         lr_mult=lr_mult, decay_mult=decay_mult,
                         w_filler=xavier, b_filler=const_0)
        n.pool1 = L_Pool(n.conv1, kernel_size=2, stride=2, pad=0, pool_op=P.Pooling.MAX)
        n.eltwise1 = MFM(n, n.pool1, 1)

        # Layer 2
        n.conv2 = L_Conv(n.eltwise1, num_output=192, kernel_size=5, pad=0, group=1, stride=1,
                         lr_mult=lr_mult, decay_mult=decay_mult,
                         w_filler=xavier, b_filler=const_0)
        n.pool2 = L_Pool(n.conv2, kernel_size=2, stride=2, pad=0, pool_op=P.Pooling.MAX)
        n.eltwise2 = MFM(n, n.pool2, 2)


        # Layer 3
        n.conv3 = L_Conv(n.eltwise2, num_output=256, kernel_size=5, pad=0, group=1, stride=1,
                         lr_mult=lr_mult, decay_mult=decay_mult,
                         w_filler=xavier, b_filler=const_0)
        n.pool3 = L_Pool(n.conv3, kernel_size=2, stride=2, pad=0, pool_op=P.Pooling.MAX)
        n.eltwise3 = MFM(n, n.pool3, 3)

        # Layer 4
        n.conv4 = L_Conv(n.eltwise3, num_output=384, kernel_size=4, pad=0, group=1, stride=1,
                         lr_mult=lr_mult, decay_mult=decay_mult,
                         w_filler=xavier, b_filler=const_0)
        n.pool4 = L_Pool(n.conv4, kernel_size=2, stride=2, pad=0, pool_op=P.Pooling.MAX)
        n.eltwise4 = MFM(n, n.pool4, 4)

        # Layer 5
        n.fc5 = L_Fc(n.eltwise4, num_output=512, lr_mult=lr_mult, decay_mult=decay_mult,
                     w_filler=xavier, b_filler=const_0)
        n.eltwise5 = MFM(n, n.fc5, 5)
        n.l2norm5 = L_Normalize(n.eltwise5, normalize_type='L2', rescale=1.0)
        n.id_feats = L_Reshape(n.l2norm5, dimensions=[0, -1])

        # Layer 6
        n.fc6 = L_Fc(n.eltwise4, num_output=512, lr_mult=lr_mult, decay_mult=decay_mult,
                     w_filler=gauss_001, b_filler=const_0)
        n.eltwise6 = MFM(n, n.fc6, 6)
        n.diff = L_Eltwise(n.eltwise6, n.id_feats, eltwise_op=P.Eltwise.SUM, coeff=[1.0, -1.0])

        # Layer 7
        n.fc7 = L_Fc(n.diff, num_output=self.num_output, lr_mult=lr_mult, decay_mult=decay_mult,
                     w_filler=gauss_001, b_filler=const_0)
        return

    def build_outputs_for_train(self):
        n = self.net_spec
        output_name1 = self.output_names['train'][0]
        output_name2 = self.output_names['train'][1]
        n[output_name1] = L_RankHardLoss(n.id_feats, n.id_labels, neg_num=4, pair_size=2,
                                         hard_ratio=0.5, rand_ratio=0.5, margin=0.5)
        n[output_name2] = L_SigmoidCrossEntropyLoss(n.fc7, n.au_labels, ignore_label=-1)
        return

    def build_outputs_for_test(self):
        n = self.net_spec
        output_name = self.output_names['test'][0]
        n[output_name] = L_Sigmoid(n.fc7, in_place=False)
        return
