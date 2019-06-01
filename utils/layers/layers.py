from __future__ import print_function
import functools
import sys
from utils import caffe_path
sys.path.insert(0, caffe_path)
import caffe
from caffe import layers as L, params as P


def get_param(param_name, lr_mult, decay_mult):
    param = [dict(lr_mult=lr_mult[i], decay_mult=decay_mult[i]) for i in xrange(2)]
    if param_name != '':
        param[0]['name'] = '{}_w'.format(param_name)
        param[1]['name'] = '{}_b'.format(param_name)
    return param


def add_activation(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        activation = kwargs.get('activation', '')
        in_place   = kwargs.get('in_place', True)
        kwargs.pop('activation', None)
        kwargs.pop('in_place', None)
        original = func(*args, **kwargs)
        if activation == '':
            output = original
        else:
            activated = getattr(L, activation)(original, in_place=in_place)
            output = (original, activated)
        return output
    return wrapper_func


def L_ImageAUsDataLayer(param_str):
    python_param = dict(module='utils.layers', layer='ImageAUsDataLayer',
                        param_str=str(param_str))
    output = L.Python(python_param=python_param, ntop=2)
    return output


def L_ImageIdAUsTripletDataLayer(param_str):
    python_param = dict(module='utils.layers', layer='ImageIdAUsTripletDataLayer',
                       param_str=str(param_str))
    output = L.Python(python_param=python_param, ntop=3)
    return output


def L_ImageIdAUsDataLayer(param_str):
    python_param = dict(module='utils.layers', layer='ImageIdAUsDataLayer',
                        param_str=str(param_str))
    output = L.Python(python_param=python_param, ntop=3)
    return output


@add_activation
def L_Conv(input_blob, num_output, kernel_size, lr_mult, decay_mult, w_filler, b_filler,
           pad=0, stride=1, group=1, dilation=1, param_name=''):
    conv_param = get_param(param_name, lr_mult, decay_mult)
    output = L.Convolution(input_blob, num_output=num_output, kernel_size=kernel_size,
                           pad=pad, stride=stride, group=group, dilation=dilation,
                           param=conv_param, weight_filler=w_filler, bias_filler=b_filler)
    return output


@add_activation
def L_Fc(input_blob, num_output, lr_mult, decay_mult,
         w_filler, b_filler, param_name=''):
    fc_param = get_param(param_name, lr_mult, decay_mult)
    output = L.InnerProduct(input_blob, num_output=num_output, param=fc_param,
                            weight_filler=w_filler, bias_filler=b_filler)
    return output


def L_Pool(input_blob, kernel_size, pad=0, stride=1, pool_op=P.Pooling.MAX,
           global_pooling=False):
    output = L.Pooling()
    if global_pooling:
        kwargs = dict(pool=pool_op, global_pooling=True)
    else:
        kwargs = dict(kernel_size=kernel_size, pad=pad, stride=stride,
                      pool=pool_op)
    output = L.Pooling(input_blob, **kwargs)
    return output


def L_LRN(input_blob, local_size=5, alpha=1.0, beta=0.75,
          norm_region=P.LRN.ACROSS_CHANNELS, engine=P.LRN.DEFAULT):
    output = L.LRN(input_blob, local_size=local_size, alpha=alpha, beta=beta,
                   norm_region=norm_region, engine=engine)
    return output


def L_Drop(input_blob, dropout_ratio=0.5, in_place=True):
    output = L.Dropout(input_blob, dropout_ratio=dropout_ratio, in_place=in_place)
    return output


def L_SigmoidCrossEntropyLoss(input_scores, input_labels, ignore_label=None):
    if ignore_label is None:
        output = L.SigmoidCrossEntropyLoss(input_scores, input_labels)
    else:
        output = L.SigmoidCrossEntropyLoss(input_scores, input_labels,
                                           loss_param=dict(ignore_label=ignore_label))
    return output


def L_RankHardLoss(input_feats, input_labels, neg_num, pair_size,
                   hard_ratio, rand_ratio, margin):
    rank_hard_loss_param = dict(neg_num=neg_num, pair_size=pair_size,
                                hard_ratio=hard_ratio, rand_ratio=rand_ratio,
                                margin=margin)
    output = L.RankHardLoss(input_feats, input_labels,
                            rank_hard_loss_param=rank_hard_loss_param)
    return output


def L_Sigmoid(input_blob, in_place=True):
    output = L.Sigmoid(input_blob, in_place=in_place)
    return output


def L_ReLU(input_blob, in_place=True):
    output = L.ReLU(input_blob, in_place=in_place)
    return output


def L_Box(input_blob, width, height, xcoord, ycoord):
    assert(len(xcoord) == len(ycoord)), 'Length of xcoord and ycoord should be the same'
    output = L.Box(input_blob, width=width, height=height,
                   xcoord=xcoord, ycoord=ycoord, ntop=len(xcoord))
    return output


def L_Splice(input_blob_list, xcoord, ycoord):
    assert(len(xcoord) == len(ycoord)), 'Length of xcoord and ycoord should be the same'
    output = L.Splice(*input_blob_list, xcoord=xcoord, ycoord=ycoord)
    return output


def L_BatchNorm(input_blob, use_global_stats=None):
    param = [dict(lr_mult=0.0)]*3
    if use_global_stats is None:
        output = L.BatchNorm(input_blob, param=param)
    else:
        output = L.BatchNorm(input_blob, param=param, use_global_stats=use_global_stats)
    return output


def L_Eltwise(input_blob1, input_blob2, eltwise_op, coeff):
    output = L.Eltwise(input_blob1, input_blob2, operation=eltwise_op, coeff=coeff)
    return output


def L_Slice(input_blob, axis=1, ntop=2):
    slice_param = dict(axis=1)
    output = L.Slice(input_blob, slice_param=slice_param, ntop=ntop)
    return output


def L_Normalize(input_blob, normalize_type='L2', rescale=1.0):
    output = L.Normalize(input_blob, normalize_type=normalize_type, rescale=rescale)
    return output


def L_Reshape(input_blob, dimensions):
    output = L.Reshape(input_blob, reshape_param=dict(shape=dict(dim=dimensions)))
    return output
