from __future__ import print_function
import sys
import os
import importlib
import argparse
from utils import root_path, mkdir_if_not_exists


def parse_args():
    parser = argparse.ArgumentParser(description='In make_network_protos.py')
    parser.add_argument('--network', dest='network',
                        help='Target network',
                        required=True, type=str)
    parser.add_argument('--exp_path', dest='exp_path',
                        help='Path of the experiment directory',
                        required=True, type=str)
    parser.add_argument('--mode', dest='mode',
                        help='Mode for constructing the network (train or test)',
                        required=True, type=str)
    parser.add_argument('--input_source', dest='input_source',
                        help='Input source for constructing the network (file or python)',
                        required=True, type=str)
    parser.add_argument('--num_output', dest='num_output',
                        help='Dimension of the output features of the network',
                        required=True, type=int)
    parser.add_argument('--param_str', dest='param_str',
                        help='Parameter string of the network with \'file\' as input_source',
                        type=str, nargs=2, action='append', default=list())
    parser.add_argument('--input_shape', dest='input_shape',
                        help='Input shape of the network with \'python\' as input_source',
                        type=int, nargs='+', action='append', default=list())
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def cast_param(key, val):
    int_param  = ['batch_size', 'height', 'width']
    bool_param = ['grayscale']
    if key in int_param:
        val = int(val)
    elif key in bool_param:
        val = True if (val == 'True' or val == 'true') else False
    return val


def main():
    args = parse_args()
    network_handler = getattr(importlib.import_module('utils.networks'), args.network)
    output_dir = os.path.join(root_path, args.exp_path, 'network_protos')
    mkdir_if_not_exists(output_dir)
    assert(args.mode in ['train', 'test']), 'Invalid value for mode'
    assert(args.input_source in ['file', 'python']), 'Invalid value for input_source'

    if args.mode == 'train' and args.input_source == 'python':
        raise ValueError('Not allow training with \'python\' as input_source')

    param_str   = {p[0]: cast_param(p[0], p[1]) for p in args.param_str}
    input_shape = [[int(x) for x in s] for s in args.input_shape]

    if args.input_source == 'file':
        network_kwargs = dict(param_str=param_str)
        output_name = os.path.splitext(os.path.basename(param_str['data_file']))[0] + '.prototxt'
    else:
        network_kwargs = dict(input_shape=input_shape)
        output_name = '{}_output{}.prototxt'.format(args.network, args.num_output)

    network = network_handler(num_output=args.num_output, mode=args.mode,
                              input_source=args.input_source, **network_kwargs)
    network.build()
    print('Output: {}'.format(os.path.join(output_dir, output_name)))
    with open(os.path.join(output_dir, output_name), 'w') as f:
        f.write(network.to_str())
    return


if __name__ == "__main__":
    main()
