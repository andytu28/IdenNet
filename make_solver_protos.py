from __future__ import print_function
import sys
import os
import argparse
from utils import root_path, caffe_path, mkdir_if_not_exists
sys.path.insert(0, caffe_path)
from caffe.proto import caffe_pb2
from ConfigParser import SafeConfigParser


def parse_args():
    parser = argparse.ArgumentParser(description='In make_solver_protos.py')
    parser.add_argument('--train_net', dest='train_net',
                        help='Prototxt of the training network',
                        required=True, type=str)
    parser.add_argument('--mode', dest='mode',
                        help='Use gpu or cpu for training',
                        type=str, default='gpu')
    parser.add_argument('--test_nets', dest='test_nets',
                        help='Prototxts of the testing networks',
                        required=True, type=str, action='append')
    parser.add_argument('--exp_path', dest='exp_path',
                        help='Path of the experiment directory',
                        required=True, type=str)
    parser.add_argument('--config_file', dest='config_file',
                        help='Configuration file storing training parameters',
                        required=True, type=str)
    parser.add_argument('--config_section', dest='config_section',
                        help='Target section in the configuration file',
                        required=True, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def read_param(config_parser, section, key):
    float_param = ['base_lr', 'momentum', 'weight_decay', 'gamma']
    int_param   = ['stepsize']
    if key in float_param:
        value = config_parser.getfloat(section, key)
    elif key in int_param:
        value = config_parser.getint(section, key)
    else:
        value = config_parser.get(section, key)
    return value


def main():
    args = parse_args()
    train_name = os.path.splitext(os.path.basename(args.train_net))[0]
    assert('_Train' in train_name), 'Assume training proto contains \'_Train\' in its name'
    model_prefix = train_name.replace('_Train', '')
    model_dir = os.path.join(root_path, args.exp_path, 'models', model_prefix)
    mkdir_if_not_exists(model_dir)
    output_dir = os.path.join(root_path, args.exp_path, 'solver_protos')
    mkdir_if_not_exists(output_dir)
    assert(args.mode in ['gpu', 'cpu']), 'Invalid value for mode'

    config_parser = SafeConfigParser()
    config_parser.read(os.path.join(root_path, args.config_file))
    s = caffe_pb2.SolverParameter()
    s.train_net = os.path.join(root_path, args.train_net)
    s.test_net.extend([os.path.join(root_path, x) for x in args.test_nets])
    max_iter = 10**8
    s.snapshot_prefix = os.path.join(model_dir, '')
    if args.mode == 'gpu':
        s.solver_mode = caffe_pb2.SolverParameter.GPU
    else:
        s.solver_mode = caffe_pb2.SolverParameter.CPU

    # Setting default parameters
    s.random_seed = 0xCAFFE
    s.test_interval = max_iter+1
    s.test_iter.extend([0 for _ in xrange(len(args.test_nets))])
    s.test_initialization = True
    s.max_iter = max_iter
    s.display = 100
    s.snapshot = max_iter + 1

    # Setting parameters from the config file
    for key in config_parser.options(args.config_section):
        value = read_param(config_parser, args.config_section, key)
        setattr(s, key, value)

    output_name = model_prefix + '_{}.prototxt'.format(args.config_section)
    print('Output: {}'.format(os.path.join(output_dir, output_name)))
    with open(os.path.join(output_dir, output_name), 'w') as f:
        f.write(str(s))
    return


if __name__ == "__main__":
    main()
