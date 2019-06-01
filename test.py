from __future__ import print_function
import sys
import os
import argparse
import cPickle
import h5py
import numpy as np
from utils import root_path, caffe_path, mkdir_if_not_exists
from utils.metrics import calc_f1_with_name
sys.path.insert(0, caffe_path)
import caffe


def parse_args():
    parser = argparse.ArgumentParser(description='In test.py')
    parser.add_argument('--mode', dest='mode',
                        help='Use gpu or cpu for training',
                        type=str, default='gpu')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='Target gpu_id when multiple gpus are available',
                        type=int, default=-1)
    parser.add_argument('--test_net', dest='test_net',
                        help='The prototxt of the testing network',
                        required=True, type=str)
    parser.add_argument('--model_prefix', dest='model_prefix',
                        help='Prefix of the testing model',
                        required=True, type=str)
    parser.add_argument('--exp_path', dest='exp_path',
                        help='Path of the experiment directory',
                        required=True, type=str)
    parser.add_argument('--store_preds', dest='store_preds',
                        help='Whether to store the predictions',
                        action='store_true')
    parser.add_argument('--not_clean', dest='to_clean',
                        help='Do not clean useless models if specified',
                        action='store_false')
    parser.add_argument('--output_blob', dest='output_blob',
                        help='Name of the output blob storing AU probabilities',
                        type=str, default='prob')
    parser.add_argument('--label_blob', dest='label_blob',
                        help='Name of the blob storing labels',
                        type=str, default='labels')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def parse_model_info(model_dir):
    f_info = open(os.path.join(model_dir, 'info.txt'), 'r')
    info_dict = dict()
    while True:
        line = f_info.readline()
        if line == '': break
        tokens = line.strip().split(':')
        if tokens[0] == 'AUs':
            info_dict['AUs'] = [int(au) for au in tokens[1].split(',')]
        elif tokens[0] == 'Best epoch':
            info_dict['Best epoch'] = int(tokens[1])
        elif tokens[0] == 'Best validation average':
            info_dict['Best validation ave.'] = float(tokens[1])
        else:
            info_dict[tokens[0]] = tokens[1]
    f_info.close()
    return info_dict


def clean_useless_models(model_dir, best_epoch):
    for filename in os.listdir(model_dir):
        name, ext = os.path.splitext(filename)
        filepath = os.path.join(model_dir, filename)
        if ext == '.solverstate':
            os.remove(filepath)
        elif ext == '.caffemodel':
            epoch = int(name[5:])
            if epoch != best_epoch:
                os.remove(filepath)
    return


def run_test(model, output_blob, label_blob, pred_names):
    next_epoch  = model.layers[0].epoch + 1
    num_samples = model.layers[0].num_samples
    labels = list()
    preds  = list()
    while model.layers[0].epoch < next_epoch:
        processed_num = model.layers[0].processed_num
        sys.stdout.flush()
        sys.stdout.write('prog: {}/{}      \r'.format(processed_num, num_samples))
        model.forward()
        labels.append(model.blobs[label_blob].data.copy())
        preds.append(model.blobs[output_blob].data.copy())
    labels = np.vstack(labels)
    preds  = (np.vstack(preds) > 0.5) * 1.0
    label_names = [int(au[2:]) for au in model.layers[0].au_names]
    results, intersect = calc_f1_with_name(labels, preds, label_names, pred_names)
    p_indices = [pred_names.index(i) for i in intersect]
    l_indices = [label_names.index(i) for i in intersect]
    return results, labels[:, l_indices], preds[:, p_indices], intersect


def main():
    args = parse_args()
    assert(args.mode in ['gpu', 'cpu']), 'Invalid value for mode'
    if args.mode == 'gpu':
        caffe.set_mode_gpu()
        if args.gpu_id != -1:
            caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    test_net = os.path.join(root_path, args.test_net)
    test_net_prefix = os.path.splitext(os.path.basename(test_net))[0].replace('_Test', '')
    model_dir  = os.path.join(root_path, args.exp_path, 'models', args.model_prefix)
    result_dir = os.path.join(root_path, args.exp_path, 'results')
    mkdir_if_not_exists(result_dir)
    if test_net_prefix == args.model_prefix:
        output_name = args.model_prefix
    else:
        output_name = '{}_{}'.format(args.model_prefix, test_net_prefix)

    info_dict = parse_model_info(model_dir)
    if args.to_clean:
        clean_useless_models(model_dir, info_dict['Best epoch'])
    model_name = 'epoch{}.caffemodel'.format(info_dict['Best epoch'])
    model = caffe.Net(test_net, os.path.join(model_dir, model_name), caffe.TEST)
    r, l, p, n = run_test(model, args.output_blob, args.label_blob, info_dict['AUs'])
    test_results = dict()
    test_results['row_name'] = n
    test_results['f1'] = r
    with open(os.path.join(result_dir, output_name+'.pkl'), 'w') as f_r:
        cPickle.dump(test_results, f_r)

    if args.store_preds:
        predict_dir = os.path.join(root_path, args.exp_path, 'predicts')
        mkdir_if_not_exists(predict_dir)
        f_p = h5py.File(os.path.join(predict_dir, output_name+'.h5'), 'w')
        f_p.attrs['AUs'] = n
        set_data = f_p.create_group('Test')
        set_data.create_dataset('preds', p.shape, dtype=np.float32, compression='gzip')
        set_data['preds'][...] = p
        set_data.create_dataset('labels', l.shape, dtype=np.float32, compression='gzip')
        set_data['labels'][...] = l
        f_p.close()
    return


if __name__ == "__main__":
    main()
