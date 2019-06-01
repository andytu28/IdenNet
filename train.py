from __future__ import print_function
import sys
import os
import argparse
import logging
import numpy as np
from utils import root_path, caffe_path
from utils.metrics import calc_f1
sys.path.insert(0, caffe_path)
import caffe


def parse_args():
    parser = argparse.ArgumentParser(description='In train.py')
    parser.add_argument('--solver', dest='solver',
                        help='Target solver to run training process',
                        required=True, type=str)
    parser.add_argument('--mode', dest='mode',
                        help='Use gpu or cpu for training',
                        type=str, default='gpu')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='Target gpu_id when multiple gpus are available',
                        type=int, default=-1)
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='Maximum number of epochs for training (Default=100)',
                        type=int, default=100)
    parser.add_argument('--max_tol', dest='max_tol',
                        help='Maximum number of tol for early stopping (Default=10)',
                        type=int, default=10)
    parser.add_argument('--weights', dest='weights',
                        help='Pretrained model weights',
                        type=str, default='')
    parser.add_argument('--output_blob', dest='output_blob',
                        help='Name of the output blob storing AU probabilities',
                        type=str, default='prob')
    parser.add_argument('--label_blob', dest='label_blob',
                        help='Name of the blob storing labels',
                        type=str, default='labels')
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def run_validation(solver, output_blob, label_blob):
    print('\n===== Run validation =====')
    next_epoch  = solver.test_nets[0].layers[0].epoch + 1
    num_samples = solver.test_nets[0].layers[0].num_samples
    labels = list()
    preds  = list()
    while solver.test_nets[0].layers[0].epoch < next_epoch:
        processed_num = solver.test_nets[0].layers[0].processed_num
        sys.stdout.flush()
        sys.stdout.write('prog: {}/{}       \r'.format(processed_num, num_samples))
        solver.test_nets[0].forward()

        labels.append(solver.test_nets[0].blobs[label_blob].data.copy())
        preds.append(solver.test_nets[0].blobs[output_blob].data.copy())
    labels = np.vstack(labels)
    preds  = (np.vstack(preds) > 0.5) * 1.0
    results = calc_f1(labels, preds)
    average = np.mean(results)
    return average, results, solver.test_nets[0].layers[0].au_names


def run_train(solver, max_epoch, max_tol, output_blob, label_blob):
    timer = caffe.Timer()
    solver.add_callback(lambda: timer.start(), lambda: timer.stop())
    loss_weights = [(n, w) for n, w in solver.net.blob_loss_weights.iteritems() if w > 0.0]
    snapshot_prefix = solver.param.snapshot_prefix
    model_dir = os.path.dirname(snapshot_prefix)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler   = logging.FileHandler(os.path.join(model_dir, 'train.log'), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    f_info = open(os.path.join(model_dir, 'info.txt'), 'w')
    f_info.write('AUs:'+','.join([au[2:] for au in solver.net.layers[0].au_names])+'\n')

    epoch   = 0
    it      = 0
    last_it = 0
    tol     = 0
    train_loss = 0.0
    best_epoch = -1
    best = -np.inf
    while True:
        solver.step(1)
        loss = np.sum([solver.net.blobs[n].data[...].copy() * w for n, w in loss_weights])
        train_loss = (train_loss * (it-last_it) + loss) / (it+1-last_it)
        message = '===Epoch<{}>...prog: {}/{}, train_loss: {}, speed: {:.3f}s/iter      \r'
        sys.stdout.flush()
        sys.stdout.write(message.format(solver.net.layers[0].epoch,
                                        solver.net.layers[0].processed_num,
                                        solver.net.layers[0].num_samples,
                                        train_loss, timer.ms/1000.0))

        if epoch < solver.net.layers[0].epoch:
            average, results, au_names = run_validation(solver, output_blob, label_blob)
            solver.snapshot()
            os.rename(snapshot_prefix+'_iter_{}.solverstate'.format(it+1),
                      os.path.join(model_dir, 'epoch{}.solverstate'.format(epoch)))
            os.rename(snapshot_prefix+'_iter_{}.caffemodel'.format(it+1),
                      os.path.join(model_dir, 'epoch{}.caffemodel'.format(epoch)))

            if average > best:
                print('GOOD!\n')
                best = average
                best_epoch = epoch
                tol = 0
            else:
                tol += 1

            val_string = '{:<5} {:<.3f}\n' * len(au_names)
            val_list   = [e for t in zip(au_names, results) for e in t]
            val_string = val_string.format(*val_list)
            print('Validation average: {}, tol: {}/{}'.format(average, tol, max_tol))
            print(val_string)

            logger.info('Epoch<{}> ends'.format(epoch))
            message = 'train_loss: {}, validation_average: {}, tol: {}/{}'
            logger.info(message.format(train_loss, average, tol, max_tol))
            logger.info('Validation results: \n'+val_string)

            if tol >= max_tol or epoch == (max_epoch - 1):
                print('\n\n\n===== Training finish =====')
                message = 'End at epoch {}, best: {}, best_epoch: {}\n'
                print(message.format(epoch, best, best_epoch))
                logger.info('Training process ends')
                logger.info(message.format(epoch, best, best_epoch))
                break

            epoch += 1
            train_loss = 0.0
            last_it = it + 1
            handler.flush()
        it += 1
    f_info.write('Best epoch:{}\n'.format(best_epoch))
    f_info.write('Best validation average:{}\n'.format(best))
    f_info.close()
    return


def main():
    args = parse_args()
    assert(args.mode in ['gpu', 'cpu']), 'Invalid value for mode'
    if args.mode == 'gpu':
        caffe.set_mode_gpu()
        if args.gpu_id != -1:
            caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    solver = caffe.get_solver(os.path.join(root_path, args.solver))

    # Loading pretrained weights if given
    if args.weights != '':
        solver.net.copy_from(os.path.join(root_path, args.weights))
        for test_net in solver.test_nets:
            test_net.share_with(solver.net)

    run_train(solver, args.max_epoch, args.max_tol, args.output_blob, args.label_blob)
    return


if __name__ == "__main__":
    main()
