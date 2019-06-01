from __future__ import print_function
import numpy as np
from sklearn.metrics import f1_score


def calc_f1(labels, preds):
    """ Input labels and preds both have shape (#samples, #labels)
    """
    assert(labels.shape == preds.shape), 'Shapes of labels and preds are not equal'
    if len(labels.shape) == 1:
        labels = labels[:, np.newaxis]
        preds  = preds[:, np.newaxis]

    results = []
    for index in xrange(labels.shape[1]):
        f1 = f1_score(labels[:, index], preds[:, index])
        results.append(f1)

    if len(results) == 1:
        results = results[0]
    return results


def calc_f1_with_name(labels, preds, label_names, pred_names):
    intersect  = list(set(label_names) & set(pred_names))
    label_inds = np.array([label_names.index(name) for name in intersect], dtype=np.uint)
    pred_inds  = np.array([pred_names.index(name)  for name in intersect], dtype=np.uint)
    results = calc_f1(labels[..., label_inds], preds[..., pred_inds])
    return results, intersect
