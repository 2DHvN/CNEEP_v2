import numpy as np
from scipy import stats


def logging(i, loss, time_step, preds, train=True):
    tmp = {}
    cum_preds = np.cumsum(preds, axis=1).mean(axis=0)
    pred_rate, _, _, _, _ = stats.linregress(np.arange(len(cum_preds)), cum_preds)
    pred_rate = (1 / time_step) * pred_rate
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = pred_rate
    if train:
        print("Train  iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, pred_rate))
    else:
        print("Test   iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, pred_rate))

    return tmp


def logging_r(i, loss, time_step, ents, preds, train=True):
    tmp = {}
    cum_preds = np.cumsum(preds, axis=1).mean(axis=0)
    pred_rate, _, _, _, _ = stats.linregress(np.arange(len(cum_preds)), cum_preds)
    pred_rate = (1 / time_step) * pred_rate
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = pred_rate
    _, _, r_value, _, _ = stats.linregress(preds.flatten(), ents.flatten())
    tmp["r_square"] = r_value ** 2
    if train:
        print("Train  iter: %d  loss: %1.4e  pred: %.5f  R-square: %.5f" % (i, loss, pred_rate, r_value ** 2))
    else:
        print("Test   iter: %d  loss: %1.4e  pred: %.5f  R-square: %.5f" % (i, loss, pred_rate, r_value ** 2))

    return tmp


def logging_seq(i, loss, seq_len, preds, train=True):
    tmp = {}
    pred = preds.flatten()
    cum_pred = np.cumsum(pred)
    slope, _, _, _, _ = stats.linregress(np.arange(len(cum_pred)) * (seq_len - 1), cum_pred)
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = slope
    if train:
        print("Train  iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, slope))
    else:
        print("Test   iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, slope))

    return tmp