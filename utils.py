import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score
import scipy.stats

def KL_divergence(p, q, epsilon=1e-12):
    return scipy.stats.entropy(p+ epsilon, q+ epsilon)

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins

def shuffle_index(x_len=8):
    idx = torch.randperm(x_len)
    s = torch.abs(idx[1:] - idx[:-1])
    while any(s == 1):
        idx = torch.randperm(x_len)
        s = torch.abs(idx[1:] - idx[:-1])
    return idx

def genMotionAnoSmps(x):
    """
    生成运动异常样本
    :param x: 输入张量，形状为 (b, c, t, h, w)
    :return: 打乱时间维度后的张量，形状为 (b, c, t, h, w)
    """
    x_shuffle = x[:, :, shuffle_index(x.size()[2]), :, :]
    return x_shuffle

def psnr(mse):
    return 10 * math.log10(1 / mse)

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc










