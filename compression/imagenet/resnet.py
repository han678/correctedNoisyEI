import os
from operator import attrgetter

import chainer
import chainer.functions as F
import numpy as np
import torch
from chainer import iterators
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.links import FeaturePredictor
from chainercv.links import ResNet50
from chainercv.utils import ProgressHook
from chainercv.utils import apply_to_iterator

from compression.comp_method import reconstruct_svd
from compression.imagenet.utils import f_norm as get_f_norm, topk, risk as get_risk


class Resnet50():

    def __init__(self, model_name, pick):
        self.model_name = model_name
        self.restore_model()
        self.pick = pick
        self.compress_layer_names = ['res2.a.conv2.conv.W', 'res2.b1.conv2.conv.W', 'res2.b2.conv2.conv.W',
                                     'res3.a.conv2.conv.W', 'res3.b1.conv2.conv.W', 'res3.b2.conv2.conv.W',
                                     'res3.b3.conv2.conv.W', 'res4.a.conv2.conv.W', 'res4.b1.conv2.conv.W',
                                     'res4.b2.conv2.conv.W', 'res4.b3.conv2.conv.W', 'res4.b4.conv2.conv.W',
                                     'res4.b5.conv2.conv.W', 'res5.a.conv2.conv.W', 'res5.b1.conv2.conv.W',
                                     'res5.b2.conv2.conv.W']

    def restore_model(self):
        self.model = ResNet50(arch='he', pretrained_model='imagenet')

    def delete_model(self):
        # to save memory
        del self.model

    def f_norm(self, n, gpu_id=-1):
        return get_f_norm(self.model, self.compressed_model, self.pick, n, gpu_id)

    def risk(self, n, gpu_id=0):
        return get_risk(self.compressed_model, n, gpu_id)

    def compress(self, ratio_list):
        m1_num, m2_num = 0, 0
        ratios = {}
        W_shapes = {}
        self.model.to_cpu()
        self.compressed_model = self.model.copy('copy')
        for i in range(len(ratio_list)):
            ratio = ratio_list[i]
            layer_name = self.compress_layer_names[i]
            param = attrgetter(layer_name)(self.compressed_model)
            W = param.data  # (C_o,C_i,kh,kw)
            Co = W.shape[0]
            recon, _m1_num, _m2_num = reconstruct_svd(W.reshape((Co, -1)), ratio)
            param.data = recon.reshape(W.shape)
            m1_num += _m1_num
            m2_num += _m2_num
            ratios[layer_name] = _m2_num / _m1_num
            W_shapes[layer_name] = W.shape

        self.ratio = m2_num / m1_num
        return m1_num, m2_num, ratios, W_shapes


def compress_and_report(m, ratio_list):
    m1_num, m2_num, ratios, _ = m.compress(ratio_list)
    print("Original: {}, compressed:{}, compression ratio:{}".format(
        m1_num, m2_num, m2_num / m1_num))
    for k, v in ratios.items():
        print("    {} => {}".format(k, v))
    for n in [10, 50, 100, 200]:
        mu, std = m.f_norm(n)
        print("n: {}, mu: {}, std: {}".format(n, mu, std))
    run_validate(m.compressed_model, "./data/ILSVRC2012", 1)


def run_validate(extractor, val_dir, gpu_id=-1, val_subset=None, options={}):
    batch_size = options.get("batch_size", 256)
    val_data = DirectoryParsingLabelDataset(val_dir)
    if val_subset is not None:
        val_data = val_data[:val_subset]
    val_iter = iterators.MultiprocessIterator(
        val_data, batch_size, repeat=False, shuffle=False, n_processes=4,
        n_prefetch=8, shared_mem=300000000)
    model = FeaturePredictor(
        extractor, crop_size=224, scale_size=256, crop='center')
    if gpu_id >= 0:
        chainer.cuda.get_device(gpu_id).use()
        model.to_gpu()

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, val_iter, hook=ProgressHook(len(val_data)))
    del in_values

    pred_probs, = out_values
    gt_labels, = rest_values

    pred_probs = np.array(list(pred_probs))
    gt_labels = np.array(list(gt_labels))
    accuracy = F.accuracy(pred_probs, gt_labels).data
    top5_acc = topk(pred_probs, gt_labels, 5)
    print()
    print('Top 1 Error {}'.format(1. - accuracy))
    print('Top 5 Error {}'.format(1. - top5_acc))
    return (1 - accuracy, 1 - top5_acc)


def test_imagenet_svd():
    m = Resnet50('resnet50', pick="fc6")
    ratio_list = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    compress_and_report(m, ratio_list)
    print("==> Original model:")
    run_validate(m.model, "./data/ILSVRC2012", 1)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'False'
    test_imagenet_svd()
