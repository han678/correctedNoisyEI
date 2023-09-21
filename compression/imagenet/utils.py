import chainer
import chainer.functions as F
import numpy as np
from chainer.backends import cuda
from chainer.datasets import TransformDataset
import os
from compression.imagenet.dataloader import LabeledImageDatasetSubset, transform_with_label, transform, \
    ImageDatasetSubset


def topk(y, t, k):
    acc = np.any(np.argpartition(y, -k, axis=1)[:, -k:].T == t, axis=0)
    acc_mean = float(acc.mean())
    return acc_mean, np.sqrt(acc_mean * (1 - acc_mean) / len(acc))


def risk(model, n, gpu_id=-1):
    batch_size = 32
    data = LabeledImageDatasetSubset("/users/visics/hzhou/data/random_50000_with_label", subset=n)
    data = TransformDataset(data, transform_with_label)
    data_iter = chainer.iterators.MultiprocessIterator(data, batch_size, repeat=False, shuffle=False)
    del data
    if gpu_id >= 0:
        cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    pred_probs = []
    all_labels = []
    for data_batch in data_iter:
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            inputs, labels = chainer.dataset.concat_examples(data_batch, device=gpu_id)
            outputs = model(inputs).array
            if gpu_id >= 0:
                outputs = chainer.cuda.to_cpu(outputs)
                labels = chainer.cuda.to_cpu(labels)
            pred_probs.append(outputs)
            all_labels += labels.tolist()
    pred_probs = np.vstack(pred_probs)
    all_labels = np.vstack(all_labels)
    top1_acc, top1_acc_std = topk(pred_probs, all_labels, 1)
    return top1_acc, top1_acc_std


def f_norm(model, compressed_model, pick, n, gpu_id=-1):
    batch_size = 32
    data = ImageDatasetSubset("/users/visics/hzhou/data/random_50000", subset=n)
    data = TransformDataset(data, transform)
    data_iter = chainer.iterators.MultiprocessIterator(data, batch_size, repeat=False, shuffle=False)
    del data
    model.pick = pick
    compressed_model.pick = pick
    if gpu_id >= 0:
        cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
        compressed_model.to_gpu()
    norms = []
    for data_batch in data_iter:
        image = chainer.dataset.concat_examples(data_batch, device=gpu_id)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            diff = model(image) - compressed_model(image)
            _norm = F.sqrt(F.batch_l2_norm_squared(diff))
            if gpu_id>=0:
                _norm = cuda.to_cpu(_norm.array)
            norms.append(_norm)  # (NB,)
    norms = np.hstack(norms)
    assert norms.shape[0] == n

    return float(np.mean(norms)), float(np.std(norms, ddof=1) / np.sqrt(n))
