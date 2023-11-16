import os
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer.backends import cuda
from operator import attrgetter

from chainer.dataset import download
from chainer.datasets._mnist_helper import preprocess_mnist, make_npz
from compression.comp_method import svd_K
import numpy


def get_mnist(withlabel=True, ndim=1, scale=1., dtype=None, label_dtype=numpy.int32, rgb_format=False):
    dtype = chainer.get_dtype(dtype)
    train_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz']
    test_urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    train_raw = retrieve_mnist('train.npz', train_urls)
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    test_raw = retrieve_mnist('test.npz', test_urls)
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                            label_dtype, rgb_format)
    return train, test


def retrieve_mnist(name, urls):
    # download.get_dataset_directory('pfnet/chainer/mnist')
    root = "./data/minst"
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    return download.cache_or_load_file(path, lambda path: make_npz(path, urls), numpy.load)


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(n_units)
            self.fc2 = L.Linear(n_units)
            self.fc3 = L.Linear(n_out)

    def __call__(self, x):
        self.info = {}
        self.info["fc1"] = x
        h1 = F.relu(self.fc1(x))
        self.info["fc2"] = h1
        h2 = F.relu(self.fc2(h1))
        self.info["fc3"] = h2
        return self.fc3(h2)


class FC3():
    def __init__(self, model_path):
        self.model = MLP(1000, 10)
        chainer.serializers.load_npz(model_path, self.model)
        self.compressed_model = self.model.copy("copy")
        self.compress_layer_names = ['fc1', 'fc2', 'fc3']

    def f_norm(self, n, gpu_id):
        batch_size = 32
        _, test = get_mnist(withlabel=False)  # chainer.datasets.get_mnist(withlabel=False)
        perm = np.arange(len(test))
        np.random.shuffle(perm)
        perm = perm[:n]
        data = [test[i] for i in perm]
        data_iter = chainer.iterators.SerialIterator(data, batch_size=batch_size, repeat=False)
        if gpu_id >= 0:
            cuda.get_device_from_id(gpu_id).use()
            self.model.to_gpu()
            self.compressed_model.to_gpu()

        norms = []
        for batch in data_iter:
            X = chainer.dataset.concat_examples(batch, device=gpu_id)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                diff = self.model(X) - self.compressed_model(X)  # (NB, 10)
                _norm = F.sum(F.square(diff), axis=1)  # (NB, )
                norms.append(cuda.to_cpu(_norm.array))  # (NB,)
        norms = np.hstack(norms)
        assert norms.shape[0] == n

        return float(np.mean(norms)), float(np.std(norms, ddof=1) / np.sqrt(n))

    def risk(self, n, gpu_id):
        # compute accuracy
        batch_size = 32
        _, test = get_mnist()  # chainer.datasets.get_mnist()
        perm = np.arange(len(test))
        np.random.shuffle(perm)
        perm = perm[:n]
        data = [test[i] for i in perm]
        data_iter = chainer.iterators.SerialIterator(data, batch_size=batch_size, repeat=False)
        if gpu_id >= 0:
            cuda.get_device_from_id(gpu_id).use()
            self.compressed_model.to_gpu()

        count = 0
        loss_list = []
        accuracy_list = []
        for batch in data_iter:
            X, t = chainer.dataset.concat_examples(batch, device=gpu_id)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                y = self.compressed_model(X)

            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            count += X.shape[0]
            loss_list.append(loss)
            accuracy_list.append((accuracy.data * float(X.shape[0])).item())
        acc_mean = np.array(accuracy_list).sum() / count
        acc_std = np.sqrt(acc_mean * (1 - acc_mean) / count)
        return acc_mean, acc_std

    def compress_svd(self, rank_list):
        m1_num, m2_num = 0, 0
        ratios = {}
        W_shapes = {}
        self.model.to_cpu()
        self.compressed_model.to_cpu()

        for i in range(len(rank_list)):
            rank = rank_list[i]
            layer_name = self.compress_layer_names[i]
            source_layer = attrgetter(layer_name)(self.model)
            W, _m1_num, _m2_num = svd_K(source_layer.W.data, rank)
            m1_num += _m1_num
            m2_num += _m2_num
            ratios[layer_name] = _m2_num / _m1_num
            W_shapes[layer_name] = source_layer.W.shape
            attrgetter(layer_name + '.W')(self.compressed_model).data = W.copy()
        ratio = m2_num / m1_num
        return ratio

    def evaluate(self, model, gpu_id=-1):
        batch_size = 32
        _, test = get_mnist()  # chainer.datasets.get_mnist()
        test_iter = chainer.iterators.SerialIterator(
            test, batch_size=batch_size, repeat=False)
        if gpu_id >= 0:
            cuda.get_device_from_id(gpu_id).use()
            model.to_gpu()

        count = 0
        sum_loss = 0
        sum_accuracy = 0
        for batch in test_iter:
            X, t = chainer.dataset.concat_examples(batch, device=gpu_id)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                y = model(X)
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            count += batch_size
            sum_loss += float(loss.data) * batch_size
            sum_accuracy += float(accuracy.data) * batch_size
            sys.stdout.write('{} / {}\r'.format(count, len(test)))
            sys.stdout.flush()

        print('mean loss:     {}'.format(sum_loss / count))
        print('mean accuracy: {}'.format(sum_accuracy / count))

        return sum_loss / count, sum_accuracy / count
