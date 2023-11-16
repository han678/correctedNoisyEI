import sys
import torch
import os
import time
from operator import attrgetter
from chainer.functions import dropout
from chainer.functions import max_pooling_2d
from chainer.functions import relu
from chainer.functions import softmax
from chainer.links import Convolution2D
from chainer.links import Linear
from chainercv.links import VGG16
from chainercv.links.connection.conv_2d_activ import Conv2DActiv
from chainercv.links.model.pickable_sequential_chain import PickableSequentialChain
from compression.comp_method import vh_decompose
from compression.imagenet.utils import f_norm as get_f_norm, risk as get_risk


def vh_block(Ci, Co, K):
    v = Convolution2D(Ci, K, (3, 1), 1, (1, 0), nobias=True)
    h = Conv2DActiv(K, Co, (1, 3), 1, (0, 1))
    return v, h


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)


class VGG16LowRank(PickableSequentialChain):
    def __init__(self, rank_list):
        super(VGG16LowRank, self).__init__()
        self.compress_layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                                     'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        with self.init_scope():
            self.conv1_1_v, self.conv1_1_h = vh_block(3, 64, rank_list[0])
            self.conv1_2_v, self.conv1_2_h = vh_block(64, 64, rank_list[1])
            self.pool1 = _max_pooling_2d
            self.conv2_1_v, self.conv2_1_h = vh_block(64, 128, rank_list[2])
            self.conv2_2_v, self.conv2_2_h = vh_block(128, 128, rank_list[3])
            self.pool2 = _max_pooling_2d
            self.conv3_1_v, self.conv3_1_h = vh_block(128, 256, rank_list[4])
            self.conv3_2_v, self.conv3_2_h = vh_block(256, 256, rank_list[5])
            self.conv3_3_v, self.conv3_3_h = vh_block(256, 256, rank_list[6])
            self.pool3 = _max_pooling_2d
            self.conv4_1_v, self.conv4_1_h = vh_block(256, 512, rank_list[7])
            self.conv4_2_v, self.conv4_2_h = vh_block(512, 512, rank_list[8])
            self.conv4_3_v, self.conv4_3_h = vh_block(512, 512, rank_list[9])
            self.pool4 = _max_pooling_2d
            self.conv5_1_v, self.conv5_1_h = vh_block(512, 512, rank_list[10])
            self.conv5_2_v, self.conv5_2_h = vh_block(512, 512, rank_list[11])
            self.conv5_3_v, self.conv5_3_h = vh_block(512, 512, rank_list[12])
            self.pool5 = _max_pooling_2d
            self.fc6 = Linear(None, 4096)
            self.fc6_relu = relu
            self.fc6_dropout = dropout
            self.fc7 = Linear(None, 4096)
            self.fc7_relu = relu
            self.fc7_dropout = dropout
            self.fc8 = Linear(None, 1000)
            self.prob = softmax


class VGGModel:
    def __init__(self, pick):
        self.model = VGG16(pretrained_model="imagenet")
        self.pick = pick

    def restore_model(self):
        self.model = VGG16(pretrained_model="imagenet")

    def delete_model(self):
        # to save memory
        del self.model

    def compress(self, rank_list):
        # only compress conv layers
        st = time.monotonic()
        m1_num, m2_num = 0, 0
        ratios = {}
        W_shapes = {}
        self.model.to_cpu()
        self.compressed_model = VGG16LowRank(rank_list)
        # copy fc layers
        self.compressed_model.fc6.copyparams(self.model.fc6)
        self.compressed_model.fc7.copyparams(self.model.fc7)
        self.compressed_model.fc8.copyparams(self.model.fc8)
        for i in range(len(rank_list)):
            rank = rank_list[i]
            layer = self.compressed_model.compress_layer_names[i]
            source_layer = attrgetter(layer + ".conv")(self.model)
            v, h, _m1_num, _m2_num, _ratio = vh_decompose(layer, source_layer.W.data, rank)
            m1_num += _m1_num
            m2_num += _m2_num
            ratios[layer] = _ratio
            W_shapes[layer] = source_layer.W.shape
            # copy v,h to compressed_model, copy bias
            attrgetter(layer + "_v.W")(self.compressed_model).data = v.copy()
            attrgetter(layer + "_h.conv.W")(self.compressed_model).data = h.copy()
            attrgetter(layer + "_h.conv.b")(self.compressed_model).data = source_layer.b.data.copy()
        info = dict()
        info["m1_num"] = m1_num
        info["m2_num"] = m2_num
        info["ratios"] = ratios
        info["W_shapes"] = W_shapes
        info["overall_ratio"] = m2_num / m1_num
        info["time"] = time.monotonic() - st

        return info

    def f_norm(self, n, gpu_id=0):
        return get_f_norm(self.model, self.compressed_model, self.pick, n, gpu_id)

    def risk(self, n, gpu_id=0):
        return get_risk(self.compressed_model, n, gpu_id)

if __name__ == '__main__':
    model = VGGModel('fc8')
    CHAINERX_CUDA_CUPY_SHARE_ALLOCATOR = 1
    with torch.no_grad():
        theta = torch.tensor([6, 10, 10, 10, 20, 30, 10, 10, 10, 10, 10, 10, 10])
        model.compress(theta)
        mu, sigma = model.f_norm(n=50, gpu_id=-1)
        print(mu, sigma)
        model.delete_model()
        top_1_acc, top_1_acc_std = model.risk(n=50, gpu_id=-1)
        print(top_1_acc, top_1_acc_std)

