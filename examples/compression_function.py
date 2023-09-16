from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from botorch.exceptions.errors import InputDataError
from torch import Tensor, device
from torch.nn import Module

from compression.imagenet.resnet import Resnet50, run_validate
from compression.imagenet.vgg import VGGModel
from compression.mnist.mlp_model import FC3


class BaseCompressProblem(Module, ABC):
    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, negate: bool = False, tkwargs: dict = {"dtype": torch.double, "device": "cpu"}, n_sample: int = 50,) -> None:
        super().__init__()
        self.negate = negate
        self.tkwargs = tkwargs
        self.n_sample = n_sample
        if len(self._bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self._bounds)=}."
            )
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2).to(**self.tkwargs)
        )

    @abstractmethod
    def evaluate_observed(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a point."""
        pass  # pragma: no cover


class CompressVGG16(BaseCompressProblem):
    dim = 13
    gamma = 1.0
    _bounds = [(0.2, 0.9), (0.2, 0.9), (0.2, 0.7), (0.2, 0.7), (0.2, 0.7), (0.2, 0.7), (0.2, 0.7), (0.2, 0.5),
               (0.2, 0.5), (0.2, 0.5), (0.2, 0.5), (0.2, 0.5), (0.2, 0.5)]
    _bounds_int_max = [8, 96, 128, 192, 256, 384, 384, 512, 768, 768, 768, 768, 768]
    model = VGGModel('fc8')

    def __init__(
            self,
            negate: bool = False,
            comp_obj: int = 0,
            tkwargs: dict = {"dtype": torch.double, "device": "cpu"},
            n_sample: int = 50,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            negate: If True, negate the function.
        """
        super().__init__(negate=negate, tkwargs=tkwargs, n_sample=n_sample)
        self.model = VGGModel('fc8')
        self.comp_obj = comp_obj
        self.gpu_id = -1 if self.tkwargs["device"] == device(type='cpu') else 0
        if self.comp_obj == 0:
            self.name = 'compress_vgg16_norm'
        elif self.comp_obj == 1:
            self.name = 'compress_vgg16_risk'

    def transform_theta(self, theta):
        theta_rank = [int(theta[i] * self._bounds_int_max[i]) for i in list(range(len(theta)))]
        return theta_rank

    def evaluate_observed(self, X: Tensor) -> Tensor:
        if len(X.shape) == 2:
            X = X.squeeze(0)  # evaluate one point every time
        if torch.is_tensor(X):
            X = X.tolist()
        theta = self.transform_theta(X)
        info = self.model.compress(theta)
        ratio = info["overall_ratio"]
        if self.comp_obj == 0:
            mu, sigma = self.model.f_norm(n=self.n_sample, gpu_id=self.gpu_id)
            print("==> ratio: {}, norm: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                        self.gamma * sigma))
        else:
            top_1_acc, sigma = self.model.risk(n=self.n_sample, gpu_id=self.gpu_id)
            mu = 1 - top_1_acc
            print("==> ratio: {}, top_1_error: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                                    self.gamma * sigma))
        #print("Current theta is: {} ".format(theta))
        y = torch.tensor([ratio + self.gamma * mu], **self.tkwargs).unsqueeze(0)
        y_err = torch.tensor([self.gamma * sigma], **self.tkwargs).unsqueeze(0)
        if self.negate == True:
            return -y, y_err
        else:
            return y, y_err


class CompressFC3(BaseCompressProblem):
    dim = 3
    gamma = 1.0
    _bounds = [(0.01, 0.9), (0.01, 0.9), (0.11, 0.9)]
    _bounds_int_max = [500, 500, 10]
    model_path = './compression/mnist/mlp_model/model_iter_11400'
    model = FC3(model_path)

    def __init__(
            self,
            negate: bool = False,
            comp_obj: int = 0,
            tkwargs: dict = {"dtype": torch.double, "device": "cpu"},
            n_sample: int = 50,
    ) -> None:
        super().__init__(negate=negate, tkwargs=tkwargs, n_sample=n_sample)
        self.comp_obj = comp_obj
        self.gpu_id = -1 if self.tkwargs["device"] == device(type='cpu') else 0
        if self.comp_obj == 0:
            self.name = 'compress_FC3_norm'
        elif self.comp_obj == 1:
            self.name = 'compress_FC3_risk'

    def transform_theta(self, theta):
        theta_rank = [int(theta[i] * self._bounds_int_max[i]) for i in list(range(len(theta)))]
        return theta_rank

    def evaluate_observed(self, X: Tensor) -> Tensor:
        if len(X.shape) == 2:
            X = X.squeeze(0)  # evaluate one point every time
        if torch.is_tensor(X):
            X = X.tolist()
        rank = self.transform_theta(X)
        ratio = self.model.compress_svd(rank_list=rank)
        if self.comp_obj == 0:
            mu, sigma = self.model.f_norm(n=self.n_sample, gpu_id=self.gpu_id)
            print("==> ratio: {}, norm: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                        self.gamma * sigma))
        else:
            top_1_acc, sigma = self.model.risk(n=self.n_sample, gpu_id=self.gpu_id)
            mu = 1 - top_1_acc
            print("==> ratio: {}, top_1_error: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                                    self.gamma * sigma))
        if ratio > 1.:
            import ipdb
            ipdb.set_trace()
            print("wrong")
        y = torch.tensor([ratio + self.gamma * mu], **self.tkwargs).unsqueeze(0)
        y_err = torch.tensor([self.gamma * sigma], **self.tkwargs).unsqueeze(0)
        if self.negate == True:
            return -y, y_err
        else:
            return y, y_err


class CompressResnet50(BaseCompressProblem):
    dim = 16
    gamma = 1.0
    _bounds = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95),
               (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95),
               (0.05, 0.95), (0.05, 0.95)]
    model = Resnet50('resnet50', pick="fc6")

    def __init__(
            self,
            negate: bool = False,
            comp_obj: int = 0,
            tkwargs: dict = {"dtype": torch.double, "device": "cpu"},
            n_sample: int = 50,
    ) -> None:

        super().__init__(negate=negate, tkwargs=tkwargs, n_sample =n_sample)
        self.comp_obj = comp_obj
        self.gpu_id = -1 if self.tkwargs["device"] == device(type='cpu') else 0
        if self.comp_obj == 0:
            self.name = 'compress_resnet50_norm'
        elif self.comp_obj == 1:
            self.name = 'compress_resnet50_risk'

    def evaluate_observed(self, X: Tensor) -> Tensor:
        if len(X.shape) == 2:
            X = X.squeeze(0)  # evaluate one point every time
        if torch.is_tensor(X):
            X = X.tolist()
        self.model.compress(X)
        ratio = self.model.ratio
        if self.comp_obj == 0:
            mu, sigma = self.model.f_norm(n=self.n_sample, gpu_id=self.gpu_id)
            print("==> ratio: {}, norm: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                        self.gamma * sigma))
        else:
            top_1_acc, sigma = self.model.risk(n=self.n_sample, gpu_id=self.gpu_id)
            mu = 1 - top_1_acc
            print("==> ratio: {}, top_1_error: {}, y: {}, ystd: {}".format(ratio, mu, ratio + self.gamma * mu,
                                                                    self.gamma * sigma))
        if ratio > 1.:
            import ipdb
            ipdb.set_trace()
            print("wrong")
        y = torch.tensor([ratio + self.gamma * mu], **self.tkwargs).unsqueeze(0)
        y_err = torch.tensor([self.gamma * sigma], **self.tkwargs).unsqueeze(0)
        if self.negate == True:
            return -y, y_err
        else:
            return y, y_err

    def theta_statistic(self, theta):
        self.model.compress(theta)
        print("Original Model:")
        run_validate(self.model.model, "/users/visics/hzhou/data/ILSVRC2012", 1)
        print("Compressed Model:")
        run_validate(self.model.compressed_model, "/users/visics/hzhou/data/ILSVRC2012", 1)
