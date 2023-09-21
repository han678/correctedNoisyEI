from typing import Optional

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from scipy.optimize import minimize
from torch import Tensor
from torch.distributions import MultivariateNormal


def findExtrema(function):
    step_size = 0.01
    bounds = function.bounds.T.tolist()
    xs = np.array([np.arange(x[0], x[1], step_size) for x in bounds]).T.tolist()
    y = [function.evaluate_true(torch.tensor(x)) for x in xs]
    x0 = xs[y.index(min(y))]

    def obj_func(x):
        return function.evaluate_true(torch.tensor(x)).item()

    # find min
    min_res = minimize(obj_func, x0, method="L-BFGS-B", bounds=bounds)
    func_min = min_res.fun
    func_min_x = min_res.x
    x1 = xs[y.index(max(y))]

    def neg_obj_func(x):
        return -function.evaluate_true(torch.tensor(x)).item()

    # find max
    max_res = minimize(neg_obj_func, x1, method="L-BFGS-B", bounds=bounds)
    func_max = -max_res.fun
    func_max_x = max_res.x

    if function.negate == True:
        print(f"min x : {func_max_x}, max x : {func_min_x},")
        return (-func_max, -func_min)
    else:
        print(f"min x : {func_min_x}, max x : {func_max_x},")
        return (func_min, func_max)


class Noisy_synthetic_function():

    def __init__(self, SyntheticTestFunction, tkwargs):
        self.noise_std = SyntheticTestFunction.noise_std
        self.function = SyntheticTestFunction
        self.dim = self.function.dim
        self.name = "{}_{}d".format(self.function._get_name(), str(self.dim))
        self.tkwargs = tkwargs
        self.bounds = self.function.bounds.to(**self.tkwargs)

    def evaluate_observed(self, train_x):
        if not torch.is_tensor(train_x):
            train_x = torch.tensor(train_x, **self.tkwargs)
        exact_obj = self.function.evaluate_true(train_x).unsqueeze(-1)
        noise = torch.randn_like(exact_obj, **self.tkwargs) * self.noise_std
        observed_obj = exact_obj + noise
        if self.function.negate == True:
            return -observed_obj, torch.abs(noise)
        else:
            return observed_obj, torch.abs(noise)

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self.function._optimal_value if self.function.negate else self.function._optimal_value

    @property
    def optimizers(self) -> float:
        return self.function._optimizers

class TestGaussian(SyntheticTestFunction):
    dim = 1
    _optimal_value = 0.03753703832626343
    _bounds = [(0.0, 10.0)]
    _optimizers = [(9.99)]

    def __init__(
            self,
            noise_std: Optional[float] = None,
            negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        d = 1
        distribution1 = MultivariateNormal(loc=5 * torch.ones(d), covariance_matrix=0.8 * torch.eye(d))
        distribution2 = MultivariateNormal(loc=1 * torch.ones(d), covariance_matrix=0.8 * torch.eye(d))
        distribution3 = MultivariateNormal(loc=8 * torch.ones(d), covariance_matrix=0.8 * torch.eye(d))
        prob = torch.exp(distribution1.log_prob(X)) + torch.exp(distribution2.log_prob(X)) + torch.exp(
            distribution3.log_prob(X))
        return prob
