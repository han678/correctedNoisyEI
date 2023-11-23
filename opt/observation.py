import numpy as np
import torch
from botorch.utils import draw_sobol_samples
from scipy.stats import beta
from botorch.utils.transforms import normalize, unnormalize


def get_x_bounds(X):
    X_bounds = torch.stack(
        [X.min() * torch.ones(X.shape[1], dtype=X.dtype, device=X.device),
         X.max() * torch.ones(X.shape[1], dtype=X.dtype, device=X.device)], )
    return X_bounds

def standardize_y(y, ystd):
    r"""
    Standardizes y (zero mean, unit variance) a tensor by dim=-2.
    """
    stddim = -1 if y.dim() < 2 else -2
    y_std = y.std(dim=stddim, keepdim=True)
    y_std = y_std.where(y_std <= 1e-9, torch.full_like(y_std, 1e-9))
    y_scaled = (y - y.mean(dim=stddim, keepdim=True)) / y_std
    ystd_scaled = ystd / y_std
    return y_scaled, ystd_scaled


def unstandardize_y(y_scaled, ystd_scaled, y):
    stddim = -1 if y.dim() < 2 else -2
    y_std = y.std(dim=stddim, keepdim=True)
    y_std = y_std.where(y_std <= 1e-9, torch.full_like(y_std, 1e-9))
    y_origin = (y_scaled * y_std) + y.mean(dim=0)
    ystd_origin = ystd_scaled * y_std
    return y_origin, ystd_origin


class Observation(object):
    def __init__(self, bounds, tkwargs):

        self._X = []
        self._y = []
        self._y_err = []
        self.bounds = bounds
        self.tkwargs = tkwargs
        self.bounds_list = bounds.T.tolist()
        self.pending = None

    @property
    def X(self):
        return torch.tensor(self._X, **self.tkwargs).detach()

    @property
    def y(self):
        return torch.tensor(self._y, **self.tkwargs).detach()

    @property
    def y_err(self):
        return torch.tensor(self._y_err, **self.tkwargs).detach()

    def normalize_x(self, x):
        X_bounds = get_x_bounds(self.X)
        x_norm = normalize(x, X_bounds)
        return x_norm

    def unnormalize_x(self, x_norm):
        X_bounds = get_x_bounds(self.X)
        x_origin = unnormalize(x_norm, X_bounds)
        return x_origin

    @property
    def y_standardize(self):
        y_scaled, ystd_scaled = standardize_y(self.y, self.y_err)
        return y_scaled, ystd_scaled

    def best_observed(self, return_y=True):
        idx = torch.argmin(self.y)
        if return_y:
            return (self.X[idx], self.y[idx])
        else:
            return self.X[idx]

    def add(self, x, y, y_err=1e-5):
        self._X.append(x)
        self._y.append(y)
        self._y_err.append(y_err)

    def check_duplication(self, x):
        # check if x is already been sampled.
        if len(self._X) == 0:
            return False
        return any(item == x.tolist() for item in self._X)

    def sobol_samples(self, n):
        return draw_sobol_samples(bounds=self.bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()).squeeze(1)

    def random_uniform_sample(self, n):
        # Draw samples from a uniform distribution within the bounds of the space.
        return np.random.uniform(*zip(*self.bounds_list), size=(n, len(self.bounds_list)))

    def beta_sample(self, n, a=2, b=5):
        # it is just a Beta density and transform rvs to (a,b)
        # mode is (a-1) / (a+b-2) = 0.2
        prior = beta(a, b).rvs(size=(n, len(self.bounds_list)))  # (n,D)
        for i in range(len(self.bounds_list)):
            a, b = self.bounds_list[i]
            prior[:, i] = prior[:, i] * (b - a) + a
        return prior
