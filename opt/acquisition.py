from __future__ import annotations

import abc
import math
import warnings
from copy import deepcopy
from typing import Optional, Union, Any

import torch
from botorch.acquisition import qNoisyExpectedImprovement, qProbabilityOfImprovement, MCAcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, ProbabilityOfImprovement, UpperConfidenceBound, \
    ExpectedImprovement, NoisyExpectedImprovement, LogProbabilityOfImprovement, LogExpectedImprovement
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform, MCAcquisitionObjective
from botorch.acquisition.utils import prune_inferior_points
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.sampling import SobolQMCNormalSampler, MCSampler
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.utils.transforms import t_batch_mode_transform, match_batch_shape, concatenate_pending_points
from torch import Tensor


def best_pred(model, train_X, return_pred=False, maximize=False):
    model.eval()
    with torch.no_grad():
        # get noiseless predictions
        posterior = model.posterior(X=train_X, observation_noise=False)
        # find the best point if we want to minimize the objective function
        if maximize:
            idx = torch.argmax(posterior.mean).item()
        else:
            idx = torch.argmin(posterior.mean).item()
    if not return_pred:
        return train_X[idx]
    else:
        return train_X[idx], posterior.mean[idx]


_sqrt_2pi = math.sqrt(2 * math.pi)
# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2 ** -0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2


class Acq(metaclass=abc.ABCMeta):

    def __init__(self, acq_kind, maximize):
        self.acq_kind = acq_kind
        self.maximize = maximize  # indicate if the goal is to maximize the problem

    def func(self, model, train_X):
        if self.acq_kind == 'UCB':
            return UpperConfidenceBound(model, beta=1.0, maximize=self.maximize)
        elif self.acq_kind == 'LCB':
            return LowerConfidenceBound(model, beta=1.0, maximize=self.maximize)
        elif self.acq_kind == 'NEI':
            return NoisyExpectedImprovement(model, X_observed=train_X, maximize=self.maximize)
        else:
            train_X_best, pred_f_best = best_pred(model, train_X, return_pred=True, maximize=self.maximize)
            if self.acq_kind == 'EI':
                return ExpectedImprovement(model, best_f=pred_f_best, maximize=self.maximize)
            elif self.acq_kind == 'log_EI':
                return LogExpectedImprovement(model, best_f=pred_f_best, maximize=self.maximize)
            elif self.acq_kind == 'PI':
                return ProbabilityOfImprovement(model, best_f=pred_f_best, maximize=self.maximize)
            elif self.acq_kind == 'log_PI':
                return LogProbabilityOfImprovement(model, best_f=pred_f_best, maximize=self.maximize)
            elif self.acq_kind == 'EI_C':
                return ModifiedExpectedImprovement(model, best_x=train_X_best, maximize=self.maximize)
            elif self.acq_kind == 'PI_C':
                return ModifiedProbabilityOfImprovement(model, best_x=train_X_best, maximize=self.maximize)
            elif self.acq_kind == 'q_NEI':
                qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
                return qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=qmc_sampler,
                                                 maximize=self.maximize)
            elif self.acq_kind == 'q_PI':
                qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
                return qProbabilityOfImprovement(model, best_f=pred_f_best, sampler=qmc_sampler, maximize=self.maximize)
            else:
                warnings.warn("The acquisition function {} has not been implemented yet.".format(self.acq_kind))


class ModifiedProbabilityOfImprovement(AnalyticAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_x: Union[float, Tensor],
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,  # consider a maximize problem
            **kwargs,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.best_x = best_x
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        u, correct_sigma = _correct_sigma(self.model, X, self.best_x, return_sigma=True, maximize=self.maximize)
        return Phi(u)


class ModifiedExpectedImprovement(AnalyticAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_x: Union[float, Tensor],
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            **kwargs,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.best_x = best_x
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        u, correct_sigma = _correct_sigma(self.model, X, self.best_x, return_sigma=True, maximize=self.maximize)
        return correct_sigma * _ei_helper(u)


def _correct_sigma(
        model: Model,
        X: Tensor,
        best_x: Union[float, Tensor],
        min_var: float = 1e-12,
        return_sigma: bool = True,
        maximize: bool = True) -> Tensor:
    new_X = torch.cat((X, best_x.unsqueeze(-2).unsqueeze(-2)), 0)  # (N+1,1,D)
    # make noiseless prediction
    posterior = model.posterior(new_X.squeeze(-2), observation_noise=False)
    mu = posterior.mean.squeeze(-1)
    mu_j, mu_i = mu[:-1], mu[-1:]  # (N,1,1),(1,1,1)
    variance = posterior.variance.clamp_min(min_var).view(mu.shape)
    var_j, var_i = variance[:-1], variance[-1:]  # (N,1,1),(1,1,1)
    cov_ji = posterior.distribution.covariance_matrix[:-1, -1:].view(var_j.shape)
    tilde_var = (var_j + var_i - 2 * cov_ji).clamp_min(min_var)
    tilde_sigma = torch.sqrt(tilde_var)
    u = torch.div(mu_j - mu_i, tilde_sigma)
    z = u if maximize else -u
    if return_sigma:
        return z, tilde_sigma
    else:
        return z


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)


class LowerConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Lower Confidence Bound (LCB).
    Analytic Lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.
    `LCB(x) = mu(x) - sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    Example:
    """

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            **kwargs,
    ) -> None:
        r"""Single-outcome Lower Confidence Bound.
        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        return -(mean if self.maximize else -mean) + self.beta.sqrt() * sigma


class qNoisyExpectedImprovement(
    MCAcquisitionFunction, CachedCholeskyMCAcquisitionFunction
):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)
    """

    def __init__(
            self,
            model: Model,
            X_baseline: Tensor,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            X_pending: Optional[Tensor] = None,
            prune_baseline: bool = True,
            cache_root: bool = True,
            maximize: bool = True,  # consider a maximize problem
            **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.

        TODO: similar to qNEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental qNEI from the new point. This would greatly increase
        efficiency for large batches.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self._setup(model=model, cache_root=cache_root)
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("X_baseline", X_baseline)

        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():
                posterior = self.model.posterior(
                    X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to `posterior.mvn` allows for reuse within
                #  this function, which may be helpful if the same root decomposition
                #  is produced by the calls to `self.base_sampler` and
                #  `self._cache_root_decomposition`.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                baseline_samples = self.get_posterior_samples(posterior)
            # We make a copy here because we will write an attribute `base_samples`
            # to `self.base_sampler.base_samples`, and we don't want to mutate
            # `self.sampler`.
            self.base_sampler = deepcopy(self.sampler)
            baseline_obj = self.objective(baseline_samples, X=X_baseline)
            self.register_buffer("baseline_samples", baseline_samples)
            self.maximize = maximize
            if self.maximize:
                self.register_buffer(
                    "baseline_obj_best_values", baseline_obj.max(dim=-1).values
                )
            else:
                self.register_buffer(
                    "baseline_obj_best_values", baseline_obj.min(dim=-1).values
                )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    def _forward_cached(self, posterior: Posterior, X_full: Tensor, q: int) -> Tensor:
        r"""Compute difference objective using cached root decomposition.

        Args:
            posterior: The posterior.
            X_full: A `batch_shape x n + q x d`-dim tensor of inputs
            q: The batch size.

        Returns:
            A `sample_shape x batch_shape`-dim tensor containing the
                difference in objective under each MC sample.
        """
        # handle one-to-many input transforms
        n_w = posterior._extended_shape()[-2] // X_full.shape[-2]
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        new_samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        new_obj = self.objective(new_samples, X=X_full[..., -q:, :])
        if self.maximize:
            new_obj_best_values = new_obj.max(dim=-1).values
        else:
            new_obj_best_values = new_obj.min(dim=-1).values
        n_sample_dims = len(self.base_sampler.sample_shape)
        view_shape = torch.Size(
            [
                *self.baseline_obj_best_values.shape[:n_sample_dims],
                *(1,) * (new_obj_best_values.ndim - self.baseline_obj_best_values.ndim),
                *self.baseline_obj_best_values.shape[n_sample_dims:],
            ]
        )
        if self.maximize:
            return new_obj_best_values - self.baseline_obj_best_values.view(view_shape)
        else:
            return self.baseline_obj_best_values.view(view_shape) - new_obj_best_values

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if self._cache_root:
            diffs = self._forward_cached(posterior=posterior, X_full=X_full, q=q)
        else:
            samples = self.get_posterior_samples(posterior)
            obj = self.objective(samples, X=X_full)
            if self.maximize:
                diffs = obj[..., -q:].max(dim=-1).values - obj[..., :-q].max(dim=-1).values
            else:
                diffs = obj[..., :-q].min(dim=-1).values - obj[..., -q:].min(dim=-1).values

        return diffs.clamp_min(0).mean(dim=0)
