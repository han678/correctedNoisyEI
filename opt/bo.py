import os

import botorch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

from opt.acquisition import Acq, best_pred
from opt.observation import Observation, unstandardize_y


def set_model_prior(model):
    model.likelihood.noise_covar._noise_constraint = GreaterThan(1.000E-08)
    model.covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(2, 1.5), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", GammaPrior(2.0, 0.15), "outputscale")


def optimize_gp(X, y, ystd):
    # define the model for objective
    # model = botorch.models.HeteroskedasticSingleTaskGP(train_X=X, train_Y=y, train_Yvar=ystd ** 2, ).to(X)
    model = botorch.models.FixedNoiseGP(train_X=X, train_Y=y, train_Yvar=ystd ** 2, ).to(X)
    #model.likelihood.noise_covar._noise_constraint = GreaterThan(0)
    # set_model_prior(model)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit the model
    fit_gpytorch_mll(mll)
    return mll, model


class BO(object):

    def __init__(self, obj_func, acq_kind, initial_design="sobol", transform_inputs=True):
        self.obj_func = obj_func
        self.initial_design = initial_design
        self.acq_kind = acq_kind
        self.tkwargs = obj_func.tkwargs
        self.acq = None
        self.obs = Observation(obj_func.bounds, obj_func.tkwargs)
        self.best_xs = []
        self.best_observed_acq = []
        self.transform_inputs = transform_inputs

    def initialize(self, n):
        # TODO: sample intial points from prior density
        if self.initial_design == "uniform":
            init_samples = self.obs.random_uniform_sample(n)
        elif self.initial_design == "beta":
            init_samples = self.obs.beta_sample(n)
        elif self.initial_design == "sobol":
            init_samples = self.obs.sobol_samples(n).to(**self.tkwargs)
        # y_err is the noisy std that we added to the true objective
        for x in init_samples:
            y, y_err = self.obj_func.evaluate_observed(x)  # y_err is the noisy level that we added
            if len(y.shape) == 1:
                self.obs.add(x.tolist(), y.tolist(), y_err.tolist())
            else:
                self.obs.add(x.tolist(), y.tolist()[0], y_err.tolist()[0])
            acqx = np.Inf
            self.best_observed_acq.append(acqx)

    def inference(self, max_iter):
        i = 0
        while i < max_iter:
            train_X = self.obs.X
            if self.transform_inputs:
                # normalize the input, and standardize y and y_std
                train_X = self.obs.normalize_x(x=train_X)
                train_y, train_ystd = self.obs.y_standardize
            else:
                train_y, train_ystd = self.obs.y, self.obs.y_err
            _, self.gp = optimize_gp(X=train_X, y=train_y, ystd=train_ystd)
            # find the point that returns the best predictive mean from the observation set
            x_best = best_pred(model=self.gp, train_X=train_X, return_pred=False, maximize=False)
            if self.transform_inputs:
                x_best = self.obs.unnormalize_x(x_norm=x_best)
            self.best_xs.append(x_best.tolist())
            # optimize acq and get new observation
            next_x, acq_x = self.maximize_acq()  # may return normalized input data
            i += 1
            self.best_observed_acq.append(acq_x.tolist())
            if self.transform_inputs:
                self.obs.pending = self.obs.unnormalize_x(x_norm=next_x)
            else:
                self.obs.pending = next_x.squeeze(0)
            while self.obs.check_duplication(self.obs.pending):
                print("warning ****, resampled.", self.obs.pending)
                self.obs.pending = self.obs.random_uniform_sample(1).squeeze(0).detach()
            # evaluate the point and add it to the observation set
            y, ystd = self.obj_func.evaluate_observed(self.obs.pending)  # observed y: may contain noise
            self.obs.add(self.obs.pending.tolist()[0], y.tolist()[0], ystd.tolist()[0])

    def maximize_acq(self):
        if self.transform_inputs:
            train_X = self.obs.normalize_x(x=self.obs.X)
            bounds = self.obs.normalize_x(x=self.obj_func.bounds)
        else:
            train_X = self.obs.X
            bounds = self.obj_func.bounds
        acq_func = Acq(self.acq_kind, maximize=False).func(self.gp, train_X)
        candidates, acq_vals = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=4,
            raw_samples=256,  # used for intialization heuristic
            options={"maxiter": 5},
        )
        # observe new values
        new_x = candidates.to(**self.tkwargs).detach()
        new_x = torch.clamp(new_x, min=bounds[0], max=bounds[1])
        new_acqx = acq_vals.detach().to(**self.tkwargs)
        return new_x, new_acqx

    def plot_all(self, n_init, fig_dir):
        # figure 1
        pmin, pmax = self.obs.bounds.T[0].tolist()
        x_test = torch.linspace(pmin, pmax, 1000)[:, None]
        if self.obj_func.function.negate == True:
            f_test = -self.obj_func.function.evaluate_true(x_test).detach().numpy()
        else:
            f_test = self.obj_func.function.evaluate_true(x_test).detach().numpy()
        if self.transform_inputs:
            x_test = self.obs.normalize_x(x=x_test)
        with torch.no_grad():
            # get the posterior of the function without noise
            posterior = self.gp.posterior(X=x_test, observation_noise=False)
            pred_mean, pred_std = posterior.mean, torch.sqrt(posterior.variance)
            if self.transform_inputs:
                pred_mean, pred_std = unstandardize_y(pred_mean, pred_std, self.obs.y[0:-1])
                x_test = self.obs.unnormalize_x(x_norm=x_test)
        x_test = x_test.detach().numpy()
        pred_mean = np.array(pred_mean.numpy())
        pred_std = np.array(pred_std.numpy())
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))
        plt.subplot(121)
        plt.plot(x_test, f_test, linewidth=1, label='Function')
        plt.plot(x_test, pred_mean, '-', color='navy', linewidth=1, label='GP Model')
        plt.fill_between(x_test.squeeze(), (pred_mean + 1.96 * pred_std).squeeze(),
                         (pred_mean - 1.96 * pred_std).squeeze(), alpha=.6,
                         color="cornflowerblue", ec='None', label='Confidence')
        plt.scatter(self.obs._X[0:n_init], self.obs._y[0:n_init], c="y", s=30, zorder=20, label='Initial points')
        plt.scatter(self.obs._X[n_init:-1], self.obs._y[n_init:-1], c="red", s=30, zorder=20, marker='^',
                    label=r'Proposed points $x_{1:t-1}$')
        # find the point that return the best predictive mean from observation set
        x_best = self.best_xs[-1]
        if self.transform_inputs:
            x_best = self.obs.normalize_x(x=torch.tensor(x_best))
        with torch.no_grad():
            posterior2 = self.gp.posterior(X=x_best, observation_noise=False)
            pred_x_best, pred_std_x_best = posterior2.mean, torch.sqrt(posterior2.variance)
            if self.transform_inputs:
                pred_x_best, _ = unstandardize_y(pred_x_best, pred_std_x_best, self.obs.y[0:-1])
        pred_x_best = pred_x_best.detach().numpy()
        plt.scatter(self.best_xs[-1], pred_x_best, c=20, s=100, zorder=25, marker='*', label='Best incumbent',
                    alpha=0.3)
        plt.axvline(x=self.best_xs[-1][0], ymax=1, color="blue", linestyle='dotted')
        plt.annotate(r'$x_t^+$', (self.best_xs[-1][0],  f_test.min()-0.08), fontsize=15)
        # draw observation set
        select_point = list(range(self.obs.X.shape[0] - n_init - 1))
        text = [str(x + 1) for x in select_point]
        for i, txt in enumerate(text):
            plt.annotate(txt, (self.obs._X[i + n_init][0], self.obs._y[i + n_init][0]), fontsize=20)
        plt.xlabel(r'$x$', fontdict={'size': 20})
        plt.ylabel(r'$f(x)$', fontdict={'size': 20})
        plt.legend()
        # figure 2
        train_X = self.obs.X[0:-1]
        if self.transform_inputs:
            train_X = self.obs.normalize_x(train_X)
        acq_ei = Acq(acq_kind="EI", maximize=False).func(self.gp, train_X)
        acq_ei_c = Acq(acq_kind="EI_C", maximize=False).func(self.gp, train_X)
        pmin, pmax = self.obs.bounds.T[0].tolist()
        x_test = torch.linspace(pmin, pmax, 1000)[:, None]
        if self.transform_inputs:
            x_test = self.obs.normalize_x(x=x_test)
        y_ei = acq_ei(x_test.unsqueeze(-1)).detach().numpy()
        y_ei_c = acq_ei_c(x_test.unsqueeze(-1)).detach().numpy()
        if self.transform_inputs:
            x_test = self.obs.unnormalize_x(x_norm=x_test)
        x_test = x_test.detach().numpy()
        plt.subplot(122)
        plt.plot(x_test, y_ei, linewidth=2, label='EI')
        x_best = self.best_xs[-1][0]
        plt.axvline(x=x_best, ymax=1, color="blue", linestyle='dotted')
        #plt.annotate(r'$x_t^+$', (x_best, f_test.min()), fontsize=15)
        plt.plot(x_test, y_ei_c, color='red', linewidth=2, label='EI_C')

        plt.xlabel(r'$x$', fontdict={'size': 20})
        plt.ylabel(r'$\alpha_t(x)$', fontdict={'size': 20})
        idx1 = np.argmax(y_ei)
        idx2 = np.argmax(y_ei_c)

        plt.scatter(x_test[idx1], y_ei[idx1], c="blue", s=60, zorder=30, label='next query by EI', alpha=0.5)
        plt.scatter(x_test[idx2], y_ei_c[idx2], c="red", s=60, zorder=30, marker='^', label='next query by EI_C',
                    alpha=0.5)
        plt.annotate(r'$x_t^+$', (self.best_xs[-1][0], 1e-9), fontsize=15)
        plt.legend(fontsize=15)
        fig_path = os.path.join(fig_dir, f"{self.obj_func.name}_plots.png")
        plt.savefig(fig_path)
        plt.show()
        plt.close(fig)
