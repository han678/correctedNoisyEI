import time
import torch
from monai.utils import set_determinism
from opt.bo import BO
import os
from examples.noisy_sythetic_functions import Noisy_synthetic_function, findExtrema, TestGaussian
import numpy as np

if __name__ == '__main__':
    seed = 1234567
    while True:
        try:
            set_determinism(seed)
            n_init = 4
            n_iter = 4
            tkwargs = {"dtype": torch.double, "device": "cpu"}
            noise_level = 0.3
            noisy_func = Noisy_synthetic_function(TestGaussian(negate=False), tkwargs=tkwargs)
            extreme = findExtrema(noisy_func.function)
            noisy_func.noise_std = noise_level * (extreme[1] - extreme[0])
            print(extreme)
            bo = BO(noisy_func, acq_kind="EI", initial_design="sobol", transform_inputs=True)
            dir = os.path.join("toy_result")
            if not os.path.exists(dir):
                os.makedirs(dir)
            bo.initialize(n_init)
            st = time.time()
            bo.inference(max_iter=n_iter)
            break  # If the above succeeds, we break here
        except Exception as e:
            print(e)
            print("BOtorch fail to fit the model, try re-initialization")
            seed = +1
            time.sleep(1)

    print("==> time is {}.".format(time.time() - st))
    # get best objective of the observation set
    obs_best_f = np.Inf
    for i in range(len(bo.obs.y)):
        obs_x = bo.obs.X[i].tolist()
        obs_y = bo.obs.y[i].tolist()
        true_f = noisy_func.function.evaluate_true(torch.tensor(obs_x))
        if true_f <= obs_best_f:
            obs_best_f = true_f
        print(f"iteration {i + 1}: params: {obs_x}, y: {obs_y}, best_obj_info: {obs_best_f}")
        true_f = noisy_func.function.evaluate_true(torch.tensor(obs_x))
        if obs_best_f >= true_f:
            obs_best_f = true_f
        print(f"iteration {i + 1}: params: {obs_x}, best_obj_info: {obs_best_f}")

    # get the best observed point
    best_x = bo.obs.best_observed(return_y=False)
    true_value = noisy_func.function.evaluate_true(best_x).unsqueeze(-1)
    print("best observed x:", best_x)
    print("True Function value is:", true_value)
    print("Optimum is:", noisy_func.function.optimal_value)
    # plot all
    bo.plot_all(n_init, fig_dir=dir)
