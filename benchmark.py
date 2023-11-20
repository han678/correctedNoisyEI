import json
import argparse
import os
import time
import numpy as np
import torch
from botorch.test_functions import Ackley, Levy, DropWave, EggHolder, Hartmann, Griewank, Rastrigin
from monai.utils import set_determinism
from opt.bo import BO
from examples.noisy_sythetic_functions import Noisy_synthetic_function, findExtrema

parser = argparse.ArgumentParser(description='BO Training for benchmark functions')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--iter', type=int, default=100, metavar='N',
                    help='number of iterations to train (default: 100)')
parser.add_argument('--noise_level', type=float, default=0.05)


def distance_to_optimal(x, optimal):
    return torch.cdist(x, optimal, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    tkwargs = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}
    acqs = ['NEI']# ['PI', 'UCB', "EI_C", 'PI_C', 'EI', "q_NEI", "NEI"]
    # run experiments
    bench_funcs = []
    noise_level = args.noise_level  # 0.006
    bench_funcs.append(Levy(dim=2, negate=False))
    seed = args.seed
    for function in bench_funcs:
        noisy_func = Noisy_synthetic_function(function, tkwargs=tkwargs)
        extreme = findExtrema(noisy_func.function)
        noisy_func.noise_std = noise_level * (extreme[1] - extreme[0])
        print(extreme)
        print("Optimum is:", noisy_func.function._optimal_value)
        output_dir = f"./benchmark_results/{noisy_func.name}_noise{noise_level}"
        os.makedirs(output_dir, exist_ok=True)
        while True:
            try:
                set_determinism(seed + 123456)
                for acq in acqs:
                    bo = BO(noisy_func, acq_kind=acq, initial_design="sobol", transform_inputs=True)
                    bo.initialize(int(noisy_func.dim * 3))
                    st = time.time()
                    bo.inference(max_iter=args.iter)
                    print("==> time is {}.".format(time.time() - st))
                    # record the best point
                    acq_dir = os.path.join(output_dir, acq, str(seed))
                    os.makedirs(acq_dir, exist_ok=True)
                    fn_path = os.path.join(acq_dir, f"info.json")
                    all_info = {}
                    # get best objective of the observation set
                    obs_best_f = np.Inf
                    for i in range(len(bo.obs.y)):
                        obs_x = bo.obs.X[i].tolist()
                        obs_y = bo.obs.y[i].tolist()
                        true_f = noisy_func.function.evaluate_true(torch.tensor(obs_x)).tolist()
                        if true_f <= obs_best_f:
                            obs_best_f = true_f
                        all_info.update({f"iteration {i + 1}": {"x": obs_x, "y": obs_y, "best_obj": obs_best_f,
                                                                "acq_x": bo.best_observed_acq[i]}})
                    with open(fn_path, "w") as f:
                        json.dump(all_info, f)
                break  # If the above succeeds, we break here
            except Exception as e:
                print(e)
                seed = seed + 10
                time.sleep(1)
