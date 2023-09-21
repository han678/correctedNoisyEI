import argparse
import json
import logging
import os
import time

import torch
import numpy as np
from monai.utils import set_determinism
from opt.bo import BO
from examples.compression_function import CompressVGG16, CompressResnet50, CompressFC3

parser = argparse.ArgumentParser(description='Compress model via BO')

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--iter', type=int, default=1, metavar='N',
                    help='number of iterations to train (default: 200)')
parser.add_argument('--n_init', type=int, default=1, metavar='N',
                    help='number of initial points (default:30)')
parser.add_argument('--comp_obj', type=int, default=1, metavar='N',
                    help='use norm (=0) or top-1 error rate(=1) in the compression objective')
parser.add_argument('--test_size', type=int, default=50, metavar='N',
                    help='size of the test set')
parser.add_argument('--model', type=str, default="Resnet50", metavar='N',
                    help='VGG16 model or FC3 or Resnet50')


def read_results(bo):
    all_info = {}
    # record the maximum of obs_y
    obs_best_y = np.Inf
    for i in range(int(n_iter + n_init)):
        obs_x = bo.obs.X[i].tolist()
        obs_y = bo.obs.y[i].item()
        if obs_best_y >= obs_y:
            obs_best_y = obs_y
        all_info.update(
            {"iteration {}".format(i + 1): {"opt_params": obs_x, "obj_info": obs_y, "best_observed_info": obs_best_y}})
    return all_info


if __name__ == '__main__':
    args = parser.parse_args()
    acqs = ["q_NEI", "NEI", 'PI', 'UCB', "EI_C", 'PI_C', 'EI']
    # run experiments
    tkwargs = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}
    if args.model == "Resnet50":
        obj_func = CompressResnet50(comp_obj=args.comp_obj, tkwargs=tkwargs, n_sample=args.test_size, negate=False)
    elif args.model == "VGG16":
        obj_func = CompressVGG16(comp_obj=args.comp_obj, tkwargs=tkwargs, n_sample=args.test_size, negate=False)
    elif args.model == "FC3":
        obj_func = CompressFC3(comp_obj=args.comp_obj, tkwargs=tkwargs, n_sample=args.test_size, negate=False)
    logger = logging.getLogger(obj_func.name)
    logger.setLevel(logging.DEBUG)
    output_dir = "./comp_results/{}_{}".format(obj_func.name, args.test_size)
    os.makedirs(output_dir, exist_ok=True)
    seed = args.seed
    n_iter = args.iter
    n_init = args.n_init
    for acq in acqs:
        set_determinism(seed + 123456)
        bo = BO(obj_func, acq_kind=acq, initial_design="sobol", transform_inputs=True)
        bo.initialize(n_init)
        bo.inference(n_iter)
        dir = os.path.join(output_dir, acq, str(seed))
        os.makedirs(dir, exist_ok=True)
        fn_path = os.path.join(dir, "info.json")
        info = read_results(bo)
        with open(fn_path, "w") as f:
            json.dump(info, f)
