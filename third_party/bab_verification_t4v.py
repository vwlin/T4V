"""
Copyright (c) 2021 Kaidi Xu, Huan Zhang, Shiqi Wang and Yihan Wang
All rights reserved.

Minor modifications made by Vivian Lin.
"""

import argparse
import copy
import random
import sys
import time
import gc
import math
sys.path.append('../auto_LiRPA')

from model_CROWN_gurobi import LiRPAConvNet
from relu_conv_parallel_LP_t4v import relu_bab_parallel

from utils_t4v import load_model_data_t4v
import numpy as np

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *

parser = argparse.ArgumentParser()

parser.add_argument('--no_LP', action='store_true', help='verification with CROWN only without guroby')
parser.add_argument('--no_solve_slope', action='store_false', dest='solve_slope', help='do not optimize slope/alpha in compute bounds')
parser.add_argument("--load", type=str, default="models/cifar_base_kw.pth", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="CIFAR-hard", choices=["CIFAR-easy", "CIFAR-med", "CIFAR-hard",
                                                                       "CIFAR-wide", "CIFAR-deep"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cifar_model",
                    choices=["mnist_model", "lenet5_model", "cnn_model", "cifar_model", "cifar_model_deep", "cifar_model_wide", "cnn_model0"],
                    help='model name')

parser.add_argument("--num_hidden_layers", type=int, default=1, help='used for mnist_model')
parser.add_argument("--layer_size", type=int, default=50, help='used for mnist_model')
parser.add_argument("--batch_size", type=int, default=200, help='batch size')
parser.add_argument("--bound_opts", type=str, default="same-slope", choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options for relu')
parser.add_argument('--no_warm', action='store_true', default=False, help='using warm up for lp solver, true by default')
parser.add_argument("--max_subproblems_list", type=int, default=12000, help='max length of sub-problems list')
parser.add_argument("--growth_rate", type=float, default=0, help='minimal growth_rate gained by CROWN')
parser.add_argument("--decision_thresh", type=float, default=0, help='decision threshold of lower bounds')
parser.add_argument("--lr_alpha", type=float, default=0.1, help='learning rate for relu slopes/alpha')
parser.add_argument("--start", type=int, default=0, help='start from i-th property')
parser.add_argument("--timeout", type=int, default=3600, help='timeout for one property')
parser.add_argument("--eps", type=float, default=0.01, help='epsilon, unnormalized')
parser.add_argument("--img", type=int, default=0, help='index of image to be evaluated')


args = parser.parse_args()

def bab(model_ori, data, target, norm, eps, args, data_max=None, data_min=None):

    if norm == np.inf:
        if data_max is None:
            data_ub = data + eps  # torch.min(data + eps, data_max)  # eps is already normalized
            data_lb = data - eps  # torch.max(data - eps, data_min)
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data

    pred = torch.argmax(model_ori(data), dim=1)
    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, pred, target, solve_slope=args.solve_slope, device=args.device, in_size=data.shape)

    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    min_lb, min_ub, ub_point, nb_states, max_length = relu_bab_parallel(model, domain, x, batch=args.batch_size, no_LP=args.no_LP,
                                                            decision_thresh=args.decision_thresh, lr_alpha=args.lr_alpha,
                                                            max_subproblems_list=args.max_subproblems_list, timeout=args.timeout)
    
    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states, max_length


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    
    if args.model == "mnist_model":
        args.eps = args.eps / 255
    else:
        args.eps = args.eps / 255.0 / 0.225

    model_ori, test_data, data_min, data_max = load_model_data_t4v(args)

    ret = []

    imag_idx = args.img

    print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% img ID:', imag_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    running_time = 0

    x, y = test_data[imag_idx]

    x = x.unsqueeze(0)
    model_ori.to('cpu')
    # first check the model is correct at the input
    y_pred = torch.max(model_ori(x)[0], 0)[1].item()

    num_unsafe_properties = 0
    num_timeouts = 0

    print('predicted label ', y_pred, ' correct label ', y)

    if y_pred != y:
        print('model prediction is incorrect for the given model')
        return
    else:
        for prop_idx in range(10):
            if prop_idx == y_pred:
                continue

            print('\ntested against ', prop_idx)
            torch.cuda.empty_cache()
            gc.collect()

            start = time.time()
            try:
                l, nodes, max_length = bab(model_ori, x, prop_idx, args.norm, args.eps, args, data_max=data_max, data_min=data_min)
                temp_time = time.time() - start
                print('Image {} verify end, Time cost: {}'.format(imag_idx, temp_time))
                running_time += temp_time
                ret.append([prop_idx, l, nodes, temp_time, max_length])
                np.save('ICLR_{}_{}_alpha01_005_iter20_b{}_LP_{}_multi_alpha.npy'.
                        format(args.model, args.data, args.batch_size, not args.no_LP), np.array(ret))
                if(math.isnan(l)):
                    print('Image {}, property {}: TIMEOUT'.format(imag_idx, prop_idx))
                    num_timeouts += 1
                elif(l > args.decision_thresh):
                    print('Image {}, property {}: SAFE'.format(imag_idx, prop_idx))
                else:
                    print('Image {}, property {}: UNSAFE'.format(imag_idx, prop_idx))
                    num_unsafe_properties += 1
            except KeyboardInterrupt:
                print('time:', imag_idx, time.time()-start, "\n",)
                break

    # some results analysis
    np.set_printoptions(suppress=True)
    print('\n{} safe, {} unsafe, {} timeouts'.format(10 - num_unsafe_properties - num_timeouts, num_unsafe_properties, num_timeouts))
    ret = np.array(ret)
    print(ret)
    print('PER PROPERTY\ntime mean: {}, branches mean: {}, number of timeouts: {}'.format(
        ret[:, 3].mean(), ret[:, 2].mean(), (ret[:, 1] < 0).sum()))
    print('ACROSS ALL PROPERTIES\ntotal time: {}'.format(running_time))
    print('Image {} did not time out.'.format(imag_idx))

if __name__ == "__main__":
    main(args)