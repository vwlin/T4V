"""
Code credit: Oscar Knagg
Modifications made by Vivian Lin
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from CNN import CNN

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, help='Load pretrained model')
parser.add_argument("--seed", type=int, default=100, help='random seed - default 100')
parser.add_argument("--eps", type=float, default=1, help='epsilon, normalized - default 1')
parser.add_argument('--interval_width', type=float, default=0.01, help="input interval width for interval propogation penalty - default 0.01")
parser.add_argument("--interval_prop", action='store_true', help='use flag to add interval bound propagation penalty (L_IBP)')
parser.add_argument("--asymmetric_l2", action='store_true', help='use flag to add asymmetric L2 penalty (L_v)')
parser.add_argument("--alpha", type=float, default=0.9, help='weighting for base loss function - default 0.9')
parser.add_argument("--beta", type=float, default=0.9, help='weighting for interval propogation regularization - default 0.9')
parser.add_argument("--gamma", type=float, default=0.8, help='weighting for asymmetric l2 regularization - default 0.8')
parser.add_argument("--steps", type=int, default=40, help='maximum number of steps for PGD - default 40')
parser.add_argument("--cnn_model", type=float, default=0, help='cnn model to use (0 or 1)  - default 0')
args = parser.parse_args()

def new_loss(x, y, prediction, net, interval_halfwidth, interval, asym_l2, alpha, beta, gamma):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction, y)

    # AL2 loss
    l2_loss_pos = 0
    l2_loss_neg = 0
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            pos_weight = torch.relu(parameter)
            neg_weight = -torch.relu(-parameter)
            l2_loss_pos = l2_loss_pos + torch.norm(pos_weight)
            l2_loss_neg = l2_loss_neg + torch.norm(neg_weight)

    # IP loss
    state_ranges = np.ones((model.in_dim))
    halfwidths = torch.from_numpy(np.asarray(interval_halfwidth * state_ranges))

    upper = x + halfwidths
    lower = x - halfwidths

    loss_IP = model.width(lower.float(), upper.float()).max()

    # L = a * (b*O + (1-b)*IP  ) + (1 - a) * AL2 
    if interval == 1 and asym_l2 == 1: 
        loss_IP = model.width(lower.float(), upper.float()).max()
        loss = beta * loss + (1-beta) * loss_IP

        loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
        loss = alpha * loss + (1-alpha) * loss_AL2

    # L = b*O + (1-b)*IP
    if interval == 1 and asym_l2 == 0:
        loss_IP = model.width(lower.float(), upper.float()).max()
        loss = beta * loss + (1-beta) * loss_IP

    # L = b*O + (1-b)*AL2
    if interval == 0 and asym_l2 == 1:     
        loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
        loss = alpha * loss + (1-alpha) * loss_AL2                 
    
    return loss


def projected_gradient_descent(model, x, y, data_max, data_min, interval_halfwidth, interval, asym_l2, alpha, beta, gamma, num_steps, step_size, step_norm, eps, eps_norm,
                               clamp=(0,1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = F.log_softmax(model(_x_adv), dim=1)
        if targeted:
            y = y_target
        loss = new_loss(x=_x_adv, y=y, prediction=prediction, net=model, interval_halfwidth=interval_halfwidth, interval=interval, asym_l2=asym_l2, alpha=alpha, beta=beta, gamma=gamma)
        loss.backward()

        with torch.no_grad():
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                    .norm(step_norm, dim=-1)\
                    .view(-1, num_channels, 1, 1)

            if targeted:
                x_adv -= gradients
            else:
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            upper = torch.min(x + eps, data_max)
            lower = torch.max(x - eps, data_min)
            x_adv = torch.max(torch.min(x_adv, upper), lower)
        else:
            delta = x_adv - x

            mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
            scaling_factor[mask] = eps

            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta

    return x_adv.detach()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.eps = args.eps / 255 / 0.225

    model = CNN(args.cnn_model)
    model.eval()
    model.load_state_dict(torch.load(args.load))

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.225, 0.225, 0.225])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    test = datasets.CIFAR10('../data/', train=False,
                       transform=transforms.Compose([transforms.ToTensor(), normalize]),
                       download=True)
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    num_images = 200

    test_loader = DataLoader(test, batch_size=num_images, shuffle=False)

    total_adversarial_imgs = 0
    total_correctly_classified = 0

    for x, y, in test_loader:
        x_adv = projected_gradient_descent(model=model, x=x, y=y,
                                           data_max=data_max, data_min=data_min,
                                           interval_halfwidth=args.interval_width,
                                           interval=args.interval_prop,
                                           asym_l2=args.asymmetric_l2,
                                           alpha=args.alpha,
                                           beta=args.beta,
                                           gamma=args.gamma,
                                           num_steps=args.steps, step_size=0.01,
                                           eps=args.eps, eps_norm='inf',
                                           step_norm='inf')

        for i in range(0,num_images):
            prediction = torch.max(model(torch.unsqueeze(x[i],0))[0], 0)[1].item()
            if prediction != y[i]:
                continue
            total_correctly_classified += 1
            adv_prediction = torch.max(model(torch.unsqueeze(x_adv[i],0))[0], 0)[1].item()
            if prediction != adv_prediction:
                total_adversarial_imgs += 1

        break

    print('Total number of adversarial examples:', total_adversarial_imgs)
    print('Total number of correctly classified images:', total_correctly_classified)
    print('Percent of correctly classified images found unsafe by PGD attack test:', 100*total_adversarial_imgs/total_correctly_classified)