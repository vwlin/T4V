"""
Copyright (c) 2021 Kaidi Xu, Huan Zhang, Shiqi Wang and Yihan Wang
All rights reserved.

Minor modifications made by Vivian Lin.
"""

from modules import Flatten
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
from collections import OrderedDict
import sys
sys.path.append('../..')

from FC import FC

########################################
# Defined the model architectures
########################################

def mnist_model(num_layers=2, layer_size=50):
    temp = FC(in_dim=28*28, out_dim=10, num_hidden_layers=num_layers, layer_size=layer_size)
    model = nn.Sequential(*temp.layer_list)
    return model

def lenet5_model():
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 6, 5)),
        ('relu1', nn.ReLU()),
        ('maxpool1', nn.MaxPool2d((2, 2), 2)),

        ('conv2', nn.Conv2d(6, 16, 5)),
        ('relu2', nn.ReLU()),
        ('maxpool2', nn.MaxPool2d((2, 2), 2)),

        ('flatten', Flatten()),

        ('lin1', nn.Linear(16 * 5 * 5, 120)),
        ('relu3', nn.ReLU()),
        ('lin2', nn.Linear(120, 84)),
        ('relu4', nn.ReLU()),
        ('lin3', nn.Linear(84, 10))
    ]))
    return model

def cnn_model():
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 8, 4, stride=2, padding=1)),
        ('relu1', nn.ReLU()),

        ('conv2', nn.Conv2d(8, 16, 4, stride=2, padding=1)),
        ('relu2', nn.ReLU()),

        ('flatten', Flatten()),

        ('lin1', nn.Linear(1024, 100)),
        ('relu3', nn.ReLU()),
        ('lin2', nn.Linear(100, 10))
    ]))
    return model

def cnn_model0():
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 8, 4, stride=1, padding=0)),
        ('relu1', nn.ReLU()),
        
        ('conv2', nn.Conv2d(8, 16, 4, stride=1, padding=0)),
        ('relu2', nn.ReLU()),
        
        ('flatten', Flatten()),
        
        ('lin1', nn.Linear(10816, 128)), # 26 * 26 * 16 = 10816
        ('relu3', nn.ReLU()),
        ('lin2', nn.Linear(128, 10))
    ]))
    return model

def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def load_model_data(args):
    model_ori = eval(args.model)()
    print(model_ori)
    print("[no_LP]:", args.no_LP)
    # loaded_model = torch.load(args.load)
    
    model_ori.load_state_dict(torch.load(args.load, map_location=torch.device(args.device))['state_dict'][0])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    test_data = datasets.CIFAR10("../data", train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor([0.485, 0.456, 0.406])
    test_data.std = torch.tensor([0.225, 0.225, 0.225])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))

    if args.data == 'CIFAR-deep':
        gt_results = pd.read_pickle('../data/deep.pkl')
    elif args.data == 'CIFAR-wide':
        gt_results = pd.read_pickle('../data/wide.pkl')
    elif args.data == 'CIFAR-easy':
        gt_results = pd.read_pickle('../data/base_easy.pkl')
    elif args.data == 'CIFAR-med':
        gt_results = pd.read_pickle('../data/base_med.pkl')
    elif args.data == 'CIFAR-hard':
        gt_results = pd.read_pickle('../data/base_hard.pkl')

    return model_ori, gt_results, test_data, data_min, data_max

def load_model_data_t4v(args):
    if args.model == "mnist_model":
        model_ori = eval(args.model)(num_layers=args.num_hidden_layers, layer_size=args.layer_size)
        print(model_ori)
        print("[no_LP]:", args.no_LP)
    

        model_ori.load_state_dict(torch.load(args.load, map_location=torch.device(args.device)))

        model_ori = nn.Sequential(Flatten(), *model_ori)

        #https://www.kaggle.com/berrywell/calculating-mnist-inverted-mnist-mean-std
        normalize = transforms.Normalize(mean=[0.1325], std=[0.3105])
        test_data = datasets.MNIST("../data", train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data.mean = torch.tensor([0.1325])
        test_data.std = torch.tensor([0.3105])
        # set data_max and data_min to be None if no clip
        data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1))
        data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1))

    else:
        model_ori = eval(args.model)()
        print(model_ori)
        print("[no_LP]:", args.no_LP)
        model_ori.load_state_dict(torch.load(args.load, map_location=torch.device(args.device)))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
        test_data = datasets.CIFAR10("../data", train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), normalize]))
        test_data.mean = torch.tensor([0.485, 0.456, 0.406])
        test_data.std = torch.tensor([0.225, 0.225, 0.225])
        # set data_max and data_min to be None if no clip
        data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))

    return model_ori, test_data, data_min, data_max