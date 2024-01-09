import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision
from torch.utils.data import Subset
import random

from CNN import CNN

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", type=str, default="models", help='name of model directory - default \"models\"')
parser.add_argument('--train_batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument("--warmup_epochs", type=int, default=0, help='number of epochs to train without regularization for - default 0')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--interval_width', type=float, default=0.01, help="input interval width for interval propogation penalty - default 0.01")
parser.add_argument("--interval_prop", action='store_true', help='use flag to add interval bound propagation penalty (L_IBP)')
parser.add_argument("--asymmetric_l2", action='store_true', help='use flag to add asymmetric L2 penalty (L_v)')
parser.add_argument("--alpha", type=float, default=0.9, help='weighting for base loss function - default 0.9')
parser.add_argument("--beta", type=float, default=0.9, help='weighting for interval propogation regularization - default 0.9')
parser.add_argument("--gamma", type=float, default=0.9, help='weighting for asymmetric l2 regularization - default 0.8')
parser.add_argument("--cnn_model", type=float, default=0, help='cnn model to use (0 or 1)  - default 0')
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
args = parser.parse_args()

def split_train_val(org_train_set, shuffle=False, valid_ratio=0.1, val_idx=None):
    num_train = len(org_train_set)

    if val_idx is None:
        if valid_ratio < 1:
            split = int(np.floor(valid_ratio * num_train))
        else:
            split = valid_ratio

        indices = list(range(num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        new_train_set = Subset(org_train_set, train_idx)
        val_set = Subset(org_train_set, val_idx)

        assert num_train - split == len(new_train_set)
        assert split == len(val_set)
    else:
        all_indices = set(list(range(len(org_train_set))))
        train_idx = list(all_indices - set(val_idx))

        new_train_set = Subset(org_train_set, train_idx)
        val_set = Subset(org_train_set, val_idx)

        assert num_train == (len(new_train_set) + len(val_set))

    return new_train_set, val_set, train_idx, val_idx
  
def load_train_val_dataset(shuffle=True):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    crop = torchvision.transforms.RandomCrop(32, padding=4)
    flip = torchvision.transforms.RandomHorizontalFlip()
    train_dataset = torchvision.datasets.CIFAR10('./datasets/',
                                            train=True,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                crop, flip, torchvision.transforms.ToTensor(),
                                                normalize, 
                                            ]))

    train_dataset, val_dataset, train_idx, val_idx = split_train_val(train_dataset, shuffle=shuffle, valid_ratio=1.0 / 6.0)
    return train_dataset, val_dataset, train_idx, val_idx

def load_test_dataset(trans_type=None, trans_arg=None):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      normalize
    ])

    test_dataset = torchvision.datasets.CIFAR10('./datasets/',
                                            train=False,
                                            download=True,
                                            transform=transforms)

    return test_dataset

def prepare_loaders(train_dataset,
                    val_dataset,
                    test_dataset,
                    train_batch_size,
                    test_batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=train_batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, val_loader, test_loader

def load_data(train_batch_size, test_batch_size):
    train_dataset, val_dataset, _, val_idx = load_train_val_dataset(shuffle=True)
    test_dataset = load_test_dataset(trans_type=None)

    return prepare_loaders(train_dataset,
                           val_dataset,
                           test_dataset,
                           train_batch_size,
                           test_batch_size), val_idx

def train(train_loader, model, criterion, optimizer, epoch, device, interval_halfwidth=0.01,
          interval=False, asym_l2=False,
          alpha=0.9, beta=0.9, gamma=0.9):
    model.train()

    train_loss = 0.0
    correct = 0
    for i, (xs, ys) in enumerate(train_loader):
        xs = xs.to(device)
        ys = ys.to(device)

        optimizer.zero_grad()
        outputs = model(xs)

        # L0 = O
        loss = criterion(outputs, ys)

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
        halfwidths = torch.from_numpy(np.asarray(interval_halfwidth * state_ranges)).to(device)
        upper = xs + halfwidths
        lower = xs - halfwidths

        # L = a * (b*O + (1-b)*IP  ) + (1 - a) * AL2 
        if interval and asym_l2: 
            loss_IP = model.width(lower.float(), upper.float()).max()
            loss = beta * loss + (1-beta) * loss_IP

            loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
            loss = alpha * loss + (1-alpha) * loss_AL2

        # L = b*O + (1-b)*IP
        if interval and not asym_l2:
            loss_IP = model.width(lower.float(), upper.float()).max()
            loss = beta * loss + (1-beta) * loss_IP

        # L = b*O + (1-b)*AL2
        if not interval and asym_l2:     
            loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
            loss = alpha * loss + (1-alpha) * loss_AL2             
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += (pred.eq(ys.data.view_as(pred)).sum().item())

        if i % 100 == 0:
          print('\nEpoch: {}, {}/{} loss: {:.3f}'.format(epoch, i+1, len(train_loader), train_loss / (i+1)))
          
          if interval:
            print(f'Interval loss: {loss_IP:.6f}')
          if asym_l2:
            print(f'L2 positive loss: {l2_loss_pos:.6f}')
            print(f'L2 negative loss: {l2_loss_neg:.6f}')

    train_acc = 1.0 * correct / len(train_loader.dataset)
    print('Epoch: {}, Avg. Loss: {:.4f} Train Accuracy: {}/{} ({:.2f}%)'.format(epoch,
                                                                                train_loss / len(train_loader),
                                                                                correct,
                                                                                len(train_loader.dataset),
                                                                                100.0 * train_acc))


    return train_loss, train_acc

def test(test_loader, model, criterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (xs, ys) in enumerate(test_loader):
          xs = xs.to(device)
          ys = ys.to(device)

          output = model(xs)
          test_loss += criterion(output, ys)

          pred = output.data.max(1, keepdim=True)[1]
          correct += (pred.eq(ys.data.view_as(pred)).sum().item())

    test_acc = 1.0 * correct / len(test_loader.dataset)
    print('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        (100. * test_acc)))

    return test_loss, test_acc

def run(args):
    print(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = CNN(args.cnn_model)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    (train_loader, val_loader, test_loader), val_idx = load_data(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)

    def scheduler(epoch):
        if epoch < 100:
            return 0.01
        elif epoch < 150:
            return 0.005
        else:
            return 0.001

    best_acc = 0.0
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    filepath ='CNN_model_' + str(int(args.cnn_model))
    if args.interval_prop:
        filepath += '_interval_beta_' + str(args.beta)
    if args.asymmetric_l2:
        filepath += '_asymL2_alpha_' + str(args.alpha) + '_gamma_' + str(args.gamma)
    filepath += '_seed' + str(args.seed)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: scheduler(e))
    for epoch in range(args.epochs):
        if epoch <= args.warmup_epochs:
            train(train_loader, model, criterion, optimizer, epoch, device, interval=False, asym_l2=False)
        else:
            train(train_loader, model, criterion, optimizer, epoch, device, interval_halfwidth=args.interval_width/0.225, interval=args.interval_prop, asym_l2=args.asymmetric_l2, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        test_loss, test_acc = test(test_loader, model, criterion, device)
        lr_scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            data = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'val_idx': val_idx,
                'lr': args.lr,
                'momentum': args.momentum,
                'interval_width': args.interval_width
            }
            torch.save(data, os.path.join(args.model_dir, filepath + "_info.pth"))

    data = {
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'val_idx': val_idx,
        'lr': args.lr,
        'momentum': args.momentum,
        'interval_width': args.interval_width
    }
    torch.save(data, filepath + "_info.pth")
    torch.save(model.state_dict(), os.path.join(args.model_dir, filepath + ".pth"))

if __name__ == '__main__':
    run(args)
