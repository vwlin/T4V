import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset

from FC import FC

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="models", help='name of model directory - default \"models\"')
parser.add_argument("--num_hidden_layers", type=int, default=2, help='number of hidden layers - default 2')
parser.add_argument("--layer_size", type=int, default=50, help='number of neurons per hiddnen layer - default 50')
parser.add_argument("--epochs", type=int, default=10, help='number of epochs for training - default 10')
parser.add_argument("--warmup_epochs", type=int, default=0, help='number of epochs to train without regularization for - default 0')
parser.add_argument('--interval_width', type=float, default=0.04, help="input interval width for interval propogation penalty - default 0.04")
parser.add_argument("--interval_prop", action='store_true', help='use flag to add interval bound propagation penalty (L_IBP)')
parser.add_argument("--asymmetric_l2", action='store_true', help='use flag to add asymmetric L2 penalty (L_v)')
parser.add_argument("--seed", type=int, default=0, help='random seed for training - default 0')
parser.add_argument("--alpha", type=float, default=0.9, help='weighting for asymmetric L2 regularization - default 0.9')
parser.add_argument("--beta", type=float, default=0.9, help='weighting for interval bound propogation regularization - default 0.9')
parser.add_argument("--gamma", type=float, default=0.9, help='weighting for asymmetric l2 regularization - default 0.8')
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
    train_dataset = torchvision.datasets.MNIST('./datasets/',
                                            train=True,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                            ]))

    train_dataset, val_dataset, train_idx, val_idx = split_train_val(train_dataset, shuffle=shuffle, valid_ratio=1.0 / 6.0)
    return train_dataset, val_dataset, train_idx, val_idx


def load_test_dataset(trans_type=None, trans_arg=None):
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.MNIST('./datasets/',
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


def train(net, loader, optimizer, epoch, log_interval=100, interval_halfwidth=0.04,
          interval=False, asym_l2=False,
          alpha=0.9, beta=0.9, gamma=0.9):
    net.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = F.log_softmax(net(data), dim=1)

        new_data = data.view(-1, net.in_dim)

        # Prepare for IBP calculations
        state_ranges = np.ones((net.in_dim))
        halfwidths = torch.from_numpy(np.asarray(interval_halfwidth * state_ranges)).cuda()
        upper = new_data + halfwidths # scale half-width by state ranges
        lower = new_data - halfwidths

        # Prepare for AL2 calculations
        l2_loss_pos = 0
        l2_loss_neg = 0
        for name, parameter in net.named_parameters():
            if 'weight' in name:
                pos_weight = torch.relu(parameter)
                neg_weight = -torch.relu(-parameter)
                l2_loss_pos = l2_loss_pos + torch.norm(pos_weight)
                l2_loss_neg = l2_loss_neg + torch.norm(neg_weight)

        # L0 = O
        original_loss = F.nll_loss(output, target)
        loss = original_loss

        # L = a*(b*O + (1-b)*IP) + (1-a)*AL2 
        if interval and asym_l2: 
            loss_IP = net.width(lower.float(), upper.float()).max()
            loss = beta * loss + (1-beta) * loss_IP

            loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
            loss = alpha * loss + (1-alpha) * loss_AL2

        # L = b*O + (1-b)*IP
        if interval and not asym_l2:
            loss_IP = net.width(lower.float(), upper.float()).max()
            loss = beta * loss + (1-beta) * loss_IP

        # L = a*O + (1-a)*AL2
        if not interval and asym_l2:     
            loss_AL2 = (1 - gamma) * l2_loss_pos + gamma * l2_loss_neg
            loss = alpha * loss + (1-alpha) * loss_AL2 

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))

            print(f'Original loss: {original_loss:.6f}')
            if interval:
                print(f'Interval loss: {loss_IP:.6f}')
            if asym_l2:
                print(f'L2 positive loss: {l2_loss_pos:.6f}')
                print(f'L2 negative loss: {l2_loss_neg:.6f}')

    print('Accuracy: {:.2f}%'.format(100.0 * correct / len(loader.dataset)))

def test(net, loader):
    net.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))))

    return 100.0 * correct / len(loader.dataset)

def run(model_dir="models",
        num_hidden_layers=2, layer_size=50,
        train_batch_size=100, test_batch_size=100,
        n_epochs=10, n_warmup_epochs=0, learning_rate=1e-4,
        interval=False, asym_l2=False,
        seed=100,
        alpha=0.9, beta=0.9, gamma=0.8):
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    (train_loader, val_loader, test_loader), val_idx = load_data(train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    network = FC(in_dim=28*28, out_dim=10, num_hidden_layers=num_hidden_layers, layer_size=layer_size)
    if torch.cuda.is_available():
        network = network.cuda()

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    test(network, test_loader)

    for epoch in range(1, n_epochs + 1):
        if epoch <= n_warmup_epochs:
            train(network, train_loader, optimizer, epoch, interval=False, asym_l2=False)
        else:
            train(network, train_loader, optimizer, epoch, interval_halfwidth=args.interval_width, interval=interval, asym_l2=asym_l2, alpha=alpha, beta=beta, gamma=gamma)
        test(network, test_loader)

    # save network
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    alpha_str = str(alpha).replace(".","-")
    beta_str = str(beta).replace(".","-")
    gamma_str = str(gamma).replace(".","-")
    i_str = str(args.interval_width).replace(".","-")

    model_path = f'{num_hidden_layers}x{layer_size}'
    if interval and asym_l2:
        model_path += f'_interval_asymL2_alpha{alpha_str}_beta{beta_str}_gamma{gamma_str}_i{i_str}'
    if interval and not asym_l2:
        model_path += f'_interval_beta{beta_str}_i{i_str}'
    if not interval and asym_l2:
        model_path += f'_asymL2_alpha{alpha_str}_gamma{gamma_str}'
    model_path += f'_seed{seed}.pth'

    # save like this to work with lirpa verify (FAC)
    current_dict = network.state_dict()
    prefix = 'layer_list.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in current_dict.items()
                    if k.startswith(prefix)}
    torch.save(adapted_dict, os.path.join(model_dir,model_path))

if __name__ == '__main__':
  run(model_dir=args.model_dir,
        num_hidden_layers=args.num_hidden_layers, layer_size=args.layer_size,
        n_epochs=args.epochs, n_warmup_epochs=args.warmup_epochs,
        learning_rate=1e-4,
        interval=args.interval_prop,
        asym_l2=args.asymmetric_l2,
        seed=args.seed,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma
    )
