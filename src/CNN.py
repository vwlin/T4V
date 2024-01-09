import torch as tc
import torch.nn as nn
import torch.nn.functional as F


##
## LeNet5 for CIFAR10
##
class CNN(nn.Module):
    
    def __init__(self, cnn_model=0):
        super().__init__()

        self.cnn_model = cnn_model

        self.in_dim = (3, 32, 32)

        self.n_labels = 10

        self.layer1_filters = 8
        self.layer2_filters = 16

        if self.cnn_model == 0:
            self.layer1_kernel_size = 4
            self.layer1_stride = 1
            self.layer1_padding = 0
            self.layer2_kernel_size = 4
            self.layer2_stride = 1
            self.layer2_padding = 0

            #NB: this calculations assume: 1) padding is 0; 2) stride is 1; 3) the kernel is square
            self.layer1_dim = self.in_dim[1] - self.layer1_kernel_size + 1
            self.layer2_dim = self.layer1_dim - self.layer2_kernel_size + 1         

        if self.cnn_model == 1:
            self.layer1_kernel_size = 4
            self.layer1_stride = 2
            self.layer1_padding = 1
            self.layer2_kernel_size = 4
            self.layer2_stride = 2
            self.layer2_padding = 1

            #NB: this calculations assume: 1) padding is 1; 2) stride is 2; 3) the kernel is square
            self.layer1_dim = int(self.in_dim[1]/2)
            self.layer2_dim = int(self.layer1_dim/2)
        
        self.fc_layer_neurons = 128

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer2_padding)
        self.conv2 = nn.Conv2d(self.layer1_filters, self.layer2_filters, self.layer2_kernel_size, stride=self.layer2_stride, padding=self.layer2_padding)

        self.lin1 = nn.Linear(self.layer2_filters * self.layer2_dim * self.layer2_dim, self.fc_layer_neurons)

        self.lin2 = nn.Linear(self.fc_layer_neurons, self.n_labels)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.lin2(x) ########
        return x

    def feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        return x

    def width(self, lower, upper):

            new_upper = upper
            new_lower = lower

            # Conv 1, ReLu
            bias = tc.repeat_interleave(self.conv1.bias.T, repeats=self.layer1_dim*self.layer1_dim, dim=0).reshape(self.layer1_filters, self.layer1_dim, self.layer1_dim)
            sum_upper = F.conv2d(new_upper, tc.relu(self.conv1.weight), stride=self.layer1_stride, padding=self.layer1_padding) + F.conv2d(new_lower, -tc.relu(-self.conv1.weight), stride=self.layer1_stride, padding=self.layer1_padding) + bias
            sum_lower = F.conv2d(new_lower, tc.relu(self.conv1.weight), stride=self.layer1_stride, padding=self.layer1_padding) + F.conv2d(new_upper, -tc.relu(-self.conv1.weight), stride=self.layer1_stride, padding=self.layer1_padding) + bias

            new_upper = tc.relu(sum_upper)
            new_lower = tc.relu(sum_lower)

            # # Conv2, ReLu
            bias = tc.repeat_interleave(self.conv2.bias.T, repeats=self.layer2_dim*self.layer2_dim, dim=0).reshape(self.layer2_filters, self.layer2_dim, self.layer2_dim)
            sum_upper = F.conv2d(new_upper, tc.relu(self.conv2.weight), stride=self.layer2_stride, padding=self.layer2_padding) + F.conv2d(new_lower, -tc.relu(-self.conv2.weight), stride=self.layer2_stride, padding=self.layer2_padding) + bias
            sum_lower = F.conv2d(new_lower, tc.relu(self.conv2.weight), stride=self.layer2_stride, padding=self.layer2_padding) + F.conv2d(new_upper, -tc.relu(-self.conv2.weight), stride=self.layer2_stride, padding=self.layer2_padding) + bias

            new_upper = tc.relu(sum_upper)
            new_lower = tc.relu(sum_lower)

            # Flatten
            new_upper = new_upper.view(new_upper.size(0), -1)
            new_lower = new_lower.view(new_lower.size(0), -1)

            # Linear 1, ReLU
            sum_upper = tc.mm(new_upper, tc.relu(self.lin1.weight.T)) + tc.mm(new_lower, -tc.relu(-self.lin1.weight.T)) + self.lin1.bias
            sum_lower = tc.mm(new_lower, tc.relu(self.lin1.weight.T)) + tc.mm(new_upper, -tc.relu(-self.lin1.weight.T)) + self.lin1.bias

            new_upper = tc.relu(sum_upper)
            new_lower = tc.relu(sum_lower)

            # Linear 2
            sum_upper = tc.mm(new_upper, tc.relu(self.lin2.weight.T)) + tc.mm(new_lower, -tc.relu(-self.lin2.weight.T)) + self.lin2.bias
            sum_lower = tc.mm(new_lower, tc.relu(self.lin2.weight.T)) + tc.mm(new_upper, -tc.relu(-self.lin2.weight.T)) + self.lin2.bias

            return sum_upper - sum_lower
