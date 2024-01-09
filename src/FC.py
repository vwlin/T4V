import torch as tc
import torch.nn as nn

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = (num_hidden_layers-1) * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))

        for i in range(1,self.num_layers-1):
            if i % 2 == 1:
                self.layer_list.append(nn.ReLU())
            else:
                self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_layers):
            x = self.layer_list[i](x)

        return x

    def width(self, lower, upper):

            new_upper = upper
            new_lower = lower

            for i in range(self.num_layers-1):
                if i % 2 == 1: #skip relu layers
                    continue

                sum_upper = tc.mm(new_upper, tc.relu(self.layer_list[i].weight.T)) + tc.mm(new_lower, -tc.relu(-self.layer_list[i].weight.T)) + self.layer_list[i].bias

                sum_lower = tc.mm(new_lower, tc.relu(self.layer_list[i].weight.T)) + tc.mm(new_upper, -tc.relu(-self.layer_list[i].weight.T)) + self.layer_list[i].bias

                new_upper = tc.relu(sum_upper)
                new_lower = tc.relu(sum_lower)

            sum_upper = tc.mm(new_upper, tc.relu(self.layer_list[self.num_layers-1].weight.T)) + tc.mm(new_lower, -tc.relu(-self.layer_list[self.num_layers-1].weight.T)) + self.layer_list[self.num_layers-1].bias

            sum_lower = tc.mm(new_lower, tc.relu(self.layer_list[self.num_layers-1].weight.T)) + tc.mm(new_upper, -tc.relu(-self.layer_list[self.num_layers-1].weight.T)) + self.layer_list[self.num_layers-1].bias


            return sum_upper - sum_lower
