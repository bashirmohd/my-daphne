
import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, input_shape, output_shape, hid_shape, hid_num, activation='relu'):
        super().__init__()
        self.input_shape = input_shape 
        self.output_shape = output_shape
        self.hid_shape = hid_shape
        self.hid_num = hid_num
        if activation == 'tanh':
            self.activation = [nn.Tanh, torch.tanh]
        elif activation == 'relu':
            self.activation = [nn.ReLU, torch.relu]
        else:
            raise Exception('unsupported activation type')

        layers = [nn.Linear(self.input_shape, self.hid_shape), self.activation[0]()]
        for _ in range(self.hid_num):
            layers.extend([nn.Linear(self.hid_shape, self.hid_shape), self.activation[0]()])
        layers.append(nn.Linear(self.hid_shape, self.output_shape))
        # layers.append(nn.Softmax())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)
