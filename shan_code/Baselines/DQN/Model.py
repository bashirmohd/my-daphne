import torch
from torch import nn


def linear_init(module):
    if isinstance(module, nn.Linear):
        # nn.init.zeros_(module.weight)
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, hidden_layers, device='cpu'):
        super(model, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.append(hidden_size)
        layers = [linear_init(nn.Linear(self.input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], self.output_size)))
        self.mean = nn.Sequential(*layers)

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state).float()
        state = state.to(self.device, non_blocking=True)
        q_values = self.mean(state)

        return q_values
