import torch
from torch import nn
from name_dataset import N_LETTERS

class RNN(nn.Module):
    def __init__(self, output_size):
        super(RNN, self).__init__()
        input_size = N_LETTERS
        self.hidden_size = 128
        self.i2h = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(input_size + self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def process_name(self, name, init_hidden):
        output = None
        hidden = init_hidden
        for i in range(name.shape[1]):
            output, hidden = self(name[:,i], hidden)
        return output, hidden
