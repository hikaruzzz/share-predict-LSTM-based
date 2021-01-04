import torch.nn as nn
import numpy as np
import torch

class GRUNet(nn.Module):

    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        return out

B = 3
time_step = 5
dim = 2
data = np.random.random([B, time_step, dim])
data = torch.from_numpy(data).float()

# input shape = [B, time_step, dim]
rnn = GRUNet(dim)
pred = rnn(data)
# output shape = [B,1]
print(pred.shape)

i = 0
n = np.random.random([15])
a = n[i:-(time_step - i)]
print(n)
print(a)