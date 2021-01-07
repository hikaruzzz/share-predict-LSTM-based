import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_size, out_channel=1, predict_times_step=1):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.predict_times_step = predict_times_step
        if predict_times_step == 1:
            self.out1 = nn.Sequential(
                nn.Linear(128, out_channel)
            ) # output future_time = 1
        else:
            self.out = nn.Conv1d(128, out_channel, 1) # output future_time = time_step

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # print(r_out.shape) # [B, time_step, dim]
        if self.predict_times_step == 1:
            out = self.out1(r_out[:, -1])
        else:
            r_out = r_out.permute([0,2,1])
            # print(self.out(r_out).permute([0,2,1]).shape)
            out = self.out(r_out).permute([0,2,1])[:,:self.predict_times_step,:]
        return out

