import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from shrank import LED


class SimpleGLU(nn.Module):

    def __init__(self, d1, d2, bias=False):
        super().__init__()

        self.l1 = LED(d1, d2, bias=bias)
        self.l2 = LED(d1, d2, bias=bias)
        self.l3 = LED(d2, d1, bias=bias)

    def forward(self, x):
        o1 = self.l1(x)
        o2 = self.l2(x)
        output = o1 * o2
        output = self.l3(output)

        return output


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        r = config.r
        self.dense_h_to_4h = LED(hidden_size, intermediate_size, r)
        self.dense_4h_to_h = LED(intermediate_size, hidden_size, r)
        self.act = ACT2FN[config.pythia_hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class RwkvFeedForward(nn.Module):
    def __init__(self, config, layer_id=0, expand=False):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        r = config.r
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = LED(hidden_size, intermediate_size, r, bias=False)
        self.receptance = LED(hidden_size, hidden_size, r, bias=False)
        self.value = LED(intermediate_size, hidden_size, r, bias=False)

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (
            1 - self.time_mix_receptance
        )

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


class Rwkv6FeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        # https://github.com/BlinkDL/RWKV-LM/blob/3db37a72356b736966ddd377268f02b80963af3f/RWKV-v4neo/train.py#L168
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int((config.hidden_size * 3.5) // 32 * 32)
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = LED(hidden_size, intermediate_size, bias=False)
        self.receptance = LED(hidden_size, hidden_size, bias=False)
        self.value = LED(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[2][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[2][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        delta_hidden_to_shifted = shifted - hidden
        key = hidden + delta_hidden_to_shifted * self.time_maa_k
        receptance = hidden + delta_hidden_to_shifted * self.time_maa_r

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[2][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state
