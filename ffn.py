import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
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
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

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
