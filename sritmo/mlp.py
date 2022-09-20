import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, final_act = 'relu'):
        super().__init__()
        self.final_act = final_act
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, inter=-1):
        shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        ret_inter = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == inter:
                ret_inter = x
        if self.final_act == 'relu':
            act = nn.ReLU()
        elif self.final_act == 'exp':
            act = torch.exp
        else:
            act = nn.Identity()
        return act(x.view(*shape, -1)), ret_inter
