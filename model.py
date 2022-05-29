import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """General MLP"""

    def __init__(self, in_size: int, out_size: int, hidden_sizes: list, seed: int=0, out_gate=None):
        """Initialize MLP with given hidden layer sizes
        Params
        ======
            in_size (int): Dimension of input
            out_size (int): Dimension of output
            hidden_sizes (list): List of hidden layer sizes
            seed (int): Random seed
            out_gate: Output gate
        """
        super(Network, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.out_gate = out_gate
        self.linear_layers = nn.ModuleList()

        last_size = in_size
        for size in hidden_sizes:
            self.linear_layers.append(nn.Linear(last_size, size))
            last_size = size

        self.linear_layers.append(nn.Linear(last_size, out_size))

    
    def forward(self, input: torch.Tensor):
        """Forward pass of the network"""
        x = input
        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))
        
        # Last layer
        x = self.linear_layers[-1](x)

        # Apply output gate if exist
        if self.out_gate:
            x = self.out_gate(x)

        return x

class GaussianActorCriticNetwork(nn.Module):
    """Actor critic network with Gaussian sampling on the action implementation"""

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize actor critic network
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(GaussianActorCriticNetwork, self).__init__()

        self.actor_net = Network(state_size, action_size, [128, 128, 128], out_gate=F.tanh, seed=seed)
        self.critic_net = Network(state_size, 1, [128, 128, 128], seed=seed)

        # Ref: https://discuss.pytorch.org/t/understanding-enropy/34677
        self.std = nn.Parameter(torch.ones(1, action_size))    # variance will be learned

    def forward(self, state: torch.Tensor):
        """Forward pass
        Params
        ======
            state: current state
        """
        action_mean = self.actor_net(state)
        dist = torch.distributions.Normal(action_mean, F.relu(self.std))
        action = torch.clamp(dist.rsample(), min=-1, max=1)
        log_prob = dist.log_prob(action)
        # entropy can be added
        # Ref: https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_heads.py

        value = self.critic_net(state)

        return action, log_prob, value





