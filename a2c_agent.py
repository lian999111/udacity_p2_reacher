import numpy as np
from model import GaussianActorCriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            config (dict): configuration of learning parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.network = GaussianActorCriticNetwork(state_size, action_size, seed).to(config["device"])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config["learning_rate"])
    
    def train(self, rollout_experience: list):
        """Run one training step of the agent.
        
        Params
        ======
            rollout_experience (list): list containing the interaction experience within rollout period.
                                       Each entry is a dict containing: states, values, dones, actions, log_probs, rewards.
                                       For the last entry, only states, values and dones exist.
                                       Note that each term contains results from parallel envs  
        """
        config = self.config

        # The number of states at a time step == the number of workers
        num_workers = len(rollout_experience[0]["states"])
        advantages = torch.FloatTensor(np.zeros((num_workers, 1))).to(config["device"])
        returns = rollout_experience[-1]["values"] # init returns using last values in rollout

        for i in reversed(range(len(rollout_experience)-1)):
            rewards = rollout_experience[i]["rewards"]
            dones = rollout_experience[i]["dones"]
            values = rollout_experience[i]["values"]
            next_values = rollout_experience[i+1]["values"]

            returns = rewards + config["gamma"] * (1 - dones) * returns
            if not config["use_gae"]:
                advantages = returns - values
            else:
                # Ref: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
                td_error = (rewards + config["gamma"] * (1 - dones) * next_values - values).to(config["device"])
                advantages = advantages * config["gae_tau"] * config["gamma"] * (1 - dones) + td_error

            # Add resultant returns and advantages to rollout_experience
            # Note that each entry has estimates of returns and advantages using different step bootstraping
            rollout_experience[i]["returns"] = returns.detach()
            rollout_experience[i]["advantages"] = advantages.detach()

        extracted_data = [[entry["log_probs"], entry["advantages"], entry["returns"], entry["values"]] for entry in rollout_experience[:-1]]
        log_probs, advantages, returns, values = map(lambda x: torch.cat(x, dim=0).to(config["device"]), zip(*extracted_data))

        policy_loss = -(log_probs * advantages)
        critic_loss = 0.5 * (returns - values).pow(2)

        self.optimizer.zero_grad()
        (policy_loss + critic_loss).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), config["grad_clip"])
        self.optimizer.step()

    def act(self, state, train_mode=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            train_mode (bool): run actor net in train mode or not
        """
        state = torch.from_numpy(state).float().to(self.config["device"])
        if not train_mode:
            self.network.eval()

        with torch.set_grad_enabled(train_mode):
            action, log_prob, value = self.network(state)

        if not train_mode:
            self.network.train()

        return action, log_prob, value