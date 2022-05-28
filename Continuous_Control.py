# %%
import a2c_agent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

# %% Start the environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# %% Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# %% Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# %% Create an agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {}
config["device"] = device
config["gamma"] = 0.95
config["use_gae"] = True
config["gae_tau"] = 0.5
config["learning_rate"] = 1e-4
config["grad_clip"] = 5

agent = a2c_agent.Agent(state_size, action_size, config, 1)

# %% Define the training process

def train_a2c(n_episode=300, rollout_length=6, goal=35):
    scores = []         # average score (over parallel workers) of each episode
    score_window = deque(maxlen=100)
    avg_scores_over_window = []

    for i_episode in range(1, n_episode+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        parallel_scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        rollout_experience = []

        # Keep interacting until some env has reached terminal state
        step_count = 0
        while True:
            step_count += 1
            experience_entry = {}

            actions, log_probs, values = agent.act(states, train_mode=True)         # select an action (for each agent)
            env_info = env.step(actions.detach().cpu().numpy())[brain_name]         # send all actions to tne environment
            next_states = env_info.vector_observations                              # get next state (for each agent)
            rewards = env_info.rewards                                              # get reward (for each agent)
            dones = env_info.local_done                                             # see if episode finished

            experience_entry["states"] = torch.from_numpy(states).to(device)
            experience_entry["values"] = values
            experience_entry["dones"] = torch.FloatTensor(dones).unsqueeze(-1).to(device)
            experience_entry["actions"] = actions
            experience_entry["log_probs"] = log_probs
            experience_entry["rewards"] = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
            rollout_experience.append(experience_entry)

            parallel_scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            if (step_count % rollout_length == 0):
                agent.train(rollout_experience)
                rollout_experience.clear()

            if np.any(dones):                                  # exit loop if episode finished
                break
        
        scores.append(parallel_scores.mean())
        score_window.append(parallel_scores.mean())
        print('\rTotal score (averaged over workers) for episode {}: {}'.format(i_episode, parallel_scores.mean()), end="")

        avg_score_over_window = np.mean(score_window)
        avg_scores_over_window.append(avg_score_over_window)
        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score_over_window))
        if avg_score_over_window >= goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, avg_score_over_window))
            torch.save({"state_dict": agent.network.state_dict()}, "result/checkpoint.pth")
            break
    
    return scores, avg_scores_over_window

# %% Train A2C agent
scores, average_scores = train_a2c()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='score')
plt.plot(np.arange(len(average_scores)), average_scores, label='avg score')
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('result/score.jpg')

# %% Watch trained agents
agent = a2c_agent.Agent(state_size, action_size, config, 1)
agent.network.load_state_dict(torch.load('result/checkpoint.pth')['state_dict'])

for i in range(3):
    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations 
    parallel_scores = np.zeros(num_agents)
    while True:
        actions, _, _ = agent.act(states)
        env_info = env.step(actions.detach().cpu().numpy())[brain_name]                  # send the action to the env
        next_states = env_info.vector_observations                                    # get the next state
        rewards = env_info.rewards                                                    # get the reward
        dones = env_info.local_done                                                   # see if episode has finished
        states = next_states
        parallel_scores += rewards
        if np.any(dones):
            break
    
    print('Average Score: {}'.format(parallel_scores.mean()))


# %%
env.close()