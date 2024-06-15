import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib import animation
import wandb

def save_frames_as_gif(frames, path='./images/Cartpole_DQL/', filename='Cartpole_DQL.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=400)
    anim.save(path + filename, writer='imagemagick', fps=60)
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 16)
        self.fc2 = nn.Linear(16, env.action_space.n)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s


class ValueNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s


def select_action(obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.reshape(1)


def compute_returns(rewards, gamma=0.99):
    return np.flip(np.cumsum([gamma ** (i + 1) * r for (i, r) in enumerate(rewards)][::-1]), 0).copy()


def run_episode(env, policy_net, value_net, maxlen=500):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []

    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(obs, policy_net)
        value = value_net(obs)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return observations, actions, torch.cat(log_probs), rewards, values


def reinforce(policy_net, value_net, env, env_render=None, gamma=0.99, num_episodes=10, eval_interval=100, eval_episodes=10,
              standardize=True, wandb_use=False, display=False):
    opt = optim.Adam(policy_net.parameters(), lr=1e-2)
    opt_value = optim.Adam(value_net.parameters(), lr=1e-2)
    running_rewards = [0.0]

    average_rewards = []
    avg_episode_lengths = []

    if wandb_use:
        wandb.init(project="lab3 value policy reinforce")

    policy_net.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards, values) = run_episode(env, policy_net, value_net)
        values = torch.tensor(values, dtype=torch.float32)
        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        if standardize:
            returns = (returns - returns.mean()) / returns.std()

        targets = returns - values

        opt.zero_grad()

        loss = (-log_probs * targets).mean()
        opt_value.zero_grad()
        loss_value = F.mse_loss(values, returns)
        loss_value.requires_grad = True
        loss_value.backward()
        opt_value.step()
        loss += loss_value
        loss.backward()
        opt.step()

        if episode % eval_interval == 0:
            policy_net.eval()
            episode_length = 0
            total_reward = 0
            for _ in range(eval_episodes):
                (_, _, _, rewards, _) = run_episode(env, policy_net, value_net)
                total_reward += sum(rewards)
                episode_length += len(rewards)
            average_rewards.append(total_reward / eval_episodes)
            avg_episode_lengths.append(episode_length / eval_episodes)
            print(f'Episode {episode} Average Reward: {total_reward / eval_episodes}, Average Episode Length: {episode_length / eval_episodes}')
            if display:
                (obs, _, _, _, _) = run_episode(env_render, policy)
            if wandb_use:
                wandb.log({"avg_reward": np.mean(rewards), "avg_episode_length": np.mean(avg_episode_lengths)})
            policy_net.train()
            value_net.train()
    policy_net.eval()
    value_net.eval()
    return running_rewards, average_rewards, avg_episode_lengths


if __name__ == "__main__":
    # Instantiate a rendering and a non-rendering environment.
    env_render = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')

    # Print the observation space and action space of the environment.
    (obs, info) = env.reset()

    policy = PolicyNet(env)
    value_net = ValueNet(env)

    # Setting parameters
    wandb_use = False
    standardize = True
    num_episodes = 700
    eval_interval = 100
    eval_episodes = 15
    running_rewards, avg_rewards, avg_episode_lengths = reinforce(policy, value_net, env, env_render,
                                                                  num_episodes=num_episodes,
                                                                  eval_interval=eval_interval,
                                                                  standardize=standardize,
                                                                  eval_episodes=eval_episodes, wandb_use=wandb_use)

    # Plot running rewards
    plt.figure()
    plt.plot(running_rewards)
    plt.title('Running rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.savefig('./images/Cartpole_basicRL/Running_rewards.png')  # Save the figure

    # Plot average rewards
    plt.figure()
    plt.plot(range(eval_interval, num_episodes + 1, eval_interval), avg_rewards)
    plt.title('Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.savefig('./images/Cartpole_basicRL/Average_rewards.png')  # Save the figure

    # Plot average episode lengths
    plt.figure()
    plt.plot(range(eval_interval, num_episodes + 1, eval_interval), avg_episode_lengths)
    plt.title('Average episode length')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    # plt.savefig('./images/Cartpole_basicRL/Average_episode_length.png')  # Save the figure
    plt.show()
    if wandb_use:
        wandb.finish()


    env_render = gym.make('CartPole-v1', render_mode='rgb_array')

    # Run the env
    state, info = env_render.reset()
    frames = []
    for t in range(800):
        # Render to frames buffer
        frames.append(env_render.render())
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        (action, log_prob) = select_action(state, policy)
        state, _, terminated, truncated, _ = env_render.step(action)
        if terminated or truncated:
            break
    env_render.close()
    if standardize:
        save_frames_as_gif(frames, path='./images/Cartpole_basicRL/', filename='CartPole_basic.gif')
    else:
        save_frames_as_gif(frames, path='./images/Cartpole_basicRL/', filename='CartPole_basic_nostd.gif')
