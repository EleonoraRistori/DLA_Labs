import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, path='./images/Cartpole_DQL/', filename='Cartpole_DQL.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=400)
    anim.save(path + filename, writer='imagemagick', fps=60)
def select_action(obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))


# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
def compute_returns(rewards, gamma):
    return np.flip(np.cumsum([gamma ** (i + 1) * r for (i, r) in enumerate(rewards)][::-1]), 0).copy()


# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=1000):  # esegue fino ad un max di 500 passi la policy nell'ambiente env
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


def reinforce(policy, env, env_render=None, gamma=0.99, eval_interval=10, eval_episodes=5, num_episodes=50,
              baseline=False, display=True):
    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    # Track episode rewards in a list.
    running_rewards = [0.0]
    average_rewards = []
    episode_lengths = []
    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Standardize returns.# Serve perchè in alcuni casi l'episodio termina dopo pochi passi (poca varianza) ed in altri è
        # molti più lungo
        returns = (returns - returns.mean()) / returns.std()

        if baseline:
            returns = (returns - returns.mean()) / returns.std()  # SBAGLIATOOOOOO

        # Make an optimization step
        opt.zero_grad()
        loss = (-log_probs * returns).sum()
        loss.backward()
        opt.step()

        # Render an episode after every 100 policy updates.
        if episode % eval_interval == 0:
            policy.eval()
            episode_length = 0
            total_reward = 0
            print(f'Episode {episode}')
            for _ in range(eval_episodes):
                (_, _, _, rewards) = run_episode(env, policy)
                total_reward += sum(rewards)
                episode_length += len(rewards)
            average_rewards.append(total_reward / eval_episodes)
            episode_lengths.append(episode_length / eval_episodes)
            print(
                f'Average Total: {total_reward / eval_episodes}, Average Episode Length: {episode_length / eval_episodes}')
            if display:
                (obs, _, _, _) = run_episode(env_render, policy)

            policy.train()
            print(f'Running reward: {running_rewards[-1]}')

    # Return the running rewards.
    policy.eval()
    return running_rewards, average_rewards, episode_lengths


class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 16)
        self.fc2 = nn.Linear(16, env.action_space.n)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env_render = gym.make('CartPole-v1', render_mode='human')

    wandb_use = False
    num_episodes = 1000
    eval_interval = 10
    eval_episodes = 5

    # Make a policy network.
    policy = PolicyNet(env)

    # Train the agent.
    running_rewards, avg_rewards, avg_episode_lengths = reinforce(policy, env, env_render,
                                                                  num_episodes=num_episodes,
                                                                  eval_interval=eval_interval,
                                                                  eval_episodes=eval_episodes)
    # Plot running rewards
    plt.figure()
    plt.plot(running_rewards)
    plt.title('Running rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./images/Cartpole_basicRL/Running_rewards_no_baseline.png')  # Save figure

    # Plot average rewards
    plt.figure()
    plt.plot(range(eval_interval, num_episodes + 1, eval_interval), avg_rewards)
    plt.title('Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./images/Cartpole_basicRL/Average_rewards_no_baseline.png')  # Save figure

    # Plot average episode lengths
    plt.figure()
    plt.plot(range(eval_interval, num_episodes + 1, eval_interval), avg_episode_lengths)
    plt.title('Average episode length')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.savefig('./images/Cartpole_basicRL/Average_episode_length_no_baseline.png')  # Save figure
    plt.show()
    # Close up everything
    env_render.close()
    env.close()

    env_render = gym.make('CartPole-v1', render_mode='rgb_array')
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

    save_frames_as_gif(frames, path='./images/Cartpole_basicRL/', filename='CartPole_basic_no_baseline.gif')

