import numpy as np
import matplotlib.pyplot as plt
import time
from tensorforce.environments import Environment
from tensorforce.agents import Agent, TensorforceAgent


def set_env(env_name, max_episodes_timesteps):
    if env_name == 'CartPole-v1':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps,
                                 min_value=-5.0, max_value=5.0)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    elif env_name == 'MountainCar':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    elif env_name == 'Pendulum-v0':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    elif env_name == 'FrozenLake-v0':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    elif env_name == 'LunarLander-v2':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    elif env_name == 'Breakout-ram-v0' or env_name == 'Breakout-v0':
        env = Environment.create(environment='gym', level=env_name, max_episode_timesteps=max_episodes_timesteps)
        env_vis = Environment.create(environment='gym', level=env_name, visualize=True,
                                     max_episode_timesteps=max_episodes_timesteps)
    return env, env_vis


def set_agent(agent_name, env):
    if agent_name == 'tensorforce':
        agent = Agent.create(agent='tensorforce', environment=env, update=32,
                             optimizer=dict(optimizer='adam', learning_rate=1e-3),
                             objective='policy_gradient', reward_estimation=dict(horizon=10))
    elif agent_name == 'vpg':
        agent = Agent.create(agent='vpg', environment=env, batch_size=20,
                             baseline=dict(type='auto', size=32, depth=1),
                             baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10))
    elif agent_name == 'ppo':
        agent = Agent.create(agent='ppo', environment=env, batch_size=20)
        """
        agent = Agent.create(agent='ppo', environment=env, network='auto', batch_size=20,
                             learning_rate=1e-4, multi_step=10, subsampling_fraction=0.33,
                             likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
                             baseline=dict(type='auto', size=32, depth=1),
                             baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
                             l2_regularization=0.0, entropy_regularization=0.0,
                             state_preprocessing='linear_normalization', reward_preprocessing=None,
                             exploration=0.0, variable_noise=0.0, config=None)
        """
    elif agent_name == 'dpg':
        agent = Agent.create(agent='dpg', environment=env, memory=2000, batch_size=20)
    elif agent_name == 'dqn':
        agent = Agent.create(agent='dqn', environment=env, memory=50000, batch_size=1000, exploration=0.1)
    elif agent_name == 'ddqn':
        agent = Agent.create(agent='ddqn', environment=env, memory=2000, batch_size=20)
    elif agent_name == 'ac':
        agent = Agent.create(agent='ac', environment=env, batch_size=20,
                             critic=dict(type='auto', size=32, depth=1),
                             critic_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10))
    return agent


def train_agent(env, agent, nb_episodes, output_frequency, export_agent_name):
    episodes_returns = np.empty((nb_episodes))
    start = time.time()
    for ep_nb in range(0, nb_episodes):
        ep_return = 0.0
        states = env.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states, deterministic=False)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            ep_return += reward
        episodes_returns[ep_nb] = ep_return
        if (ep_nb + 1) % output_frequency == 0:
            mean_return = np.sum(episodes_returns[(ep_nb - output_frequency + 1):ep_nb + 1]) / output_frequency
            print('Episode nb {} to {}, mean return {}'.format(ep_nb + 1 - output_frequency, ep_nb, mean_return))
            save_agent(agent, export_agent_name + str(ep_nb))
    end = time.time()
    print('Training time {:1.0f}min {:1.0f}s'.format((end - start) // 60, (end - start) % 60))
    return agent, episodes_returns


def evaluate_agent(env, agent, nb_episodes):
    total_sum_rewards = 0.0
    for ep_nb in range(0, nb_episodes):
        ep_return = 0.0
        states = env.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
            states, terminal, reward = env.execute(actions=actions)
            total_sum_rewards += reward
            ep_return += reward
        print('Episode', ep_nb, 'return:', ep_return)
    print('Mean episode return:', total_sum_rewards / nb_episodes)
    return None


def plot_learning_curve(env_name, agent_name, agent_returns, export_name):
    num_episodes = agent_returns.size
    episode_arr = np.arange(1, num_episodes + 1)
    moving_avg = np.convolve(agent_returns, np.ones(50), 'same') / 50.0
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    ax.plot(episode_arr, agent_returns, label='return')
    ax.plot(episode_arr, moving_avg, label='moving_avg')
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Episode return')
    ax.set_title('Env: {}, Agent: {}'.format(env_name, agent_name))
    ax.legend()
    fig.tight_layout()
    export_path = 'Learning_Plots/' + export_name
    plt.savefig(export_path)
    print('Return plot saved in {}'.format(export_path))
    plt.show()
    return None


def save_agent(agent, export_name):
    agent.save('Trained_Agents', export_name, format='numpy')
    print('Model saved in {}'.format(export_name))
    return None


def load_agent(agent_name):
    agent = Agent.load('Trained_Agents', agent_name)
    return agent


def main_training(env_name, max_episodes_timesteps, agent_name, nb_train_episodes, output_freq, export_agent_name):
    env, env_vis = set_env(env_name, max_episodes_timesteps)
    agent = set_agent(agent_name, env)
    agent, returns = train_agent(env, agent, nb_train_episodes, output_freq, export_agent_name)
    plot_learning_curve(env_name, agent_name, returns, export_agent_name)
    save_agent(agent, export_agent_name)
    return None


def main_evaluating(env_name, max_episodes_timesteps, trained_agent_name, nb_evaluate_episodes, visualize):
    env, env_vis = set_env(env_name, max_episodes_timesteps)
    agent = load_agent(trained_agent_name)
    evaluate_agent(env, agent, nb_evaluate_episodes)
    if visualize:
        evaluate_agent(env_vis, agent, 10)
    return None


max_episodes_timesteps = 500
env_name = 'CartPole-v1'  # CartPole-v1, MountainCar, Pendulum-v0, FrozenLake-v0, LunarLander-v2, Breakout-ram-v0
agt_name = 'ppo'  # vpg, ppo, dpg, dqn, ddqn, ac
num_train_ep = 10000
freq = 200
param = ''
export_agt_name = env_name + '_' + agt_name + '_maxtimesteps' + str(max_episodes_timesteps) + '_nbepisodes' + str(
    num_train_ep) + '_' + param
main_training(env_name=env_name, max_episodes_timesteps=max_episodes_timesteps, agent_name=agt_name,
              nb_train_episodes=num_train_ep, output_freq=freq, export_agent_name=export_agt_name)

num_eval_ep = 50
vis = True
main_evaluating(env_name=env_name, max_episodes_timesteps=max_episodes_timesteps,
                trained_agent_name=export_agt_name, nb_evaluate_episodes=num_eval_ep, visualize=vis)
