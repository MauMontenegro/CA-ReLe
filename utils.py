import math

import gym
import matplotlib.pyplot as plt
import numpy as np


def create_environment(config):
    env = gym.make(config.env_name)
    if config.env_type == "Bulldozer":
        # Get The Action Space
        # 'nvec' is the action vector without subspace MultiDiscrete
        config.env_actions_n = env.action_space.nvec[0] * env.action_space.nvec[1]
        # Getting Grid Size
        obs = env.reset()
        # (rows x columns)
        h, w = obs[0].shape
        config.grid_h_size = h
        config.grid_w_size = w

    return env


def construct_state(observation):
    # Get Grid State
    state_grid = observation[0].copy()  # A copy of the Grid to not modify original
    # Get Bulldozer Position
    state_pos_x, state_pos_y = observation[1][1]
    # Adding Position to Grid
    state_grid[state_pos_x][state_pos_y] += 10
    return state_grid


# def save_frames_as_gif(frames, path='Recordings/', filename='gym_animation.gif'):

def epsilon_by_frame(config, frame):
    epsilon = config.final_epsilon + (config.initial_epsilon - config.final_epsilon) * math.exp(
        -1. * frame / config.decay_epsilon)

    return epsilon


def beta_by_frame(config, frame):
    beta = min(1.0, config.beta_start + ((frame * (1.0 - config.beta_start)) / config.beta_frames))
    return beta


def mapping_actions(action):
    # Available Actions
    actions = [
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]],
        [[2, 0], [2, 1]],
        [[3, 0], [3, 1]],
        [[4, 0], [4, 1]],
        [[5, 0], [5, 1]],
        [[6, 0], [6, 1]],
        [[7, 0], [7, 1]],
        [[8, 0], [8, 1]]
    ]

    # When Bulldozer Cuts Trees (9-17)
    if action > 8:
        Power = 1
        Movement = action - 9
    # When Bulldozer Only Moves (0-8)
    else:
        Power = 0
        Movement = action

    return np.array(actions[Movement][Power])


def plot(file_path):
    experiment_file = np.loadtxt(file_path, comments="#", delimiter=",", unpack=False)
    m_rewards = experiment_file[:, 0]
    m_samples = experiment_file[:, 1]

    x_range = range(0, len(m_rewards))
    plt.figure(figsize=(20, 5))
    plt.title("Average Reward")
    plt.ylabel('Mean Reward')
    plt.xlabel('Episode')
    plt.errorbar(x_range, m_rewards, color='gray', ecolor='lightgray', elinewidth=1, capsize=0)
    plt.show()
