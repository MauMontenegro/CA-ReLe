import math
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml


def create_environment(env_name):
    env = gym.make(env_name)
    return env


def copyEnv(dict, env):
    env.steps_elapsed = dict['steps_elapsed']
    env.steps_beyond_done = dict['steps_beyond_done']
    env.reward_accumulated = dict['reward_accumulated']
    # env._resample_initial = dict['_resample_initial']
    env.done = dict['done']
    env.context = dict['context']
    env.grid = dict['grid']
    env._fire_seed = dict['_fire_seed']

    return env


def construct_state(observation, env, config):
    state_grid, context = observation
    wind_params, (posRow, posCol), time = context
    fire, tree, empty, burned = env._fire, env._tree, env._empty, env._burned

    if config['agent']['channels']:
        return imageGridC(state_grid, fire, tree, empty, burned, posRow, posCol, config)

    return imageGrid(state_grid, fire, tree, empty, burned, posRow, posCol, config)


TREE, EMPTY, FIRE, BURNED, AGENT = 100, 0, 200, 50, 255


def imageGrid(grid: np.array,
              fire, tree, empty, burned,
              agentRow, agentCol, config) -> np.array:
    # Construct the grid in grayscale
    img = np.zeros((config['env']['grid_h'], config['env']['grid_w']), dtype=np.uint8)

    for row in range(0, config.grid_h_size - 1):
        for col in range(1, config.grid_w_size):
            currentCell, val = grid[row, col], 0
            if currentCell == fire:
                val = FIRE
            elif currentCell == tree:
                val = TREE
            elif currentCell == burned:
                val = BURNED
            else:
                val = EMPTY
            if (row == agentRow) and (col == agentCol):
                val = AGENT
            img[row, col] = val

    return img


def imageGridC(grid: np.array,
               fire, tree, empty, burned,
               agentRow, agentCol, config) -> np.array:
    # Construct the grid in grayscale
    img = np.zeros((4, config['env']['grid_h'], config['env']['grid_w']), dtype=np.bool_)

    for row in range(0, config['env']['grid_h'] - 1):
        for col in range(1, config['env']['grid_w']):
            currentCell = grid[row, col]
            if currentCell == fire:
                img[0, row, col] = 1
            elif currentCell == tree:
                img[1, row, col] = 1
            elif currentCell == burned:
                img[2, row, col] = 1
            if (row == agentRow) and (col == agentCol):
                img[3, row, col] = 1
    return img


# def save_frames_as_gif(frames, path='Recordings/', filename='gym_animation.gif'):

def epsilon_by_frame(config, frame):
    epsilon = config['policy']['final_epsilon'] + (
                config['policy']['initial_epsilon'] - config['policy']['final_epsilon']) * math.exp(
        -1. * frame / config['policy']['decay_epsilon'])

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


def getExpConfig(name, defpath=None):
    if defpath is None:
        path = Path.cwd() / 'exp_config'
    else:
        path = Path(defpath)
    pathFile = path / (name.strip() + '.yaml')

    if not pathFile.exists() or not pathFile.is_file():
        raise ValueError('Path either does not exists or is not a File')

    config = yaml.safe_load(pathFile.open('r'))

    return config

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
