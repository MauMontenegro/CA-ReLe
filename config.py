"""
Configuration File for:
        -Environment
        -Network
        -Agent
        -Training Parameters
"""
import argparse

parser = argparse.ArgumentParser()

# Environment Parameters
parser.add_argument("-env_type", default="Bulldozer")  # Type of environment (As some CA could need special properties)
parser.add_argument("-env_name", default="gym_cellular_automata:ForestFireBulldozer256x256-v2")  # Official Env name

# Agent Parameters
parser.add_argument("-agent_name", default="DQN")
parser.add_argument("-replay_memory_capacity", type=int, default=100000)  # Buffer Capacity
parser.add_argument("-buffer_name", default="PERBuffer")  # Type of Buffer Needed
parser.add_argument("-buffer_alpha", type=float, default=0.6)  # Alpha param for PER
parser.add_argument("-beta_start", type=int, default=0.4)  # Beta param for PER
parser.add_argument("-beta_frames", type=int, default=1000)  # Beta param for PER
parser.add_argument("-gamma", type=float, default=0.99)  # Importance of future rewards
parser.add_argument("-initial_epsilon", type=float, default=0.99)  # Epsilon Greedy Strategy
parser.add_argument("-final_epsilon", type=float, default=0.01)
parser.add_argument("-decay_epsilon", type=int, default=100)

# Network Parameters
parser.add_argument("-net", default="Dueling_CNN")  # Network Model
parser.add_argument("-batch_size", type=int, default=32)  # Batch Size
parser.add_argument("-learning_rate", type=float, default=0.00001)  # Learning Step
parser.add_argument("-optimizer_name", default="Adam")  # Optimizer

# Training Experiment Parameters
parser.add_argument("-total_games", type=int, default=1)  # Total numbers of Games
parser.add_argument("-total_episodes", type=int, default=1000)  # Total Episodes in a Game
parser.add_argument("-max_frames", type=int, default=800)  # Max number of frames in Episode

# Another Configuration (Logging, render, seed, etc)
parser.add_argument("-seed", type=int, default=10)
parser.add_argument("-save_model", type=bool, default=True)
parser.add_argument("-load_model", type=bool, default=False)
parser.add_argument("-save_model_n", type=int, default=20)  # Save model parameters every n steps in environment
parser.add_argument("-save_render", type=bool, default=True)
parser.add_argument("-logging", default="")
config = parser.parse_args()
config.logging = config.logging not in ["0", "false", "False"]
