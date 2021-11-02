import importlib
import sys

import matplotlib.pyplot as plt
import torch as T
from tqdm import tqdm

import config
import logger
import utils

# Path for Reinforcement Learning Agents and Neural Network Models
sys.path.append('./agents')
sys.path.append('./nets')

if __name__ == '__main__':
    # Get Initial Configuration
    config_file = config.config

    # Create Environment
    env = utils.create_environment(config_file)

    # Create Agent
    Agent = getattr(importlib.import_module("agents." + "DQN"), config_file.agent_name)
    agent = Agent(config_file)
    agent.update_target()

    # Create File Path (Experiments/Agent/Environment/File)
    file_path = 'Experiments' + '/' + str(config_file.agent_name) + '/' + "Bulldozer_v1/"
    chkpt_path = file_path + '/Check_points/' + config_file.net
    file = 'PER_DDQN_Dueling'

    # Create Logger to save results
    logger = logger.ExperimentLog(file_path, file)

    Rewards = []
    Sample_efficiency = []
    Losses = []
    frames = []

    # Play multiple fresh new games for statistics
    for game in range(1, config_file.total_games + 1):
        print("\n Playing Game: {}".format(game))

        # Reset Buffer and Agent
        agent.buffer.reset_buffer()
        agent.reset_agent()
        load_episode = 0

        # Check for previously saved models
        if config_file.load_model:
            load_episode = agent.load(chkpt_path)
            config_file.total_episodes = config_file.total_episodes - load_episode
            print("\nLoaded Parameters: {ep} Remaining Episodes ".format(ep=config_file.total_episodes))

        # Initialize Game Variables
        all_rewards = []  # Rewards for each episode
        all_losses = []  # Loss of each episode
        env_interactions = []  # Counts actions made in env for each episode (Accumulates over episodes)
        # Reset Flag Variables
        interaction = 0  # Reset accumulated env interactions
        update_flag = 0  # Reset sync between target and online net models
        save_flag = 0  # Reset save net model flag

        # Plays an entire episode until it ends or max frame config flag is reached
        for episode in tqdm(range(1, config_file.total_episodes + 1)):

            print("\nPlaying Episode: {}".format(episode))

            # Saving Net Model Parameters each N-steps
            if config_file.save_model and ((episode % config_file.save_model_n) == 0):
                agent.save(episode + load_episode, chkpt_path)

            # Reset Episode Variables
            state = env.reset()  # Send Env state to initial state
            done = False  # Reset done flag for env
            loss = 0
            episode_reward = 0
            frame = 0

            # Create a modified copy of env state
            state_c = utils.construct_state(state, env, config_file)

            # Play Episode Until Done or reach Max Frames Allowed
            while not done:
                if config_file.save_render and (game == config_file.total_games) and (
                        episode == config_file.total_episodes):
                    fig = env.render()
                    fig.savefig('Images/Emulation_{f}.png'.format(f=frame))
                    plt.close(fig)
                # Epsilon-Greedy strategy to take an action
                epsilon = utils.epsilon_by_frame(config_file, episode)

                # Getting Action
                action_buffer = agent.act(T.FloatTensor(state_c), epsilon, config_file.channels)  # To store in Buffer
                action_env = utils.mapping_actions(action_buffer)  # To make a step
                # Step in Environment
                next_state, reward, done, _ = env.step(action_env)

                # Update Variables
                interaction += 1
                frame += 1
                # Formatting next_state
                next_state_c = utils.construct_state(next_state, env, config_file)
                episode_reward += reward  # Accumulate reward

                # Store transition in Buffer
                agent.memorize(state_c, action_buffer, reward, next_state_c, done)

                # Update State
                state = next_state
                state_c = next_state_c

                # Wait to learn until buffer have at least batch size elements
                if agent.buffer.size() > config_file.batch_size:
                    loss = agent.learn(config_file)
                    update_flag += 1
                    all_losses.append(loss.item())

                # Copy parameters between models
                if update_flag % 100 == 0:
                    agent.update_target()
                    update_flag = 0

                if frame >= config_file.max_frames:
                    done = True

            # If episode ends due to failure or success, reset flags and store rewards and interactions.
            all_rewards.append(episode_reward)
            env_interactions.append(interaction)
            all_losses.append(loss)
    # Game Statistics

    logger.log_save(all_rewards, env_interactions)

    # Plotting Experiments
    utils.plot(file_path + file)

    env.close()
