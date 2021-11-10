import matplotlib.pyplot as plt
import torch as T
from tqdm import tqdm

from carele.utils.logger import ExperimentLog
from carele.utils.utils import construct_state, epsilon_by_frame, mapping_actions, plot


def dqn_train(env, agent, manager, config):
    Rewards = []
    Sample_efficiency = []
    Losses = []
    frames = []

    # Play multiple fresh new games for statistics
    for game in range(1, config['train']['games'] + 1):
        print("\n Playing Game: {}".format(game))

        # Reset Buffer and Agent
        agent.buffer.reset_buffer()
        agent.reset_agent()
        load_episode = 0
        path = str(manager.path)
        logger = ExperimentLog(path, 'results')

        # Load experiment
        # for previously saved models
        if config['manager']['load_model']:
            load_episode = agent.load(path + '/' + "chkpoint")
            config['train']['episodes'] = config['train']['episodes'] - load_episode
            print("\nLoaded Parameters: {ep} Remaining Episodes ".format(ep=config['train']['episodes']))

        # Initialize Game Variables
        all_rewards = []  # Rewards for each episode
        all_losses = []  # Loss of each episode
        env_interactions = []  # Counts actions made in env for each episode (Accumulates over episodes)
        # Reset Flag Variables
        interaction = 0  # Reset accumulated env interactions
        update_flag = 0  # Reset sync between target and online net models
        save_flag = 0  # Reset save net model flag

        # Plays an entire episode until it ends or max frame config flag is reached
        for episode in tqdm(range(1, config['train']['episodes'] + 1)):

            print("\nPlaying Episode: {}".format(episode))

            # Saving Net Model Parameters each N-steps
            if config['manager']['save_model'] and ((episode % config['manager']['save_model_freq']) == 0):
                agent.save(episode + load_episode, path + '/' + 'chkpoint')

            # Reset Episode Variables
            state = env.reset()  # Send Env state to initial state
            done = False  # Reset done flag for env
            loss = 0
            episode_reward = 0
            frame = 0

            # Create a modified copy of env state
            state_c = construct_state(state, env, config)

            # Play Episode Until Done or reach Max Frames Allowed
            while not done:
                if config['manager']['save_render'] and (game == config['train']['games']) and (
                        episode == config['train']['episodes']):
                    fig = env.render()
                    fig.savefig('Images/Emulation_{f}.png'.format(f=frame))
                    plt.close(fig)
                # Epsilon-Greedy strategy to take an action
                epsilon = epsilon_by_frame(config, episode)

                # Getting Action
                action_buffer = agent.act(T.FloatTensor(state_c), epsilon,
                                          config['agent']['channels'])  # To store in Buffer
                action_env = mapping_actions(action_buffer)  # To make a step
                # Step in Environment
                next_state, reward, done, _ = env.step(action_env)

                # Update Variables
                interaction += 1
                frame += 1
                # Formatting next_state
                next_state_c = construct_state(next_state, env, config)
                episode_reward += reward  # Accumulate reward

                # Store transition in Buffer
                agent.memorize(state_c, action_buffer, reward, next_state_c, done)

                # Update State
                state = next_state
                state_c = next_state_c

                # Wait to learn until buffer have at least batch size elements
                if agent.buffer.size() > config['agent']['mini_batch']:
                    loss = agent.learn(config)
                    update_flag += 1
                    all_losses.append(loss.item())

                # Copy parameters between models
                if update_flag % 100 == 0:
                    agent.update_target()
                    update_flag = 0

                if frame >= config['train']['frames']:
                    done = True

            # If episode ends due to failure or success, reset flags and store rewards and interactions.
            all_rewards.append(episode_reward)
            env_interactions.append(interaction)
            all_losses.append(loss)

    # Game Statistics
    logger.log_save(all_rewards, env_interactions)

    # Plotting Experiments
    plot(str(path) + '/' + 'results')

    env.close()
