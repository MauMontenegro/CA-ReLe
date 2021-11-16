import numpy as np

from carele.utils.utils import plot_learning_curve


def ppo_algorithm(env, agent, manager, config):
    env = env
    N = config['agent']['trajectory']  # Size of trajectory
    batch_size = config['agent']['batch_size']
    n_epochs = config['train']['epochs']
    alpha = config['policy']['nets']['alpha']
    agent = agent

    n_games = config['train']['episodes']

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):  # Number of Episodes to Play
        observation = env.reset()
        done = False
        score = 0
        # Play until get a done state
        #########################################
        while not done:
            # Use the critic Network to get  the action, value and his probability
            action, prob, val = agent.choose_action(observation)
            # Step in environment
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            # Accumulated reward
            score += reward
            # Store in Memory the transition
            agent.remember(observation, action, prob, val, reward, done)
            # When trajectory is of size N learn and erase memory
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        #########################################
        score_history.append(score)
        # Average score of the last 100 episodes
        avg_score = np.mean(score_history[-100:])

        # Saving the model when reaching a new best score
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_stes', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
