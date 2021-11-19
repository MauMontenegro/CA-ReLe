import gym

import carele.agents as agents
from carele.utils.utils import copyEnv


def rollout_train(env, agent, manager, config):
    episodes = config['train']['episodes']
    max_frames = config['train']['frames']
    total_games = config['train']['games']

    # Create Heuristic
    Heuristic = getattr(agents, config['heuristic']['class'])

    for game in range(0, total_games):

        for episode in range(0, episodes):
            obs = env.reset()

            # Get a Copy of Observation and environment to Rollout Functions
            rollout_obs = obs
            env_copy_dict = env.copy()
            rollout_env = gym.make(config['env']['name'])
            rollout_env = copyEnv(env_copy_dict, rollout_env)
            ###############################################################
            done = False
            trajectories = agent.create_trajectories()

            # while not done:
            agent.runSequential(trajectories, rollout_env, Heuristic, rollout_obs)
