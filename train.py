import argparse
import sys

from carele import getExpConfig, setupExperiment


def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        List of Available Algorithms:
            - DQN (Available)
            - A2C (Not Implemented yet)
            - PPO (In progress)
            - TRPO (Not Implemented yet)
            ''',
        epilog='''python train.py -a dqn -ec dqn_bulldozer'''
    )

    parser.add_argument(
        '--alg', '-a', type=str,
        help="Name of the RL algorithm to train.")
    parser.add_argument(
        '--config', '-ec', type=str,
        help="Configuration yaml file.")
    parser.add_argument(
        '--cuda', type=bool, default=True,
        help='Enables the use of CUDA devices'
    )

    return parser.parse_known_args(args)[0]


if __name__ == '__main__':
    args = argParser(sys.argv[:])
    # Retrieve Experiment Configuration
    exp_config = getExpConfig(args.config)
    env, agent, manager, train = setupExperiment(args.alg, exp_config, cuda=args.cuda)
    train(env, agent, manager, exp_config)
