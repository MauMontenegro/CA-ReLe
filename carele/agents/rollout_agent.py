from matplotlib import pyplot as plt

from carele.utils.constants import *


def expandLeafs(leafs, action_set):
    new_leafs = list()
    for leaf in leafs:
        for action in action_set:
            new_leaf = leaf.copy()
            new_leaf.append(action)
            new_leafs.append(new_leaf)
    return new_leafs


class rolloutAgent():
    def __init__(self, config):
        self.lookahead = config['agent']['lookahead']
        self.vision = config['heuristic']['vision']
        self.actions = config['env']['action_space']

    # TODO : Only take valid Movements according to position
    def create_trajectories(self):
        trajectories = list(list([a]) for a in ACTION_SET)
        for i in range(1, self.lookahead):
            trajectories = expandLeafs(trajectories, ACTION_SET)
        return trajectories

    def runSequential(self, trajectories, env, heuristic, observation):
        # First iterate over trajectories
        print('Enter')
        c = 0
        for trajectory in trajectories:
            print('Trajectory {}'.format(trajectory))
            # Traverse trajectory according to Lookahead
            for step in trajectory:
                print('Action! {}'.format(step))
                c += 1
                fig = env.render()
                fig.savefig('Images/Emulation_{f}.png'.format(f=c))
                plt.close(fig)
                env.step(step)

        # Apply Heuristic
        # h_action = heuristic({"env": env, "observation": observation})
