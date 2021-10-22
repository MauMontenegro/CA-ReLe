import numpy as np
import random as rnd
from collections import deque
from buffers.SumTree import SumTree


# Simple Buffer as Deque
class Simple_Buffer(object):
    def __init__(self, capacity):
        # Create a Deque
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        state, action, reward, next_state, done = experience
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*rnd.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def size(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()


# Prioritized Experience Replay Buffer as SumTree
class PERBuffer(object):
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []
        states = []
        actions = []
        rewards = []
        new_states = []
        dones = []
        priorities = []
        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Increment beta until max
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            states.append([data[0]])
            actions.append(data[1])
            rewards.append(data[2])
            new_states.append([data[3]])
            dones.append(data[4])
            b_idx[i] = index
            priorities.append(priority)

        states = np.concatenate(states)
        actions = list(actions)
        rewards = list(rewards)
        next_states = np.concatenate(new_states)
        dones = list(dones)

        real_batch = [states, actions, rewards, next_states, dones]

        sampling_probabilities = priorities / self.tree.total_priority
        IS_WEIGHT = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        IS_WEIGHT /= IS_WEIGHT.max()

        return b_idx, real_batch, IS_WEIGHT

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        b_size = self.tree.n_entries

        return b_size

    def reset_buffer(self):
        self.tree.reset()
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.
