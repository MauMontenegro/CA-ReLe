import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        # Arange with initial index of every batch
        # For example:
        #   from 0 to n_states=20 with interval of batch size=5
        #       [0,5,10,15]
        batch_start = np.arange(0, n_states, self.batch_size)
        # All indices
        #       [0,1,2,...,19]
        indices = np.arange(n_states, dtype=np.int64)
        # Shuffle index order inside array
        np.random.shuffle(indices)
        # Create Batches. How many? N trajectory / batch size
        # Batch start array gives the initial index for the batch and batch size the sizeof it
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
