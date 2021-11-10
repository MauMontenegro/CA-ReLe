import numpy as np


class SumTree(object):
    # Pointer to current empty leave
    data_pointer = 0

    def __init__(self, capacity):
        # Number of Leaf nodes that contains experiences (Buffer capacity)
        self.capacity = capacity
        # Construct Tree based on Binary Tree properties and capacity
        self.tree = np.zeros(2 * capacity - 1)
        # Experiences
        self.data = np.zeros(capacity, dtype=object)
        # Actual experiences stored
        self.n_entries = 0

    def add(self, priority, data):
        # As we only need to save priority in leaves
        tree_index = self.data_pointer + self.capacity - 1

        # Point to new space in Experiences array and save new experience (data)
        self.data[self.data_pointer] = data

        # Update the priority  on the tree leaves
        self.update(tree_index, priority)

        # Increment pointer to new space in buffer
        self.data_pointer += 1

        # If Buffer is full, then reset pointer and overwrite first experiences
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        # Difference between new priority and stored priority
        change = priority - self.tree[tree_index]

        # Replace priority
        self.tree[tree_index] = priority

        # Propagate upstream the tree from leave
        # As we are in Binary Tree we can divide by 2 to get parent node index until root (idx = 0)
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1  # Practically move to next level by mult *2
            right_child_index = left_child_index + 1  # As we reach next lvl, and children are neighbors only + 1

            # Until reach a leaf
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # Downstream search. Searching for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def reset(self):
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.n_entries = 0
        self.data_pointer = 0

    @property
    def total_priority(self):
        return self.tree[0]
