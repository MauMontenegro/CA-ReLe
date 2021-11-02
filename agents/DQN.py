# DQN Agent for Cellular Automata

import importlib
import random as rnd

import numpy as np
import torch
import torch as T


class DQN:
    def __init__(self, config):
        # rnd.seed(config.seed)
        self.env_type = config.env_type
        self.buffer_name = config.buffer_name
        self.action_space_num = config.env_actions_n

        # Create Buffer
        self.Buffer = getattr(importlib.import_module("buffers." + "BUFFERS"), config.buffer_name)
        self.buffer = self.Buffer(config.replay_memory_capacity)

        # Create Model for Learning
        Net = getattr(importlib.import_module("models." + "CNN"), config.net)
        self.current_model = Net(config)
        self.target_model = Net(config)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.current_model.to(self.device)
        self.target_model.to(self.device)
        self.optimizer = T.optim.Adam(self.current_model.parameters(), lr=config.learning_rate)

    def act(self, state, epsilon, channels):
        if rnd.random() > epsilon:
            if self.env_type == "Bulldozer":
                if channels:
                    # Formatting state to have 1 element of batch
                    state = state.unsqueeze(0)
                else:
                    # Formatting state to have 1 channel
                    state = state.unsqueeze(0)
                    # Formatting state to have 1 element of batch
                    state = state.unsqueeze(1)
                # For inference calculate actual q_value
                with torch.no_grad():
                    q_value = self.current_model.forward(state.to(self.device))
            # Getting the index of q_values that has max value
            action = T.argmax(q_value).item()
        else:
            action = rnd.randrange(self.action_space_num)
        return action

    def learn(self, config):
        if config.buffer_name == 'PERBuffer':
            tree_index, minibatch, weights = self.buffer.sample(config.batch_size)
            tree_index = T.LongTensor(tree_index).to(self.device)
            weights = T.FloatTensor(weights).to(self.device)
        elif config.buffer_name == 'Simple_Buffer':
            minibatch = self.buffer.sample(config.batch_size)

        state, action, reward, next_state, done = minibatch

        state = T.FloatTensor(np.float32(state)).to(self.device)
        next_state = T.FloatTensor(np.float32(next_state)).to(self.device)
        action = T.LongTensor(action).to(self.device)
        reward = T.FloatTensor(reward).to(self.device)
        done = T.FloatTensor(done).to(self.device)

        # Calculate Q_Values (Prediction) with Online Network
        q_values = self.current_model(state)

        # Calculate q values of next state with online network (To evaluate greedy Policy)(arg max)
        next_q_values = self.current_model(next_state).detach()

        # Estimate value with target network (Evaluation)
        next_q_state_values = self.target_model(next_state)

        # Evaluating next state with the arg max from target network
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        expected_q_value = reward + config.gamma * next_q_value * (1 - done)

        if config.buffer_name == 'PERBuffer':
            loss = (q_value - expected_q_value).pow(2) * weights
            priors = loss  # absolute errors (priorities) for PER
            self.buffer.batch_update(tree_index.data.cpu().numpy(), priors.data.cpu().numpy())
        elif config.buffer_name == 'Simple_Buffer':
            loss = (q_value - expected_q_value).pow(2)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def memorize(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.buffer.store(experience)

    def update_target(self):
        # print("Updating Parameters")
        self.target_model.load_state_dict(self.current_model.state_dict())

    def reset_agent(self):
        # Load Initial parameters
        self.current_model.reset_parameters()
        # Synchronize with Target Model
        self.update_target()
        # Initialize Optimizer with initial Parameters
        self.optimizer = T.optim.Adam(self.current_model.parameters())

    def save(self, episode, path):
        print("\n...Saving Checkpoint in ", path)
        T.save({
            'episode': episode,
            'current_model_state_dict': self.current_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path):
        print("\n...Loading Checkpoint from ", path)
        check_point = T.load(path)
        self.current_model.load_state_dict(check_point['current_model_state_dict'])
        self.target_model.load_state_dict(check_point['target_model_state_dict'])
        self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
        episode = check_point['episode']

        return episode
