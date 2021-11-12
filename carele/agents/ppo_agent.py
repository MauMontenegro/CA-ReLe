import numpy as np
import ppo.models as models
import ppo.storage as memory
import torch as T


class ppoAgent:
    def __init__(self, config):
        self.gamma = config['agent']['gamma']
        self.policy_clip = config['agent']['clip']
        self.n_epochs = config['algorithm']['epochs']
        self.gae_lambda = config['agent']['gae_lambda']

        self.Memory = getattr(memory, config['agent']['memory'])
        self.memory = self.Memory(config['agent']['batch_size'])

        ActNet = getattr(models, config['policy']['nets']['ActorClass'])
        CritNet = getattr(models, config['policy']['nets']['CriticClass'])
        self.actor = ActNet(config['env']['action_space'], config['env']['input_size'],
                            config['policy']['nets']['alpha'])
        self.critic = CritNet(config['env']['input_size'], config['policy']['nets']['alpha'])

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('...Saving Models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('...Loading Models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # Convert observation to Tensor type and send to actor device
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        # Get Categorical distribution probabilities of each action in environment
        dist = self.actor(state)
        # Get state-value for current state in environment via critic network
        value = self.critic(state)
        # Get an action from the distribution probabilities
        action = dist.sample()
        # Get the Logit prob of the action selected by the sample
        probs = T.squeeze(dist.log_prob(action)).item()
        # Get the actual action selected
        action = T.squeeze(action).item()
        # Get the actual Value
        value = T.squeeze(value).item()

        return action, probs, value  # Return action, logit and value

    def learn(self):
        # Training with the same Batch of trajectories n_epochs
        for _ in range(self.n_epochs):
            # Generate Random Batches from memory
            state_arr, action_arr, old_probs_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            # Construct the advantages array
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Generalized Advantage Estimator GAE
            # sum(lamda * gamma)^l * delta
            # delta = r_t + gamma * V(s_t+1) - Baseline (V(s_t))
            # Travel trough reward array [First_r_in_T, Second....,..., Last_r_in_T]
            for t in range(len(reward_arr) - 1):
                # Start with discount factor of 1 (means first reward does not have discount)
                discount = 1
                a_t = 0
                # Traverse
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # Traverse random index batches in batches (N/batch.size)
            for batch in batches:
                # Select elements in arrays according to indexes shuffles in batches
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                # Old Probability Distribution pi_(theta)_old
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # Get new Distribution from actor network
                dist = self.actor(states)
                # Get new values from critic network
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                # Calcule ratio or distance between polcies (new/old)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # rt(theta) * A_pi_theta
                weighted_probs = advantage[batch] * prob_ratio
                # clip(rt(theta), 1 - epsilon, 1 + epsilon,) * Advantage
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                # Min between dist ratio and clipped version
                # As we are using Gradient Acent, need to multiply by negative
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Convert to returns by using A= R_t - V_t
                returns = advantage[batch] + values[batch]
                # Claculate Critic Loss
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Total Loss without entropy term
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
