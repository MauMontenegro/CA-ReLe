import gym
import matplotlib.pyplot as plt
import gym_cellular_automata as gymca

env = gym.make("gym_cellular_automata:ForestFireBulldozer256x256-v2")
obs = env.reset()
# Print available CA envs
print(gymca.REGISTERED_CA_ENVS)

total_reward = 0.0
done = False
step = 0
threshold = 24

# Random Policy for at most "threshold" steps
while not done and step < threshold:
    fig = env.render()
    fig.savefig('Images/Emulation_{f}.png'.format(f=step))
    plt.close(fig)
    action = env.action_space.sample()  # Your agent goes here!
    print(action)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")