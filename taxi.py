import gym

env = gym.make("Taxi-v3").env

state = env.encode(3, 1, 3, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print("State:", state)