# Import the gym module
import gym
import time
import q_iteration
import atari_model
# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())

  q_iteration.q_iteration(env,model,state,iteration,memory)
  # Render
  env.render()
  time.sleep(0.1)
  env.close()