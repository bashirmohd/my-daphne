import gym
import gym_sdwan

MAX_RUNS = 60
total_reward = 0

env = gym.make('Sdwan-v0')

observation = env.reset()

print('Deeproutes Initial State:', observation)

error = False
for t in range (MAX_RUNS):
	action = env.action_space.sample()
	observation, reward, error, info = env.step(action)
	total_reward += reward
	print('Runs:', t+1, 'Action:', action, 'Ob:', observation, 'R:', 
			reward, 'Total Reward:', total_reward)

	if error:
		print("Episode Aborted  after {} timesteps".format(t+1))
		break

if not error:
	print("Episode Finished  after {} timesteps".format(t+1))

env.cleanup()
