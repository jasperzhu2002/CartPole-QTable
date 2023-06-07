'''

Q-Table solution to Open AI Gym Pole Cart game. 

Used info from:
https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/
https://github.com/jaekookang/RL-cartpole/blob/master/05_Q-Net.py

'''

import gym
import numpy as np
import random

# initialize environment
env = gym.make('CartPole-v1')

# initialize q-table
state_size = (1, 1, 39, 15)
action_size = env.action_space.n
qtable = np.zeros(state_size + (action_size, ))

# sets bounds
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[3] = [-np.radians(50), np.radians(50)]

# parameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate= 0.005

def make_discrete(state):
    section = []

    for i in range(len(state)):
        # for x and x'
        if i <= 1:
            section.append(0)
        # for angle and angle'
        else:
            if state[i] <= state_bounds[i][0]:
                discrete_val = 0
            elif state[i] >= state_bounds[i][1]:
                discrete_val = state_size[i] - 1
            else:
                bound_range = state_bounds[i][1] - state_bounds[i][0]
                offset = (state_size[i] - 1) * state_bounds[i][0] / bound_range
                section_scale = (state_size[i] - 1) / bound_range
                discrete_val = int(round(section_scale * state[i] - offset))
            # add to array
            section.append(discrete_val)

    return tuple(section)

#actual loop
for episode in range(5000):
    # resets
    r = 0
    state, __ = env.reset()
    discrete_state = make_discrete(state)

    for step in range(200):
        # choose between explore and exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[discrete_state])
        
        # gets action data
        new_state, reward, terminated, truncated , info = env.step(action)
        discrete_new_state = make_discrete(new_state)

        # Q-learns and updates
        # qtable[discrete_state + (action, )] = (1 - learning_rate) *  qtable[discrete_state + (action, )] + learning_rate * (reward + discount_rate * np.amax(qtable[discrete_new_state]))
        qtable[discrete_state + (action, )] += learning_rate * (reward + discount_rate * np.amax(qtable[discrete_new_state]) - qtable[discrete_state + (action, )])
        
        discrete_state = discrete_new_state

        # check for done
        if terminated:
            print('Episode:{}/{} finished at timestep:{}'.format(episode, 1000, step))
            break
    # decay epsiolon
    epsilon = np.exp(-decay_rate*episode)

print(qtable)

env.close()