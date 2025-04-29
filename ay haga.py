import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

q_table = np.zeros((env.observation_space.n, env.action_space.n))
num_episodes = 1000
alpha = 0.8
gamma = 0.95

epsilon = 1.0
eps_min = 0.01
eps_decay = 0.005

for ep in range(num_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if random.random() < epsilon:
            move = env.action_space.sample()
        else:
            move = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(move)
        done = terminated or truncated

        old_value = q_table[state, move]
        next_max = np.max(q_table[next_state])

        q_table[state, move] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

    epsilon = eps_min + (1.0 - eps_min) * np.exp(-eps_decay * ep)

print("Finished training!")

for ep in range(5):
    state = env.reset()[0]
    done = False
    print(f"Episode {ep+1}")

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()
        state = next_state

    if reward == 1:
        print("Success!")
    else:
        print("Fell into a hole.")
