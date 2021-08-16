import gym
import numpy as np
from agent import dqnAgent
import math

ENV_NAME = "CartPole-v0"
# ENV_NAME = "MountainCar-v0"
EPISODES = 30


def evaluate(env):
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = dqnAgent(observation_space, action_space)
    agent.exploration_rate = 0

    for run in range(5):
        run = 0
        state = np.reshape(env.reset(), [-1, observation_space])
        while True:
            run +=1
            action = agent.act(state)
            state_next, reward, done, _ = env.step(action)

            env.render()
            state = np.reshape(state_next, [-1, observation_space])
            if done:
                print(f"In Episode: {run}, exploration: {agent.exploration_rate}, score: {run}")
                break


def main(env):
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = dqnAgent(observation_space, action_space)
    # counter-> episode
    for episode in range(EPISODES):
        state = np.reshape(env.reset(), [-1, observation_space])
        # reward counter
        total_reward = 0
        # game flow
        run = 0
        while True:
            run += 1
            action = agent.act(state)
            state_next, reward, done, _ = env.step(action)
            if episode % 5 == 0:
                env.render()
            # reward when the car reach the top hill
            reward = (run * reward) if not done else -reward
            state_next = np.reshape(state_next, [-1, observation_space])
            """
            reward = 100 * ((math.sin(3 * state_next[0, 0]) * 0.0025 + 0.5 * state_next[0, 1] * state_next[0, 1]) - (
                        math.sin(3 * state[0, 0]) * 0.0025 + 0.5 * state[0, 1] * state[0, 1]))
            """

            agent.remember(state, action, reward, state_next, done)
            state = state_next
            if done:
                print(f"In Episode: {episode}, exploration: {agent.exploration_rate}, score: {run}")
                break
            agent.experience_replay()
    agent.save()


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    # main(env)
    print("-" * 50)
    evaluate(env)
