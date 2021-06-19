import random
import os
import gym
import numpy as np
import time
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAYMEMORY_SIZE = 1000
MODEL_NAME = 'CartPole'
STATE_SHAPE = 4
ACTION_N = 2
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
EPISODES = 20000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


def create_model():
    """
    used for constructing DNN model
    :return: model
    """
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(1, 4)))
    model.add(Dense(256, input_shape=(STATE_SHAPE, ), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(ACTION_N, activation="linear"))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=.001), metrics=['accuracy'])
    return model


class DQNAgent:
    def __init__(self):
        self.model = create_model()

        self.target_model = create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, STATE_SHAPE))

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAYMEMORY_SIZE:
            return

        # s, a, r, s_, done = minibatch
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # current state
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # new state
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        # prepare the data for training
        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update the Q-value table
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = DQNAgent()

    if not os.path.isdir('models'):
        os.makedirs('models')

    for episode in range(1, EPISODES+1):
        current_state = env.reset()
        done = False
        episode_reward = 0
        step = 1

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            if episode % 200 == 0:
                env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

