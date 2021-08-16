from collections import deque
from keras import Sequential
from keras import layers, optimizers
import numpy as np
import random

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
MEMORY_SIZE = 1000000
LEARNING_RATE = 0.001
BATCH_SIZE = 24
GAMMA = 0.95


class dqnAgent(object):
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential([
            layers.Dense(128, input_shape=[observation_space], activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_space)
        ])
        self.model_path = './model.h5'
        self.model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE), loss='mse')
        try:
            self.model.load_weights(self.model_path)
        except:
            pass

    def remember(self, obs, act, reward, next_state, done):
        self.memory.append((obs, act, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        Q_list = self.model.predict(state)
        return np.argmax(Q_list[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            if terminal:
                q_update = reward
            else:
                q_update = reward + (GAMMA * np.max(self.model.predict(state_next)[0]))

            # 修正Q(state, action) <- target_Q
            Q = self.model.predict(state)
            Q[0][action] = q_update
            self.model.fit(state, Q, verbose=0)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self):
        self.model.save_weights(self.model_path)
