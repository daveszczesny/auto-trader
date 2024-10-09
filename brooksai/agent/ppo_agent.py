import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


from brooksai.env.models.constants import ApplicationConstants
from brooksai.nnmodels.transformer import TransformerModel

class PPOAgent:
    def __init__(self, env: gym.Env, state_dim, action_dim, hidden_dim, num_heads, num_layers):
        self.env = env

        self.model = TransformerModel(
            state_dim,
            hidden_dim,
            action_dim,
            num_heads,
            num_layers
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.model(state)
        action_probs = torch.sigmoid(action_probs)
        action_probs = action_probs.detach()

        action_low = torch.FloatTensor(self.env.action_space.low)
        action_high = torch.FloatTensor(self.env.action_space.high)

        action = action_low + action_probs * (action_high - action_low)

        return action.numpy()

    def update(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).view(states.size(0), -1)
        rewards = torch.FloatTensor(rewards)

        self.optimizer.zero_grad()
        action_probs = self.model(states).view(states.size(0), -1)
        loss = self.criterion(action_probs, actions)
        loss.backward()
        self.optimizer.step()

    def learn(self, total_timesteps: int = 4_000_000):
        
        state = self.env.reset()[0]
        for _ in range(total_timesteps):
            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action)
            self.update([state], [action], [reward])
            state = next_state
            if done:
                state = self.env.reset()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, observation):
        raw_action, _ = self.model.predict(observation)
        return raw_action
