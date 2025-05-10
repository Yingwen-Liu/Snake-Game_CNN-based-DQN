import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNSnake(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNSnake, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * input_shape[1] * input_shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class DQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNSnake(input_shape, num_actions).to(self.device)
        self.target_model = DQNSnake(input_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 3)  # Random action (0: left, 1: up, 2: right, 3: down)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def preprocess_state(grid, apple_grid):
    # Combine the snake grid and apple grid into a single input tensor
    state = np.stack([grid, apple_grid], axis=0)
    return state.astype(np.float32)

def train_snake_dqn(game, episodes=1000, target_update=10):
    input_shape = (2, game.rows, game.cols)
    num_actions = 4
    agent = DQNAgent(input_shape, num_actions)

    for episode in range(episodes):
        game.reset()
        state = preprocess_state(game.grid, game.apple.grid)
        total_reward = 0

        while True:
            action = agent.act(state)
            direction = [(0, -1), (-1, 0), (0, 1), (1, 0)][action]
            game.move(direction)

            reward = game.update()
            done = reward < 0
            next_state = preprocess_state(game.grid, game.apple.grid)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
                break

            agent.replay()

        if episode % target_update == 0:
            agent.update_target_model()

if __name__ == "__main__":
    from game_array import Game
    rows, cols = 15, 15
    game = Game(rows, cols)
    train_snake_dqn(game)
