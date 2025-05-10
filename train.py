import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from collections import deque
import numpy as np  # Add this import for grid rotation


class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
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
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.batch_size = 32

        self.model = CNN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            target = reward
            if reward > 0:  # Non-terminal state
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            # Perform gradient descent
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path="snake_dqn.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="snake_dqn.pth"):
        import os
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint.get('epsilon')
            self.model.to(self.device)  # Ensure model is on the correct device
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print("Model file not found. Starting with a new model.")


def encode_state(game, device):
    # Rotate the apple's position
    snakegrid = torch.from_numpy(game.grid).float()
    apple_grid = torch.from_numpy(game.apple.grid).float()

    state = torch.stack([snakegrid, apple_grid], dim=0).to(device)  # Shape: (2, rows, cols)
    return state.unsqueeze(0)


def train(agent, rows, cols, show=True, episodes=10000):
    game = Game(rows, cols)

    total_length = 0

    if show:
        draw = Draw(rows, cols)

    with tqdm(total=episodes, desc="Training") as pbar:
        for episode in range(episodes):
            state = encode_state(game, agent.device)
            while True:
                action = agent.act(state)

                game.turn(action)
                reward = game.update()

                if show and draw.running:
                    draw.display(game)

                next_state = encode_state(game, agent.device)
                agent.remember(state, action, reward, next_state)

                if reward < 0:  # Game over
                    total_length += game.length
                    game.reset()

                    pbar.set_postfix({"Average Length": f"{total_length / (episode+1):.2f}"})
                    pbar.update(1)
                    break

                state = next_state
                agent.replay()

def test(agent, rows, cols, show=True, episodes=1000):
    game = Game(rows, cols)

    total_length = 0

    if show:
        draw = Draw(rows, cols)

    with tqdm(total=episodes, desc="Testing") as pbar:
        for episode in range(episodes):
            state = encode_state(game, agent.device)
            while True:
                action = agent.act(state)
                game.turn(action)
                reward = game.update()
                
                if show and draw.running:
                    draw.display(game)
        
                if reward < 0:  # Game over
                    total_length += len(game.length)
                    game.reset()

                    pbar.set_postfix({"Average Length": f"{total_length / (episode+1):.2f}"})
                    pbar.update(1)
                    break

                state = encode_state(game, agent.device)

if __name__ == "__main__":
    #from game_array import Game, Draw
    from game_deque import Game, Draw

    rows = 9
    cols = 9

    state_size = 2      # 2 channels: snake grid and apple grid
    action_size = 4     # 4 actions: left, down, right, up

    agent = DQNAgent([state_size, rows, cols], action_size)
    
    # agent.load_model()

    train(agent, rows, cols)
    #agent.save_model()

    #agent.load_model()
    test(agent, rows, cols)