import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from collections import deque
from game import Game, Draw
import numpy as np  # Add this import for grid rotation


class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.batch_size = 16

        self.model = CNN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        state = state.to(self.device)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

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
    direction_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}  # Left, Down, Right, Up
    rotation_steps = direction_map[game.direction]  # Number of 90-degree rotations

    # Rotate the grid so the snake's head always faces upward
    snake_grid = torch.from_numpy(np.rot90(game.grid, k=rotation_steps).copy()).float()
    snake_grid /= (game.rows * game.cols)

    apple_grid = torch.zeros((game.rows, game.cols), dtype=torch.float32)
    apple_pos = game.apple.position

    # Rotate the apple's position
    for _ in range(rotation_steps):
        apple_pos = (apple_pos[1], game.rows - 1 - apple_pos[0])
    apple_grid[apple_pos[0], apple_pos[1]] = 1.0

    state = torch.stack([snake_grid, apple_grid], dim=0).to(device)  # Shape: (2, rows, cols)
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
                    draw.display(game.snake, game.apple.position)

                next_state = encode_state(game, agent.device)
                agent.remember(state, action, reward, next_state)

                if reward < 0:  # Game over
                    total_length += len(game.snake)
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
                    draw.display(game.snake, game.apple.position)
        
                if reward < 0:  # Game over
                    total_length += len(game.snake)
                    game.reset()

                    pbar.set_postfix({"Average Length": f"{total_length / (episode+1):.2f}"})
                    pbar.update(1)
                    break

                state = encode_state(game, agent.device)

if __name__ == "__main__":
    rows = 9
    cols = 9

    state_size = 2      # 2 channels: snake grid and apple grid
    action_size = 3     # 3 actions: turn left, go straight, turn right

    agent = DQNAgent(state_size, action_size)
    
    # agent.load_model()

    train(agent, rows, cols)
    #agent.save_model()

    #agent.load_model()
    test(agent, rows, cols)