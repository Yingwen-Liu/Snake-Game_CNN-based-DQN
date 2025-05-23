import numpy as np
import random
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class Apple:
    def __init__(self, rows, cols, num):
        # Initialize the apple grid and number of apples
        self.num = num
        self.grid = np.zeros((rows, cols), dtype=int)

    def reset(self, snake_grid):
        # Reset apple positions, avoiding the snake's grid
        self.grid.fill(0)
        empty = np.argwhere(snake_grid == 0)

        chosen_pos = empty[np.random.choice(len(empty), self.num, replace=False)]
        self.grid[tuple(chosen_pos.T)] = 1

    def update(self, snake_grid, head):
        # Update apple position after being eaten
        self.grid[head] = 0
        empty = np.argwhere(snake_grid + self.grid == 0)

        chosen_pos = empty[random.randint(0, len(empty) - 1)]
        self.grid[tuple(chosen_pos)] = 1


class Game:
    def __init__(self, rows, cols, apple_num=1, boarder=True):
        # Initialize the game grid, snake, and apple
        self.rows = rows
        self.cols = cols
        self.boarder = boarder
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]    # left, down, right, up
        
        self.apple = Apple(rows, cols, apple_num)

        self.reset()
    
    def reset(self):
        # Reset the game state
        self.grid = np.zeros((self.rows, self.cols), dtype=int)   # snake grid
        self.grid[(self.rows // 2, self.cols // 2)] = 1

        self.length = 1
        self.direction = random.choice(self.directions)

        self.apple.reset(self.grid)
    
    def move(self, dir):
        # Change the snake's direction if valid
        dir = self.directions[dir]
        if (dir[0] * -1, dir[1] * -1) != self.direction:
            self.direction = dir
    
    def turn(self, dir):
        # Turn the snake based on relative direction
        dir_idx = self.directions.index(self.direction)

        # Map relative action to absolute direction
        if dir == 0:  # Turn left
            self.direction = self.directions[(dir_idx - 1) % 4]
        elif dir == 2:  # Turn right
            self.direction = self.directions[(dir_idx + 1) % 4]
    
    def update(self):
        # Update the game state after each move
        head = np.unravel_index((self.grid == self.length).argmax(), self.grid.shape)
        new = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self.boarder:
            if new[0] < 0 or new[0] >= self.rows or new[1] < 0 or new[1] >= self.cols:
                return -1      # Penalty for hitting the wall
        else:
            new = (new[0] % self.rows, new[1] % self.cols)
        
        if self.grid[new] > 0:
            return -10          # Penalty for hitting itself
        
        self.grid[new] = self.length + 1

        if self.apple.grid[new] == 1:
            self.length += 1
            self.apple.update(self.grid, new)
            return 50           # Reward for eating an apple

        self.grid[self.grid > 0] -= 1
        
        return 0                # No reward for moving


class Draw:
    def __init__(self, rows, cols):
        # Initialize OpenGL and set up the display
        self.rows = rows
        self.cols = cols

        # Colors
        self.BG = (1.0, 1.0, 1.0)       # Background
        self.GRID = (0.0, 0.0, 0.0)     # Grid
        self.APPLE = (1.0, 0.0, 0.0)    # Apple

        self.grid_size = 50

        width = rows * self.grid_size
        height = cols * self.grid_size

        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption('Snake')
        
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, cols, rows, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(*self.BG, 1.0)
        self.running = True

        self.grid_lines = self.create_grid_lines()
        self.grid_indices = self.create_grid_indices()

    def create_grid_lines(self):
        # Precompute grid lines for faster rendering
        lines = []
        for r in range(self.rows + 1):
            lines.append((0, r))
            lines.append((self.cols, r))
        for c in range(self.cols + 1):
            lines.append((c, 0))
            lines.append((c, self.rows))
        return np.array(lines, dtype=np.float32)
    
    def create_grid_indices(self):
        # Generate indices for grid lines
        indices = []
        for i in range(0, len(self.grid_lines), 2):
            indices.append(i)
            indices.append(i + 1)
        return indices

    def draw_rect(self, x, y, color):
        # Draw a rectangle at the specified position with the given color
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + 1, y)
        glVertex2f(x + 1, y + 1)
        glVertex2f(x, y + 1)
        glEnd()

    def display(self, game):
        # Render the game state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT)

        # Draw the apple and snake
        for r in range(self.rows):
            for c in range(self.cols):
                if game.grid[r, c] > 0:
                    shade = max(0.0, 1.0 - (game.grid[r, c] - 1.0) / game.length)
                    self.draw_rect(c, r, (0.0, shade, 0.0))
                elif game.apple.grid[r, c] == 1:
                    self.draw_rect(c, r, self.APPLE)
        
        # Draw the grid
        glColor3f(*self.GRID)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, self.grid_lines)
        glDrawElements(GL_LINES, len(self.grid_indices), GL_UNSIGNED_INT, self.grid_indices)
        glDisableClientState(GL_VERTEX_ARRAY)

        pygame.display.flip()


def play():
    # Main game loop
    rows = 15
    cols = 15

    game = Game(rows, cols)
    draw = Draw(rows, cols)

    clock = pygame.time.Clock()
    tick = 10

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.move(3)
                elif event.key == pygame.K_DOWN:
                    game.move(1)
                elif event.key == pygame.K_LEFT:
                    game.move(0)
                elif event.key == pygame.K_RIGHT:
                    game.move(2)
                elif event.key == pygame.K_p:
                    print(game.grid - game.apple.grid)

        if game.update() < 0:
            game.reset()
            continue

        draw.display(game)
        clock.tick(tick)

if __name__ == "__main__":
    play()