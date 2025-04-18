import numpy as np
from collections import deque
import random
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class Apple:
    def __init__(self, grid):
        self.grid = grid

        self.update()
        
    def update(self):
        empty = [(r, c) for r in range(self.grid.shape[0]) for c in range(self.grid.shape[1]) if self.grid[r, c] == 0]
        if len(empty) > 0:
            self.position = random.choice(list(empty))
        else:
            self.position = None


class Game:
    def __init__(self, rows, cols, boarder=True):
        self.rows = rows
        self.cols = cols
        self.boarder = boarder
        
        self.grid = np.zeros((rows, cols), dtype=np.float32)
        self.apple = Apple(self.grid)

        self.init()
    
    def init(self):
        self.snake = deque()
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])

        start_pos = (self.rows // 2, self.cols // 2)
        self.snake.append(start_pos)
        self.grid[start_pos] = 1

        self.apple.update()
    
    def reset(self):
        self.grid[:, :] = 0

        self.init()
    
    def move(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir
    
    def turn(self, new_dir):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        current_direction = self.direction
        direction_index = directions.index(current_direction)

        # Map relative action to absolute direction
        if new_dir == 0:  # Turn left
            self.direction = directions[(direction_index - 1) % 4]
        elif new_dir == 1:  # Go straight
            self.direction = current_direction
        elif new_dir == 2:  # Turn right
            self.direction = directions[(direction_index + 1) % 4]
    
    def update(self):
        head = self.snake[-1]

        if self.boarder:
            new = (head[0] + self.direction[0], head[1] + self.direction[1])
            if new[0] < 0 or new[0] >= self.rows or new[1] < 0 or new[1] >= self.cols:
                return -1      # Penalty for hitting the wall
        else:
            new = ((head[0] + self.direction[0]) % self.rows,
                   (head[1] + self.direction[1]) % self.cols)
        
        if self.grid[new] > 0:
            return -10          # Penalty for hitting itself
        
        self.snake.append(new)
        self.grid[new] = len(self.snake)

        if new == self.apple.position:
            self.apple.update()
            return 50           # Reward for eating an apple

        tail = self.snake.popleft()
        self.grid[tail] = 0

        self.grid[self.grid > 0] -= 1
        
        return 0                # No reward for moving


class Draw:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # Colors
        self.BG = (1.0, 1.0, 1.0)  # OpenGL uses normalized RGB (0.0 to 1.0)
        self.GRID = (0.0, 0.0, 0.0)
        self.APPLE = (1.0, 0.0, 0.0)

        self.grid_size = 50

        width = rows * self.grid_size
        height = cols * self.grid_size

        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption('Snake')
        
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, cols, rows, 0)  # Top-left origin
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(*self.BG, 1.0)
        self.running = True

    def draw_rect(self, x, y, color):
        # Intermediate mode
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + 1, y)
        glVertex2f(x + 1, y + 1)
        glVertex2f(x, y + 1)
        glEnd()

    def display(self, snake_pos, apple_pos):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT)

        # Draw the grid
        glColor3f(*self.GRID)
        glBegin(GL_LINES)
        for r in range(self.rows + 1):
            glVertex2f(0, r)
            glVertex2f(self.cols, r)
        for c in range(self.cols + 1):
            glVertex2f(c, 0)
            glVertex2f(c, self.rows)
        glEnd()

        # Draw the apple
        if apple_pos is not None:
            self.draw_rect(apple_pos[1], apple_pos[0], self.APPLE)

        # Draw the snake
        for idx, segment in enumerate(snake_pos):
            shade = max(0.0, 1.0 - idx * (1.0 / len(snake_pos)))
            self.draw_rect(segment[1], segment[0], (0.0, shade, 0.0))

        pygame.display.flip()


def play():
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
                    game.move((-1, 0))
                elif event.key == pygame.K_DOWN:
                    game.move((1, 0))
                elif event.key == pygame.K_LEFT:
                    game.move((0, -1))
                elif event.key == pygame.K_RIGHT:
                    game.move((0, 1))
                elif event.key == pygame.K_p:
                    print(game.grid)

        if game.update() < 0:
            game.reset()
            continue

        draw.display(game.snake, game.apple.position)
        clock.tick(tick)

if __name__ == "__main__":
    play()