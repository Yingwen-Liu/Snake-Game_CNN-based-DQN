import random
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class Apple:
    def __init__(self, rows, cols, num):
        # Initialize the apple grid and number of apples
        self.num = num
        self.position = []
        self.grid = np.zeros((rows, cols), dtype=int)

    def reset(self, empty):
        # Reset apple positions, avoiding the snake's grid
        self.position = random.sample(empty, self.num)
        
        self.grid.fill(0)
        for pos in self.position:
            self.grid[pos] = 1

    def update(self, empty, head):
        # Update apple position after being eaten
        self.position.remove(head)
        self.grid[head] = 0

        chosen_pos = random.choice([x for x in empty if x not in self.position])
        self.position.append(chosen_pos)
        self.grid[chosen_pos] = 1


class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class Deque:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value):
        new_node = Node(value)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def appendleft(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

    def pop(self):
        value = self.tail.value
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        return value

    def popleft(self):
        value = self.head.value
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        else:
            self.head.prev = None
        return value
    
    def __iter__(self):
        # Iterate through the deque from head to tail
        current = self.head
        while current:
            yield current
            current = current.next


class Game:
    def __init__(self, rows, cols, apple_num=1, boarder=True):
        # Initialize the game grid, snake, and apple
        self.rows = rows
        self.cols = cols
        self.boarder = boarder
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        self.apple = Apple(rows, cols, apple_num)

        self.reset()

    def reset(self):
        # Reset the game state
        self.position = Deque() # Snake position
        start_pos = (self.rows // 2, self.cols // 2)
        self.position.append(start_pos)

        self.empty = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.empty.remove(start_pos)

        self.grid = np.zeros((self.rows, self.cols), dtype=int)   # snake grid
        self.grid[start_pos] = 1

        self.length = 1
        self.direction = random.choice(self.directions)

        self.apple.reset(self.empty)

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
        head = self.position.tail.value
        new = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self.boarder:
            if new[0] < 0 or new[0] >= self.rows or new[1] < 0 or new[1] >= self.cols:
                return -1  # Penalty for hitting the wall
        else:
            new = (new[0] % self.rows, new[1] % self.cols)

        if new in [node.value for node in self.position]:
            return -10  # Penalty for hitting itself

        self.position.append(new)
        self.grid[new] = self.length + 1

        if new in self.apple.position:
            self.length += 1
            self.apple.update(self.empty, new)
            self.empty.remove(new)
            return 50  # Reward for eating an apple

        tail = self.position.popleft()
        self.empty.append(tail)
        self.grid[self.grid > 0] -= 1

        return 0  # No reward for moving


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

        width = cols * self.grid_size
        height = rows * self.grid_size

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
        # Precompute grid vertices for faster rendering
        lines = []
        for r in range(self.rows + 1):
            lines.append((0, r))
            lines.append((self.cols, r))
        for c in range(self.cols + 1):
            lines.append((c, 0))
            lines.append((c, self.rows))
        return [coord for line in lines for coord in line]  # Flatten the list

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

        # Draw the apple
        for pos in game.apple.position:
            self.draw_rect(pos[1], pos[0], self.APPLE)

        # Draw the snake
        for idx, segment in enumerate(game.position):
            shade = max(0.0, 1.0 - idx / game.length)
            self.draw_rect(segment.value[1], segment.value[0], (0.0, shade, 0.0))
        
        # Draw the grid using glDrawElements
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
