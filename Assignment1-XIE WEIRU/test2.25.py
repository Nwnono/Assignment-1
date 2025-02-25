# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:08:40 2025

@author: dell
"""

import numpy as np
import pygame
import time
import os

# Maze parameter settings
MAZE_ROWS = 6
MAZE_COLS = 6
BLOCK_SIZE = 100
WINDOW_WIDTH = MAZE_COLS * BLOCK_SIZE
WINDOW_HEIGHT = MAZE_ROWS * BLOCK_SIZE
# Added UI display area height
UI_HEIGHT = 100
TOTAL_WINDOW_HEIGHT = WINDOW_HEIGHT + UI_HEIGHT

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)  # Added yellow definition for the exit
BLUE = (0, 0, 255)

# 6x6 maze definition, adding more obstacles
# 0: path, 1: trap, 2: start point, 3: end point
maze = np.array([
    [2, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 3]
])

# Q-table save file path
OPTIMAL_PATH_FILE = 'optimal_path.npy'
Q_TABLE_FILE = 'q_table.npy'

# Load the Q-table to get the best number of steps and the final reward score
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE)
    best_score = float('-inf')
    best_steps = float('inf')
    optimal_path = None
    # Simulate the training process to get the best number of steps and the final reward score (a better saving method can be considered in practice)
    MAX_EPISODES = 500
    MAX_STEPS = 100
    for episode in range(MAX_EPISODES):
        state = tuple(np.argwhere(maze == 2)[0])
        total_reward = 0
        step = 0
        path = []
        while state != tuple(np.argwhere(maze == 3)[0]) and step < MAX_STEPS:
            state_index = state[0] * MAZE_COLS + state[1]
            q_values = q_table[state_index, :]
            action_index = np.argmax(q_values)
            row, col = state
            action = ['up', 'down', 'left', 'right'][action_index]
            if action == 'up':
                next_row = max(0, row - 1)
                next_col = col
            elif action == 'down':
                next_row = min(MAZE_ROWS - 1, row + 1)
                next_col = col
            elif action == 'left':
                next_row = row
                next_col = max(0, col - 1)
            elif action == 'right':
                next_row = row
                next_col = min(MAZE_COLS - 1, col + 1)
            next_state = (next_row, next_col)
            if next_state == tuple(np.argwhere(maze == 3)[0]):
                reward = 100
            elif maze[next_state] == 1:
                reward = -10
            else:
                reward = -1
            total_reward += reward
            state = next_state
            step += 1
            path.append(state)
        if total_reward > best_score or (total_reward == best_score and step < best_steps):
            best_score = total_reward
            best_steps = step
            optimal_path = path
else:
    print("Q-table file not found. Please run the training code first.")
    exit()

# Load the optimal path
if os.path.exists(OPTIMAL_PATH_FILE):
    optimal_path = np.load(OPTIMAL_PATH_FILE)
else:
    print("Optimal path file not found. Please run the training code first.")
    exit()


def draw_maze():
    """Draw the maze"""
    screen.fill(WHITE)
    for row in range(MAZE_ROWS):
        for col in range(MAZE_COLS):
            if maze[row, col] == 1:
                pygame.draw.rect(screen, RED, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            elif maze[row, col] == 2:
                pygame.draw.rect(screen, GREEN, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            elif maze[row, col] == 3:
                pygame.draw.rect(screen, YELLOW, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, BLACK, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)


def draw_agent(state):
    """Draw the agent"""
    row, col = state
    pygame.draw.circle(screen, (0, 128, 255), (col * BLOCK_SIZE + BLOCK_SIZE // 2, row * BLOCK_SIZE + BLOCK_SIZE // 2),
                       BLOCK_SIZE // 3)


def draw_text(text, x, y):
    """Display text on the interface"""
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, (x, y + WINDOW_HEIGHT))


# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, TOTAL_WINDOW_HEIGHT))
pygame.display.set_caption('View Optimal Path')

try:
    step_count = 1
    for state in optimal_path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        draw_maze()
        draw_agent(state)

        step_text = f"Optimal Path Step: {step_count}"
        draw_text(step_text, 10, 10)

        score_text = f"Best Steps: {best_steps}, Final Reward: {best_score}"
        draw_text(score_text, 10, 40)

        step_count += 1
        time.sleep(0.5)
        pygame.display.flip()

    # Keep the window open until the user closes it
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

except SystemExit:
    pass
finally:
    pygame.quit()