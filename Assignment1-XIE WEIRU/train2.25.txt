# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:09:18 2025

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

# Action definitions
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_NUM = len(ACTIONS)

# Q-learning parameters
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1
MAX_EPISODES = 500  # Maximum number of training episodes
MAX_STEPS = 100  # Maximum number of steps per episode

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

# Find the start and end points
start = tuple(np.argwhere(maze == 2)[0])
end = tuple(np.argwhere(maze == 3)[0])

# Q-table save file path
Q_TABLE_FILE = 'q_table.npy'
OPTIMAL_PATH_FILE = 'optimal_path.npy'

# Try to load the Q-table
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE)
else:
    # Initialize the Q-table
    q_table = np.zeros((MAZE_ROWS * MAZE_COLS, ACTION_NUM))


def get_state_index(state):
    """Convert the state (row, column) to the index of the Q-table"""
    return state[0] * MAZE_COLS + state[1]


def choose_action(state):
    """Choose an action according to the ε-greedy strategy"""
    state_index = get_state_index(state)
    if np.random.uniform() < EPSILON:
        action_index = np.random.randint(0, ACTION_NUM)
    else:
        q_values = q_table[state_index, :]
        action_index = np.argmax(q_values)
    return action_index


def get_next_state(state, action_index):
    """Calculate the next state based on the current state and action"""
    row, col = state
    action = ACTIONS[action_index]
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
    return (next_row, next_col)


def get_reward(state):
    """Calculate the reward based on the current state"""
    if state == end:
        return 100
    elif maze[state] == 1:
        return -10
    else:
        return -1


def update_q_table(state, action_index, next_state, reward):
    """Update the Q-table"""
    state_index = get_state_index(state)
    next_state_index = get_state_index(next_state)
    q_predict = q_table[state_index, action_index]
    q_target = reward + GAMMA * np.max(q_table[next_state_index, :])
    q_table[state_index, action_index] += ALPHA * (q_target - q_predict)


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
                pygame.draw.rect(screen, YELLOW, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))  # The exit is changed to yellow
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
pygame.display.set_caption('Simple Maze Game with Q - learning')

try:
    best_score = float('-inf')
    best_steps = float('inf')
    optimal_path = None
    # Training process
    if not os.path.exists(Q_TABLE_FILE):
        for episode in range(MAX_EPISODES):
            state = start
            total_reward = 0
            step = 0
            path = []  # Record the path of each episode
            while state != end and step < MAX_STEPS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit

                draw_maze()
                action_index = choose_action(state)
                next_state = get_next_state(state, action_index)
                reward = get_reward(next_state)
                update_q_table(state, action_index, next_state, reward)

                total_reward += reward
                state = next_state
                step += 1
                path.append(state)

                # Draw the maze first, then draw the agent to ensure the agent is above the exit
                draw_agent(state)

                # Display the real-time action status and reward score
                action_text = f"Action: {ACTIONS[action_index]}"
                reward_text = f"Total Reward: {total_reward}"
                step_text = f"Step: {step}"
                draw_text(action_text, 10, 10)
                draw_text(reward_text, 10, 40)
                draw_text(step_text, 10, 70)

                pygame.display.flip()

                # Speed up the training and reduce unnecessary delays
                time.sleep(0.005)

            print(f"Episode {episode + 1} finished with total reward {total_reward} in {step} steps.")

            if total_reward > best_score or (total_reward == best_score and step < best_steps):
                best_score = total_reward
                best_steps = step
                optimal_path = path

        # Save the trained Q-table
        np.save(Q_TABLE_FILE, q_table)
        if optimal_path is not None:
            # Save the optimal path
            np.save(OPTIMAL_PATH_FILE, np.array(optimal_path))

            step_count = 1  # Initialize the step counter
            # Display the optimal path at a slower speed
            for state in optimal_path:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit

                draw_maze()
                draw_agent(state)

                step_text = f"Optimal Path Step: {step_count}"
                draw_text(step_text, 10, 10)

                # Display the shortest number of steps and the final reward score
                score_text = f"Best Steps: {best_steps}, Final Reward: {best_score}"
                draw_text(score_text, 10, 40)

                step_count += 1
                time.sleep(0.5)
                pygame.display.flip()

except SystemExit:
    pass
finally:
    pygame.quit()