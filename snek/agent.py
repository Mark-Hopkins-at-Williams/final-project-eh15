import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotter import plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

class Agent:
    def __init__(self):
        self.n_games = 0 #The number of games played
        self.eps = 0 # randomness parameter for exploration/exploitation tradeoff
        self.gamma = 0.9 # discount rate for future rewards
        self.memory = deque(maxlen=MAX_MEMORY) # stores agent's experiences for training. when we exceed our max memory, popleft()
        self.model = Linear_QNet(11, 256, 3) #The Q-learning model used by the agent (input size, hidden size, output size). Hidden size can be changed, but the input is always 11 and output is always 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) #The trainer object for the Q-learning model


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, ends = zip(*mini_sample) #same as a for loop:
        self.trainer.train_step(states, actions, rewards, next_states, ends)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.eps = 80 - self.n_games
        next_move = [0,0,0]
        if random.randint(0, 200) < self.eps:
            move = random.randint(0, 2)
            next_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            next_move[move] = 1

        return next_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get curr state
        state_curr = agent.get_state(game)

        # get next move based on the state
        next_move = agent.get_action(state_curr)

        # perform move and get new state
        reward, game_over, score = game.play_step(next_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_curr, next_move, reward, state_new, game_over)

        # remember
        agent.remember(state_curr, next_move, reward, state_new, game_over)

        if game_over:
            # train long memory aka replay memory or experience memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            #plotting
            plot_scores.append(score) #append score to plot list
            total_score += score #update total score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score) #append mean score to plot list
            plot(plot_scores, plot_mean_scores) #plot scores and mean scores


if __name__ == '__main__':
    train()