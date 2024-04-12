import os, sys
import gymnasium as gym
import time
import pickle
from utils import SarsaAgent

import text_flappy_bird_gym

if __name__ == '__main__':

    # Import the trained agent
    with open('sarsa_agent.pkl', 'rb') as inp:
        sarsa_agent = pickle.load(inp)

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    state, info = env.reset()

    # iterate
    while True:

        # Select next action
        action = sarsa_agent.act(state)

        # Appy action and return new observation of the environment
        state, reward, done, _, info = env.step(action)

        # Render the game
        os.system("cls")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close()
