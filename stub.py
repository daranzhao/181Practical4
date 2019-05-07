# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import util

from SwingyMonkey import SwingyMonkey

class Learner(object):
    '''
    This agent plays the swinging monkey game
    '''
    def __init__(self, it, round_size, gamma, learning_rate):
        self.last_state  = None         # Previous state
        self.last_state_unformat = None # Previous state, unformatted
        self.last_action = None         # Last action
        self.last_reward = None         # Last reward
        self.curr_gravity = 0           # Gravity in this stage
        self.epoch = 1                  # Number of epochs covered
        self.round_size = round_size    # Number of pixels to round states to
        self.it = it                    # Number of iterations run by alg
        self.values = util.Counter()    # Q values
        self.epsilon = [0]              #Surprinsgly, no exploration works best | Prev Exploration: [0.2 * self.transformSigmoid(self.epoch)]
        self.death_causes = {-10:[], -5:[]} # Monkey bot height at each death
        self.gamma = gamma              # Discount factor
        self.learning_rate = learning_rate  # Learning Rate
        self.state_counter = util.Counter() # Number of times a state has been encountered

    def reset(self):
        # Resets some variables
        self.last_state  = None
        self.last_state_unformat = None
        self.last_action = None
        self.last_reward = None
        self.curr_gravity = None

        # Iterate epoch
        self.epoch += 1

    # Caller functions
    def getValues(self):
        return self.values

    def getEpsilon(self):
        return self.epsilon

    def getDeathCauses(self):
        return self.death_causes

    # Transforms sigmoid function for exploration
    def transformSigmoid(self, it):
        x = self.it/1.1 - it
        sig = np.exp(x)/(np.exp(x) + 1)
        return sig

    # Returns features describing a state
    def format_into_tuple(self, state_dict, gravity):
        danger_high = state_dict['monkey']['top'] > 350
        danger_low = state_dict['monkey']['bot'] < 50
        return (state_dict['tree']['dist'] // self.round_size, (state_dict['tree']['top'] - state_dict['monkey']['top']) // self.round_size, \
                gravity, danger_high, danger_low)

    # Returns the desired action
    def action_callback(self, state):
        # Get hyperparameters
        epsilon = self.epsilon[-1]
        gamma = self.gamma
        learning_rate = self.learning_rate

        # Infer Gravity
        if self.curr_gravity == None and self.last_state_unformat != None:
            time = (state['tree']['dist'] - self.last_state_unformat['tree']['dist'])/25.
            vel_change = state['monkey']['vel'] - self.last_state_unformat['monkey']['vel']
            gravity = vel_change/time
            if(gravity == 4 or gravity == 1):
                self.curr_gravity = gravity

        # Format state
        if self.curr_gravity == None:
            state_formatted = self.format_into_tuple(state, 4)
        else:
            state_formatted = self.format_into_tuple(state, self.curr_gravity)

        # Determine action via Q-values and learn Q-values
        if self.last_action is not None:
            last_Q = self.values[self.last_state, self.last_action]
            left = self.values[state_formatted, 1]
            right = self.values[state_formatted, 0]
            if left > right:
                next_val = self.values[state_formatted, 1]
                next_action = 1
            else:
                next_val = self.values[state_formatted, 0]
                next_action = 0
            if npr.rand() < epsilon:
                next_action = npr.rand() < 0.1

            self.values[self.last_state, self.last_action] =\
                last_Q - (learning_rate)*(last_Q - self.last_reward - gamma*next_val)
            self.last_action = next_action
        else:
            if npr.rand() < epsilon:
                self.last_action = 0
            else:
                self.last_action = 1

        # Update Counter for number of times a state has been encountered
        if (state_formatted, self.last_action) in self.state_counter.keys():
            self.state_counter[(state_formatted, self.last_action)] += 1
        else:
            self.state_counter[(state_formatted, self.last_action)] = 1

        self.last_state = state_formatted
        self.last_state_unformat = state
        return self.last_action

    # Returns the reward
    def reward_callback(self, reward):
        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 10):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    pg.quit()
    return


if __name__ == '__main__':
    # Hyper parameters and global vars
    round_size = [100]
    learning_rate = [0.1]
    gamma = [1]
    values = util.Counter()
    death_causes = util.Counter()
    scores = {}

    # Runs iterations of monkey for 30 iterations, and 300 epochs within each iteration
    for l in learning_rate:
        for g in gamma:
            for r in round_size:
                repeats = 30
                it_in_repeat = 300
                curr_round_score = []
                for i in range(repeats):
                    # Select agent.
                    agent = Learner(it_in_repeat, r, g, l)

                    # Empty list to save history.
                    hist = []

                    # Run games.
                    run_games(agent, hist, it_in_repeat, 1)

                    # Add to values counter
                    values += agent.getValues()

                    # Score Avg
                    curr_round_score.append(hist)

                    # Print Death Causes
                    death_causes += agent.getDeathCauses()

                    # Save history.
                    np.save('hist',np.array(hist))
                scores[(r, g, l)] = curr_round_score


