# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import util
import random

from SwingyMonkey import SwingyMonkey



class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, it, round_size):
        self.last_state  = None
        self.last_state_unformat = None
        self.last_action = None
        self.last_reward = None
        self.curr_gravity = 0
        self.epoch = 1
        self.round_size = round_size
        self.it = it
        self.values = util.Counter()
        self.epsilon = [0.2 * self.transformSigmoid(self.epoch)]
        self.death_causes = {-10:[], -5:[]}


    def reset(self):
        self.last_state  = None
        self.last_state_unformat = None
        self.last_action = None
        self.last_reward = None
        self.curr_gravity = None
        self.epoch += 1
        self.epsilon.append(0.2 * self.transformSigmoid(self.epoch))
        #print(self.death_causes)

    def getValues(self):
        return self.values

    def getEpsilon(self):
        return self.epsilon

    def getDeathCauses(self):
        return self.death_causes

    def transformSigmoid(self, it):
        x = self.it/1.1 - it
        sig = np.exp(x)/(np.exp(x) + 1)
        return sig

    # in order: tree dist, tree top, tree bot, monkey vel, monkey top, monkey bot
    def format_into_tuple(self, state_dict, gravity):
        # return (state_dict['tree']['dist'], state_dict['tree']['top'],\
        #     state_dict['tree']['bot'], state_dict['monkey']['vel'],\
        #     state_dict['monkey']['top'], state_dict['monkey']['bot'])

        danger_high = state_dict['monkey']['top'] > 350
        danger_low = state_dict['monkey']['bot'] < 50
        # danger_high = 0
        # danger_low = 0
        return (state_dict['tree']['dist'] // self.round_size, (state_dict['tree']['top'] - state_dict['monkey']['top']) // self.round_size, \
                gravity, danger_high, danger_low)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        epsilon = self.epsilon[-1]
        gamma = 1
        learning_rate = 0.3

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

        if self.last_action is not None:
            last_Q = self.values[self.last_state, self.last_action]
            if self.values[state_formatted, 1] > self.values[state_formatted, 0]:
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

        self.last_state = state_formatted
        self.last_state_unformat = state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward
        if(reward < 0):
            self.death_causes[reward].append(self.last_state_unformat['monkey']['bot'])
            #print(reward)
            #print(np.mean(self.death_causes[reward]))

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

round_size = [10, 50, 100, 150, 200]
round_size = [100]
if __name__ == '__main__':

    values = util.Counter()
    death_causes = util.Counter()
    scores = {}
    for r in round_size:
        repeats = 50
        it_in_repeat = 100
        curr_round_score = []
        for i in range(repeats):
            # Select agent.
            agent = Learner(it_in_repeat, r)

            # Empty list to save history.
            hist = []

            # Run games.
            run_games(agent, hist, it_in_repeat, 1)

            # Add to values counter
            values += agent.getValues()

            # # Get Epsilon
            # import matplotlib.pyplot as plt
            #
            # plt.figure()
            # plt.plot(agent.getEpsilon())
            # plt.show()

            # Score Avg
            curr_round_score.append(hist)

            # Print Death Causes
            death_causes += agent.getDeathCauses()

            # Save history.
            np.save('hist',np.array(hist))
        scores[r] = curr_round_score
    # Get Values
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(list(values.values()), bins = np.arange(-10, 10, 1))
    # plt.title("Q Value Distribution")
    # plt.show()

import matplotlib.pyplot as plt
# # Death Causes: Show distribution of death locations
# plt.figure()
# plt.hist(death_causes[-10])
# plt.title("Location of Character (bot) before Death")
# plt.show()


for r in round_size:
    curr_scores = np.asarray(scores[r])
    print(scores)

    print_scores = curr_scores.flatten()
    print_scores = [max(j) for j in curr_scores]
    plt.figure()
    bins = np.arange(0, max(print_scores) + 2, 1)
    plt.hist(print_scores, bins = bins)
    plt.xticks(np.arange(0, max(print_scores) + 2, 1))
    plt.title('Max Scores Across Iterations for Rounding ' + str(r))
    plt.show()
    print("hello")

