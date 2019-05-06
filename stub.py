# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import util
import random

from SwingyMonkey import SwingyMonkey

#in order: tree dist, tree top, tree bot, monkey vel, monkey top, monkey bot
def format_into_tuple(state_dict, gravity):
    # return (state_dict['tree']['dist']//5, state_dict['tree']['top']//5,\
    #     state_dict['tree']['bot']//5, state_dict['monkey']['vel']//5,\
    #     state_dict['monkey']['top']//5, state_dict['monkey']['bot']//5)

    danger_high = state_dict['monkey']['top'] > 350
    danger_low = state_dict['monkey']['bot'] < 50
    return (state_dict['tree']['dist']//100, (state_dict['tree']['top'] - state_dict['monkey']['top'])//100,\
            (state_dict['tree']['bot']-state_dict['monkey']['bot'])//100, gravity, danger_high, danger_low)

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_state_unformat = None
        self.last_action = None
        self.last_reward = None
        self.curr_gravity = 0
        self.cnt = 0
        self.values = util.Counter()
        self.death_causes = {-10:[], -5:[]}


    def reset(self):
        self.last_state  = None
        self.last_state_unformat = None
        self.last_action = None
        self.last_reward = None
        self.curr_gravity = None
        self.cnt = 0

        #print(self.death_causes)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        epsilon = 0.2
        gamma = 1
        learning_rate = 0.3

        # Infer Gravity
        self.cnt += 1
        if self.curr_gravity == None and self.last_state_unformat != None:
            time = (state['tree']['dist'] - self.last_state_unformat['tree']['dist'])/25.
            vel_change = state['monkey']['vel'] - self.last_state_unformat['monkey']['vel']
            gravity = vel_change/time
            if(gravity == 4 or gravity == 1):
                self.curr_gravity = gravity

        # Format state
        if self.curr_gravity == None:
            state_formatted = format_into_tuple(state, 4)
        else:
            state_formatted = format_into_tuple(state, self.curr_gravity)

        if self.last_action is not None:
            last_Q = self.values[self.last_state, self.last_action]
            if self.values[state_formatted, 1] > self.values[state_formatted, 0]:
                next_val = self.values[state_formatted, 1]
                next_action = 1
            else:
                next_val = self.values[state_formatted, 0]
                next_action = 0
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



        # use tree positions in game data (SwingMonkey.trees) if we move past
        # the next stump

        # # State Calculations - Jump
        # if self.last_state != None and self.last_state['tree']['dist'] < state['tree']['dist']:
        #     new_score = self.last_state['score'] + 1
        #     new_dist = state['tree']['dist']
        #     new_top = state['tree']['top']
        #     new_bot = state['tree']['bot']
        # elif self.last_state == None:
        #     new_score = 0
        #     new_dist = state['tree']['dist']
        #     new_top = state['tree']['top']
        #     new_bot = state['tree']['bot']
        # else:
        #     new_score = self.last_state['score']
        #     new_dist = self.last_state['tree']['dist']
        #     new_top = self.last_state['tree']['top']
        #     new_bot = self.last_state['tree']['bot'] 
        # jump_state = {
        #     'score': new_score,
        #     'tree': {
        #         'dist': new_dist,
        #         'top': new_top,
        #         'bot': new_bot,
        #     },
        #     'monkey': {
        #         'vel': 15,
        #         'top': state['monkey']['top'] + 15,
        #         'bot': state['monkey']['bot'] + 15,
        #     }
        # }

        # # State Calculation - Stay
        # stay_state = {
        #     'score': new_score,
        #     'tree': {
        #         'dist': new_dist,
        #         'top': new_top,
        #         'bot': new_bot,
        #     },
        #     'monkey': {
        #         'vel': state['monkey']['vel'],
        #         'top': state['monkey']['top'] - state['monkey']['vel'],
        #         'bot': state['monkey']['bot'] - state['monkey']['vel'],
        #     }
        # }

        # !!!!!!!!!!!!!!!
        # Potential issue: score is included in the state but probably shouldnt affect
        # our decision

        # if self.last_action == 0:
        #     state = stay_state
        # else:
        #     state = jump_state

        # jump_value = (self.values[(format_into_tuple(state),1)],1)
        # stay_value = (self.values[(format_into_tuple(state),0)],0)
        # qValues = [jump_value,stay_value]
        # if util.flipCoin(epsilon):
        #     new_action = random.choice([0,1])
        # else:
        #     new_action = max(qValues)[1]
        

        # # Update qValues
        # derivative = self.values[(format_into_tuple(self.last_state), self.last_action)]\
        #             - (self.last_reward - gamma * self.values[(format_into_tuple(self.state), new_action)])
        # self.values[format_into_tuple(self.last_state), self.last_action] -= learning_rate * derivative

        # self.last_action = new_action
        # self.last_state = state
        # return self.last_action

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


if __name__ == '__main__':

    it = 20
    avg_scores = []
    for i in range(it):
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, 100, 1)

        # Score Avg
        avg_scores.append(np.mean(hist))

        # Save history.
        np.save('hist',np.array(hist))

print(avg_scores)
print(np.mean(avg_scores))


