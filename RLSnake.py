import random
import numpy as np
import pickle
import math
import abc

import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class RLSnake(py_environment.PyEnvironment):
    def __init__(self, pickle_eval = False):
        self._action_spec = array_spec.BoundedArraySpec((), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec((7,), dtype=np.int32, minimum=[0, -1, -1, 0, 0, 0, 0], maximum=[3, 1, 1, 1, 1, 1, 1], name='observation')
        #state direction:right, apple:right, apple:below, all sides:safe left,right,up, down
        self._state = [1, 1, 0, 0, 1, 0, 1]
        #so it doesn't loop forever
        self.move_limit = 5000
        #update every move to make sure under move limit
        self.num_moves = 0
        self.num_games = 1
        #all the fruit locations that game to later append to all the fruit ever gathered
        self.fruit_locations = []
        #all fruit ever, it's a list of lists where each individual list contains a single games fruit locations
        self.all_fruit = []
        #all moves ever, list of lists where each list contains a single games moves
        self.all_moves = []
        #this is a list that stores all the moves and fruit locations
        self.all = [self.all_moves, self.all_fruit]
        self.eval_game = pickle_eval
        self.new_game()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def new_game(self):

        self.snake_locations = [[0,0]]
        self.apple_pos = (40,0)
        self.PART_RADIUS = 20
        #directions 1:left 2:right 3:up 4:down
        self.dir = 2
        #amount of apples collected
        self.score = 0
        # if status is false our snake is dead
        self.status = True

        # List of all the moves in this game
        self.moves = []

        # add the first apple pos to fruit_locations
        self.fruit_locations = []
        self.fruit_locations.append(self.apple_pos)
        self.num_moves = 0


    def _reset(self):
        # Resets the state to the initial state
        self._state = np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int32)

        self.status = True

        # Writes the moves to the persistence file so we can see what the computer did later.
        # Also prints the score every couple of rounds.
        if (self.eval_game):
            self.all_moves.append(self.moves)
            self.all_fruit.append(self.fruit_locations)
            self.persistence()

            # Prints score at the end of every game
            print("After evaluating for " + str(self.num_games) + " games --> Score: " + str(self.score))

        # Increments num_games because a game just ended.
        self.num_games += 1

        # Resets the parameters of the snake to the initial parameters
        self.new_game()

        # Tells tensor flow to restart with the new state
        return ts.restart(self._state)



    def _step(self, action):
        action +=1
        if self.status == False:
            self.reset()

        elif action >=1 and action <=4:
            self.setDir(action)

        else:
            raise ValueError('Direction must be between 1 and 4')

        reward = 0.0
        #discount rate discounts the value of future rewards, 5 dollars now worth more than 5 dollars a year later

        if self.num_moves < self.move_limit:
            if self.check_collision():
                # if snake is colliding with apple move snake and add on part to end
                tail_part = self.snake_locations[-1][:]
                self.move_snake()
                if len(self.snake_locations) == 1:
                    #because the parts are circles and the head is a rectangle the centers of reference are different. We attach head by left coordinate, top coord,
                    #we attach circles by center coords
                    self.snake_locations.append([tail_part[0]+self.PART_RADIUS, tail_part[1]+self.PART_RADIUS])
                else:
                    self.snake_locations.append(tail_part)

                reward += 10.0
                self.generate_apple()
                fruit_dir = self.fruit_location()
                danger_info = self.danger()
        #        if self.eval_game:
                    #print(self.snake_locations[0])
                    #print(self.apple_pos)

                self._state = np.array([action-1, fruit_dir[0], fruit_dir[1], danger_info[0], danger_info[1], danger_info[2], danger_info[3]], dtype=np.int32)
                return ts.transition(self._state, reward, discount=.97)
            else:
                # if it's not check that it's still in bounds and not hitting itself
                self.check_life()
                if self.status == True:
                    self.move_snake()
                    fruit_dir = self.fruit_location()
                    danger_info = self.danger()
                    self._state = np.array([action-1, fruit_dir[0], fruit_dir[1], danger_info[0], danger_info[1], danger_info[2], danger_info[3]], dtype=np.int32)
                    return ts.transition(self._state, reward, discount=.97)
                elif self.status == False:
                    reward -= 20.0
                    return ts.termination(self._state, reward)
        else:
            self.status = False
            reward -= 20.0
            return ts.termination(self._state, reward)






    def persistence(self):
        with open('game_simulations.txt', 'wb') as sim_moves:
            pickle.dump(self.all, sim_moves)


    def setDir(self, dir):
        correspondance = {1:2, 2:1, 3:4, 4:3}
        if correspondance[int(dir)] != int(self.dir):
            self.dir = dir


    def check_collision(self):
        #check to see if we hit the apple
        if self.apple_pos == tuple(self.snake_locations[0]):
            self.score+=1
            return True
        else:
            return False


    def move_snake(self):

        self.num_moves+=1
        self.moves.append(self.dir)
        head = self.snake_locations[0]

        #each part must go to the position of the old part, so the old part is the new location
        new_location = [self.snake_locations[0][0]+self.PART_RADIUS, self.snake_locations[0][1] + self.PART_RADIUS]

        #move head according to the direction given
        if self.dir == 1:
            head[0]-= self.PART_RADIUS*2
        elif self.dir == 2:
            head[0] += self.PART_RADIUS*2
        elif self.dir == 3:
            head[1] -= self.PART_RADIUS*2
        elif self.dir == 4:
            head[1] += self.PART_RADIUS*2

        #move parts in pursuit of the head
        for part in self.snake_locations[1:]:
            current_location = part[:]
            part[0] = new_location[0]
            part[1] = new_location[1]

            new_location = current_location


    def check_life(self):
        #see if the snake is still alive
        if [self.snake_locations[0][0]+self.PART_RADIUS, self.snake_locations[0][1] + self.PART_RADIUS] in self.snake_locations[1:]:
            self.status = False
        elif self.snake_locations[0][0]<0 or self.snake_locations[0][0]>=720 or self.snake_locations[0][1]<0 or self.snake_locations[0][1]>=720:
            self.status = False

        else:
            self.status = True


    def generate_apple(self):

        self.apple_locations = []
        #all possible apple locations that aren't in the snake's body and on the board
        for i in range(0,720,self.PART_RADIUS*2):
            for j in range(0,720,self.PART_RADIUS*2):
                if [i+self.PART_RADIUS,j+self.PART_RADIUS] not in self.snake_locations[1:] and [i,j]!= self.snake_locations[0]:
                    self.apple_locations.append((i,j))

        #random choice of an apple location
        self.apple_pos = random.choice(self.apple_locations)
        self.fruit_locations.append(self.apple_pos)


    def fruit_location(self):
        fruit_dir = [0,0]
        if self.apple_pos[0]< self.snake_locations[0][0]:
            #apple to the left of head
            fruit_dir[0]-=1

        elif self.apple_pos[0]> self.snake_locations[0][0]:
            #apple to the right of head
            fruit_dir[0]+=1

        elif self.apple_pos[1]< self.snake_locations[0][1]:
            #apple is above snake head
            fruit_dir[1] +=1
        elif self.apple_pos[1]> self.snake_locations[0][1]:
            #apple below snake head
            fruit_dir[1] -=1
        #if self.eval_game:
            #print(fruit_dir)
        return fruit_dir


    def danger(self):
        def check_life(location):
            #checks to see if the location is a danger block is in the form [x,y]
            if [location[0]+self.PART_RADIUS, location[1] + self.PART_RADIUS] in self.snake_locations[1:]:
                return False
            elif location[0]<0 or location[0]>=720 or location[1]<0 or location[1]>=720:
                return False

            else:
                return True

        head = self.snake_locations[0][:]
        left = [head[0]-self.PART_RADIUS*2, head[1]]
        right = [head[0]+self.PART_RADIUS*2, head[1]]
        up = [head[0], head[1]-self.PART_RADIUS*2]
        down = [head[0], head[1]+self.PART_RADIUS*2]

        danger_arr = [int(check_life(left)), int(check_life(right)), int(check_life(up)), int(check_life(down))]
        #if self.eval_game:
        #    print(danger_arr)
        return danger_arr
