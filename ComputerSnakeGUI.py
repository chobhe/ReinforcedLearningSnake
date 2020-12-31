import pygame as pg
import os
import random

#play the game without input and get input from a player class
##assume we have the directions to move

class ComputerSnake:
    def __init__(self):
        #initiate pygame
        pg.init()
        self.win = pg.display.set_mode((0,0), pg.FULLSCREEN)
        pg.display.set_caption("Snake")

        #load in our images
        self.apple = pg.image.load(os.getcwd() + '/apple.jpg')
        self.snake_head = pg.image.load(os.getcwd() + '/snake.jpg')

        #draw board
        self.surface = pg.Surface((720,720))
        self.surface.fill((255,255,255))
        self.fruit_locations = []



        #self.start_game()

    def start_game(self):
        #directions 1:left 2:right 3:up 4:down
        self.dir = 2
        #amount of apples collected
        self.score = 0

        self.status = True
        # if status is false our snake is dead

        #head location is top right corner of head, for each body each list is the center of the part
        self.PART_RADIUS = 20
        self.snake_locations = [[0,0]]

        #constant first apple position
        self.apple_num = 0
        self.apple_pos = (40,0)
        self.generate_snake()
        #self.update()

    def wipe(self):
        return
    def generate_apple(self):

        self.apple_num+=1
        self.apple_pos = self.fruit_locations[self.apple_num]


    def generate_snake(self):

        #blit the snakehead onto the surface, notice that it isn't the canvas
        self.surface.blit(pg.transform.scale(self.snake_head,(self.PART_RADIUS*2, self.PART_RADIUS*2)), tuple(self.snake_locations[0]))
        #draw circle at the part location, the coordinates are the center of the circle
        for part in self.snake_locations[1:]:
            pg.draw.circle(self.surface, (0,255,0), (part[0],part[1]), self.PART_RADIUS)


    def move_snake(self):
        head = self.snake_locations[0]

        #each part must go to the position of the old part, so the old part is the new location
        new_location = (self.snake_locations[0][0]+self.PART_RADIUS, self.snake_locations[0][1] + self.PART_RADIUS)

        #move head according to the direction given
        if self.dir == 1:
            head[0]-= self.PART_RADIUS*2
        elif self.dir == 2:
            head[0] += self.PART_RADIUS*2
        elif self.dir == 3:
            head[1] -= self.PART_RADIUS*2
        else:
            head[1] += self.PART_RADIUS*2

        #move parts in pursuit of the head
        for part in self.snake_locations[1:]:
            current_location = part[:]
            part[0] = new_location[0]
            part[1] = new_location[1]

            new_location = current_location

    def change_dir(self, new_dir):
        #change direction from Player class
        self.dir = new_dir

    def check_collision(self):
        #check to see if we hit the apple
        if self.apple_pos == tuple(self.snake_locations[0]):
            self.score+=1
            return True
        else:
            return False

    def display_score(self):
        self.font = pg.font.SysFont('Arial', 20)
        self.rect = pg.draw.rect(self.win, (255,255,255), (100, 50, 100, 50))
        self.win.blit(self.font.render('score: '+ str(self.score), True, (0,0,0)), (110, 55))



    def check_life(self):
        #see if the snake is still alive
        if [self.snake_locations[0][0]+self.PART_RADIUS, self.snake_locations[0][1] + self.PART_RADIUS] in self.snake_locations[1:]:
            self.status = False
        elif self.snake_locations[0][0]<0 or self.snake_locations[0][0]>=720 or self.snake_locations[0][1]<0 or self.snake_locations[0][1]>=720:
            self.status = False

        else:
            self.status = True


    def update(self):
        #update each movement every few ms

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        #clear the canvas
        self.surface.fill((255,255,255))
        #draw the snake
        self.generate_snake()

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
            self.generate_apple()
            self.display_score()

        else:
            # if it's not check that it's still in bounds and not hitting itself
            self.check_life()
            if self.status == True:
                self.move_snake()
            else:
                self.wipe()


        #blit everything from the surface onto the main canvas
        self.surface.blit(pg.transform.scale(self.apple,(self.PART_RADIUS*2, self.PART_RADIUS*2)),self.apple_pos)

        self.win.blit(self.surface,(400,100))
        pg.display.update()
