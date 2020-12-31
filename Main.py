import Snake
import Player
import pygame as pg
import ComputerEnv
import pickle

class Main:
    def __init__(self):
        self.game = Snake.Snake()
        self.opening_screen()

    def play_game(self):
        Player.Player(self.game)

    def computer_game(self):
        #simulate
        #feed into the GUI
        pg.display.quit()
        pg.quit()
        instance = ComputerEnv.Computer()
        instance.train_eval()




    def opening_screen(self):
        run = True
        while run:
            pg.display.update()

            #draw on button for player game
            self.font = pg.font.SysFont('Arial', 50)
            self.rect = pg.draw.rect(self.game.win, (255,255,255), (400, 100, 720, 300))
            self.game.win.blit(self.font.render('Player Version', True, (0,0,0)), (600, 220))

            #draw on button for computer game
            self.fonttwo = pg.font.SysFont('Arial', 50)
            self.recttwo = pg.draw.rect(self.game.win, (255,255,255), (400, 500, 720, 300))
            self.game.win.blit(self.font.render('Computer Version', True, (0,0,0)), (570, 620))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.MOUSEBUTTONUP:
                    #get mouse position as a tuple
                    self.mouse_location = pg.mouse.get_pos()
                    if self.mouse_location[0] <=1120 and self.mouse_location[0] >=400 and self.mouse_location[1]<=400 and self.mouse_location[1] >=100:
                        self.play_game()
                        run = False

                    if self.mouse_location[0] <=1120 and self.mouse_location[0] >=400 and self.mouse_location[1]<=800 and self.mouse_location[1] >=500:
                        self.computer_game()
                        run = False








Main()
