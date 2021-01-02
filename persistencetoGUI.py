import pickle
import ComputerSnakeGUI
import pygame

def input_data_to_GUI():
    start_game = 30
    game = ComputerSnakeGUI.ComputerSnake()
    #list of lists where each list is an individual games set of moves
    moves = pickle_obj[0]
    fruit = pickle_obj[1]
    #end_game = len(moves)
    end_game = 31

    pygame.time.set_timer(pygame.USEREVENT+2,200)
    index = 0

    for i in range(start_game, end_game):
        specific_moves = moves[i]
        specific_fruit = fruit[i]
        #print(specific_fruit)
        #print(specific_moves)
        game.fruit_locations = specific_fruit
        game.start_game()
        run = True
        while run:
            pygame.display.update()
            for _ in pygame.event.get():
                if _.type == pygame.USEREVENT + 2:
                    if index<len(specific_moves):
                        game.change_dir(specific_moves[index])
                        game.update()
                        index+=1
                    else:
                        run = False

                if _.type == pygame.QUIT:
                    pygame.quit()
        index = 0




pickle_in = open('game_simulations.txt', 'rb')
pickle_obj = pickle.load(pickle_in)
input_data_to_GUI()
