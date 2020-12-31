import pygame


class Player:
    def __init__(self, game):
        self.game = game
        self.play_game()

    def play_game(self):
        self.game.start_game()
        first_event = pygame.USEREVENT+1
        #this event repeat every 50ms
        pygame.time.set_timer(first_event,100)


        run = True
        while run:
            pygame.display.update()
            for event in pygame.event.get():
                self.press_keys()
                #update game every time the event triggers
                if event.type == first_event:
                    self.game.update()
                if event.type == pygame.QUIT:
                    run = False
            self.press_keys()




        pygame.quit()


    def press_keys(self):
        keys = pygame.key.get_pressed()
        correspondance = {1:keys[pygame.K_RIGHT] or keys[pygame.K_d], 2:keys[pygame.K_LEFT] or keys[pygame.K_a], 3:keys[pygame.K_DOWN] or keys[pygame.K_s], 4:keys[pygame.K_UP] or keys[pygame.K_w]}
        if not correspondance[self.game.dir]:

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.game.change_dir(1)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.game.change_dir(2)
            elif keys[pygame.K_UP] or keys[pygame.K_w]:
                self.game.change_dir(3)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.game.change_dir(4)
