import pygame

class Card:

    def __init__(self,
                 x:int,
                 y:int,
                 width:int,
                 height:int,
                 text:str):
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

        self.rect = pygame.Rect(x, y, width, height)

        self.button = Button()

    def set_button_callback(self, callback):
        self.button.onclick = callback

    def update(self, event):
        self.button.update(event)

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            self.rect
        )

        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class Button:

    def __init__(
            self, x: int, y: int, width: int, height: int, text: str
    ):
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

        self.rect = pygame.Rect(x, y, width, height)


        self.onclick = None

    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.onclick:
                    self.onclick()

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            self.rect
        )

        font = pygame.font.Font(None, 16)
        text_surface = font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

