import pygame


class Drawer:

    def __init__(self,
                 x: int,
                 y: int,
                 width: int,
                 height: int,
                 sub_components: list,
                 text: str = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)

        self.sub_components: list = sub_components
        self.text = text
        self.is_open: bool = False

    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open

            for component in self.sub_components:
                if self.is_open:
                    component.update(event)


    def draw(self, screen):
        if self.is_open:
            # draw opened drawer
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                self.rect
            )
            pygame.draw.circle(
                screen,
                (20, 200, 20),
                (self.x + 10, self.y + self.height // 2),
                5
            )

            for component in self.sub_components:
                component.draw(screen)
        else:
            # draw closed drawer
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                self.rect
            )
            pygame.draw.circle(
                screen,
                (200, 20, 20),
                (self.x + 10, self.y + self.height // 2),
                5
            )

        if self.text:
            font = pygame.font.Font(None, 20)
            text_surface = font.render(self.text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)