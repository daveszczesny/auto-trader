from typing import Tuple, List, Any

import pygame

class Panel:

    surface: pygame.Surface = None
    


    def __init__(self, 
                 width: int,
                 height: int,
                 color: Tuple[int, int, int] = (0, 0, 0)):

        self.components: List[Any] = []
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

        self.color = color

    def add_component(self, component: Any):
        if not self.components:
            self.components = []

        self.components.append(component)

    def add_components(self, *components: Any):
        for component in components:
            self.add_component(component)

    def update(self, event: pygame.event.Event):
        for component in self.components:
            component.update(event)

    def draw(self, screen: pygame.Surface, x: int = 0, y: int = 0):
        self.surface.fill(self.color)

        for component in self.components:
            component.draw(self.surface)

        screen.blit(self.surface, (x, y))
