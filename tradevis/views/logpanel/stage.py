from typing import Optional

import pygame

from tradevis.objects.drawer import Drawer
from tradevis.views.logpanel.log import LogPanel, LogViewPanel


class Stage:

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.log_panel = LogPanel(
            width = 300,
            height = height,
            color = (20, 20, 20)
        )

        self.log_drawers = self.log_panel.components

        self.log_viewer = LogViewPanel(width - 300, height, color = (90,90,90))

    def update(self, event: pygame.event.Event) -> None:
        self.log_panel.update(event)
        self.log_viewer.update(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            component = self._get_active_button()
            if component:
                self.log_viewer.update_view(component, event)

    def draw(self, screen: pygame.Surface) -> None:
        self.log_panel.draw(self.log_panel.surface)
        self.log_viewer.draw(self.log_viewer.surface)

        screen.blit(self.log_panel.surface, (self.x, self.y))
        screen.blit(self.log_viewer.surface, (self.x + 300, self.y))

    def _get_active_button(self) -> Optional[Drawer]:
        active_components = [c for c in self.log_drawers if c.is_open]

        if active_components:
            return active_components[0]
        
        return None