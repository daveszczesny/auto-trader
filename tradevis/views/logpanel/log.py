from typing import Tuple

from tradevis.views.panel import Panel


class LogPanel(Panel):

    def __init__(self,
                 width: int,
                 height: int,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__(width, height, color)

    
    def components(self):
        self._create_drawers()


    
    def _create_drawers(self):
        
