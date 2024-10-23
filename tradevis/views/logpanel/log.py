import os
from typing import Tuple
from collections import deque

import pygame

from tradevis.views.panel import Panel
from tradevis.objects.drawer import Drawer
from tradevis.objects.card import Card


class LogPanel(Panel):

    def __init__(self,
                 width: int,
                 height: int,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__(width, height, color)

        self._create_drawers()

    def update(self, event):

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.components:
                combined_rect = self.components[0].rect.copy()
                for component in self.components[1:]:
                    combined_rect.union_ip(component.rect)

                if combined_rect.collidepoint(event.pos):
                    for comp in self.components:
                        comp.is_open = False

            super().update(event)

    def draw(self, screen):
        super().draw(screen)


    def _create_drawers(self):

        counter = 0
        log_struct = _get_log_files_structure()


        for log in log_struct:
            drawer = Drawer(
                        x = 15,
                        y = 10 + (counter * (40 + 15)),
                        width = 250,
                        height = 40,
                        sub_components=[],
                        text = log
                    )

            counter += 1

            self.add_components(drawer)


class LogViewPanel(Panel):

    def __init__(self,
                 width: int,
                 height: int,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__(width, height, color)

        self.log_struct = _get_log_files_structure()
        self.log_components = []

    def update_view(self, component, event):

        assert component.is_open, 'Component is not open'

        print("Received a component to render")

        counter = 0
        for run in self.log_struct[component.text]:
            print("Run:", run)
            for file in self.log_struct[component.text][run]:
                print("file: ", file)
                print("creating object")
                self._create_file_obj(file=f'{component.text}/{run}/{file}',
                                      x=0,
                                      y=0 + (counter * 120),
                                      width=500,
                                      height=100)
                print("object created")
                counter += 1

    def _create_file_obj(self, file, x, y, width, height):
        log, run, file_name = file.split('/')

        if os.path.exists(file):

            file_text = ''
            with open(file, 'r') as f:
                file_text = deque(f, maxlen=6)

            card = Card(
                x, y, width, height, text=file_text
            )

            self.add_component(card)


    def draw(self, screen):
        super().draw(screen, 0, 0)

    def callback(self, file):
        print('button clicked')


def _get_log_files_structure() -> dict:
    log_path = 'logs/'

    if not os.path.exists(log_path):
        print('No logs found')
        return

    log_files = os.listdir(log_path)
    run_files = []

    struct = {}
    for log_file in log_files:
        if not os.path.exists(log_path + log_file):
            continue
        runs = os.listdir(log_path + log_file)

        run_ = {}
        for run in runs:
            run_files = os.listdir(log_path + log_file + '/' + run)
            run_[run] = run_files
        
        struct[log_file] = run_

    return struct