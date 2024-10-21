from typing import Optional, Callable

import pygame
from pygame import Rect

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Rectangle:
    def __init__(self, x, y, width, height, color):
        self.rect = Rect(x, y, width, height)
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Line:
    def __init__(self, x1, y1, x2, y2, color):
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.color = color

    def draw(self, screen):
        pygame.draw.line(screen, self.color, self.start, self.end)

class Candlestick:
    def __init__(self, x: int, y: int, open: float, close: float, high: float, low: float):
        self.candle_width = 10
        self.x = x
        self.y = y
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.scaling_factor = 100_000

        if self.open < self.close:
            self.color = (0, 255, 0) # Green
        else:
            self.color = (255, 0, 0) # Red

        self.height = abs(self.open - self.close) * self.scaling_factor


    def create_candlestick(self):

        body_top = self.y
        body_length =  abs(self.open - self.close) * self.scaling_factor

        self.body = Rectangle(self.x,
                              body_top,
                              self.candle_width,
                              body_length,
                              self.color)



        # self.body = Rectangle(self.x,
        #                        self.y + self.height if self.close < self.open else self.y,
        #                        self.candle_width,
        #                        self.height,
        #                        self.color)


        length = abs(self.high - self.low) * self.scaling_factor
        top = abs(self.high - max(self.open, self.close)) * self.scaling_factor

        self.tail_line = Line(self.x + self.candle_width // 2,
                            self.y - top,
                            self.x + self.candle_width // 2,
                            self.y - length,
                            self.color)

    def draw(self, screen):
        self.create_candlestick()
        self.body.draw(screen)
        # self.tail_line.draw(screen)


"""
TODO finish this
"""
class Drawer:
    def __init__(self,
                 x,
                 y,
                 width,
                 height,
                 color,
                 text):
        self.rect = Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, 20)

        self.closed = True
        
        self.sub_components = []

    def open_drawer(self):
        self.closed = False

    def update(self, event):

        if not self.closed:
            for component in self.sub_components:
                component.update(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.open_drawer()

    def draw(self, screen):
        if not self.closed:
            for component in self.sub_components:
                component.draw(screen)


class Button:
    def __init__(self, x, y, width, height, color, text, onclick: Optional[Callable] = None):
        self.rect = Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, 20)
        self.onclick = onclick
        self.info_component = None

    def draw(self, screen, screen2):

        pygame.draw.rect(screen, self.color, self.rect)
        text = self.font.render(self.text, True, (0,0,0))
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

        if self.info_component is not None:
            self.info_component.draw(screen2)

    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.onclick is not None:
                    self.onclick()


class InfoComponent:
    def __init__(self,
                 x,
                 y,
                 width,
                 height,
                 background_color,
                 text_color,
                 font_size = 20,
                 text=""):
        
        self.rect = Rect(x, y, width, height)
        self.background_color = background_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, font_size)
        self.text = text
        self.button = Button(
            x=200,
            y=500,
            width = 100,
            height = 70,
            color = (80, 100, 20),
            text = 'Run',
            onclick= self._run_onclick
        )

        self.anim = None
        self.paused = False
        self.current_frame = 0


    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, self.rect)
        lines = self.text.split('\n')
        y_offset = self.rect.top
        for line in lines:
            text_surface = self.font.render(line, True, self.text_color)
            text_rect = text_surface.get_rect(midtop=(self.rect.centerx, y_offset))
            screen.blit(text_surface, text_rect)
            y_offset += text_surface.get_height()

        self.button.draw(screen, None)

    def update(self, event, info_panel):
        if event.type == pygame.MOUSEBUTTONDOWN:

            rect_x = self.rect.x + self.button.rect.x + 300
            rect_y = self.rect.y + self.button.rect.y

            new_rect = Rect(rect_x, rect_y, self.button.rect.width, self.button.rect.height)


            if event.button == 4: # Scroll up
                self.current_frame = max(0, self.current_frame - 10)
                self.paused = True
                self._update_anim(self.current_frame, self.data, self.ax)
            elif event.button == 5: # Scroll down
                self.current_frame = min(len(self.data) - 1, self.current_frame + 10)
                self.paused = True
                self._update_anim(self.current_frame, self.data, self.ax)
            elif event.button == 1: # left click
                self.paused = not self.paused

            if new_rect.collidepoint(event.pos):
                if self.button.onclick is not None:
                    self.button.onclick()

    def _run_onclick(self):
        print('button clicked')
        data = pd.read_csv('resources/training_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        if data.empty:
            return
        
        data = data.rename(columns={'bid_open': 'Open', 'bid_high': 'High', 'bid_low': 'Low', 'bid_close': 'Close'})
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self._update_anim, frames=len(data), fargs=(data, ax), interval=100)
        plt.show()
    
    def _update_anim(self, num, data, ax):
        ax.clear()
        if num > 0:
            start = max(0, num - 50)
            mpf.plot(data.iloc[start:num], type='candle', ax=ax, style='charles')