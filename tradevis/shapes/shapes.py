
import pygame
from pygame import Rect

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
        self.body = Rectangle(self.x,
                               self.y + self.height if self.close < self.open else self.y,
                               self.candle_width,
                               self.height,
                               self.color)

        self.high_tail = Line(self.x + self.candle_width//2,
                              self.y + self.height if self.close < self.open else self.y,
                              self.x + self.candle_width//2,
                              self.y - (self.high - self.open) * self.scaling_factor,
                              self.color)

        self.low_tail = Line(self.x + self.candle_width//2,
                             self.y + (abs(self.open-self.close) * self.scaling_factor),
                             self.x + self.candle_width//2,
                             self.y + (abs(self.close - self.low) * self.scaling_factor),
                             self.color
                            )

    def draw(self, screen):
        self.create_candlestick()
        self.body.draw(screen)
        self.high_tail.draw(screen)
        self.low_tail.draw(screen)