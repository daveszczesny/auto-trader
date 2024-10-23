
from typing import List

from tradevis.shapes.shapes import Candlestick
from tradevis.views.chart.data import data


"""

Chart renderer for TradeVis

General Idea:

We want to only have around 50ish candlesticks at one time.
I want to use entity pooling to manage the candlesticks.
We will have a list of candlesticks that are currently on the screen.
If we need to render new candlesticks, we will check which candlesticks
    are off the screen and use those objects

"""


class Chart:

    data = data

    def __init__(self, canvas):

        self.data = data
        # Keep this ordered by candlestick x coordinate
        self.candlesticks: List[Candlestick] = []
        self.candlestick_pool_limit = 50

        # Used as tracker for data
        self.current_step = 0

        self.global_y = 500

        self.canvas = canvas


        self.load_candlesticks()

    def render(self):
        """
        Draws the candlesticks
        """

        for candlestick in self.candlesticks:
            if candlestick.x < 0 or candlestick.x > self.canvas.get_width():
                pass
            candlestick.draw(self.canvas)

    def update(self):
        for candlestick in self.candlesticks:
            dx = -8

            candlestick.x += dx

        self.load_next_candlesticks()


    def load_candlesticks(self):
        """
        Draws the inital candlesticks
        """
        for i in range(self.candlestick_pool_limit):

            if self.current_step == 0:
                y_pos = self.global_y
            else:
                y_pos = self.global_y + (
                    self.data.iloc[self.current_step]['bid_open']
                    - self.data.iloc[self.current_step-1]['bid_close']) * 100_000

            candlestick = Candlestick(
                x=0 + (i*20),
                y=y_pos,
                open=self.data.iloc[self.current_step]['bid_open'],
                close=self.data.iloc[self.current_step]['bid_close'],
                high=self.data.iloc[self.current_step]['bid_high'],
                low=self.data.iloc[self.current_step]['bid_low']
            )

            self.candlesticks.append(candlestick)

            self.current_step += 1

    def load_next_candlesticks(self):
        """
        Takes candlesticks that are off the screen and moves them to the right
        """

        for candlestick in self.candlesticks:
            if candlestick.x < -50:
                new_y = self.global_y + (
                    self.data.iloc[self.current_step]['bid_open']
                      - self.data.iloc[self.current_step-1]['bid_close']) * 100_000

                new_x = max([candlestick.x for candlestick in self.candlesticks]) + 20

                candlestick.x = new_x
                candlestick.y = new_y
                candlestick.open = self.data.iloc[self.current_step]['bid_open']
                candlestick.close = self.data.iloc[self.current_step]['bid_close']
                candlestick.high = self.data.iloc[self.current_step]['bid_high']
                candlestick.low = self.data.iloc[self.current_step]['bid_low']

        self.current_step += 1

