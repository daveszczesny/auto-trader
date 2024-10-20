"""
TradeVis

TradVis is a tool for visualizing agent trade data. It is designed to be used with brooksai
"""

import pygame
import dask.dataframe as dd
from shapes.shapes import Candlestick

def main():

    pygame.init()
    pygame.font.init()
    WIDTH, HEIGHT = 1400, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    pygame.display.set_caption("TradeVis")

    data = dd.read_csv("resources/training_data_5min.csv")
    data = data.compute()

    data = data[["bid_open", "bid_close", "bid_high", "bid_low"]]

    candlesticks = []

       # Initial y position
    initial_y = 500

    canvas = pygame.Surface((1000, 800))

    # for i in range(len(data)):
    #     if i == 0:
    #         y_position = initial_y
    #     else:
    #         y_position += (data.iloc[i]["bid_open"] - data.iloc[i-1]["bid_close"]) * 100_000

    #     candlestick = Candlestick(
    #         x= 200 + (i * 20),
    #         y=y_position,
    #         open=data.iloc[i]["bid_open"],
    #         close=data.iloc[i]["bid_close"],
    #         high=data.iloc[i]["bid_high"],
    #         low=data.iloc[i]["bid_low"]
    #     )

    #     candlesticks.append(candlestick)

    candlestick = Candlestick(
        x=20,
        y=400,
        open=1.8000,
        close=1.8002,
        high=1.8005,
        low=1.7999
    )



    while running:
        screen.fill((0, 0, 0))
        canvas.fill((20, 20, 20))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        candlestick.draw(canvas)
        screen.blit(canvas, (0, 0))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()