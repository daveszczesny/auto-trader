"""
TradeVis

TradVis is a tool for visualizing agent trade data. It is designed to be used with brooksai
"""

import pygame
import dask.dataframe as dd
from shapes.shapes import Rectangle, Line, Candlestick

def main():

    pygame.init()
    pygame.font.init()
    WIDTH, HEIGHT = 1400, 1000
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    pygame.display.set_caption("TradeVis")

    data = dd.read_csv("resources/training_data_5min.csv")
    data = data.compute()

    use_data = data[["bid_open", "bid_close", "bid_high", "bid_low"]].head(1000)

    candlesticks = []

       # Initial y position
    initial_y = 500

    for i in range(len(use_data)):
        if i == 0:
            y_position = initial_y
        else:
            y_position += (use_data.iloc[i]["bid_open"] - use_data.iloc[i-1]["bid_close"]) * 100_000

        candlestick = Candlestick(
            x= 200 + (i * 20),
            y=y_position,
            open=use_data.iloc[i]["bid_open"],
            close=use_data.iloc[i]["bid_close"],
            high=use_data.iloc[i]["bid_high"],
            low=use_data.iloc[i]["bid_low"]
        )

        candlesticks.append(candlestick)


    dx, dy = -4, 0

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for candlestick in candlesticks:
            candlestick.draw(screen)
            candlestick.x += dx
            candlestick.y += dy

            if candlestick.y > HEIGHT - 100:
                dy = -2
            elif candlestick.y < 100:
                dy = 2
            else:
                dy=0


        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()