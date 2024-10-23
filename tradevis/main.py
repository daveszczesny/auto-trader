"""
TradeVis

TradVis is a tool for visualizing agent trade data. It is designed to be used with brooksai
"""
import os
import pygame

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from views.logpanel.log import LogPanel
from views.logpanel.stage import Stage

def main():

    pygame.init()
    pygame.font.init()
    WIDTH, HEIGHT = 800, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    pygame.display.set_caption("TradeVis")

    main_stage = Stage(
        0,0, WIDTH, HEIGHT
    )

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            main_stage.update(event)


        main_stage.draw(screen)

        pygame.display.flip()
        clock.tick(120)
    pygame.quit()


"""

log file contain a structure

logs/
2024-01-01T00:00: {
    run1: [log01.csv, log02.csv, etc... ]
}

given this structure we can iterate down the parent directories to get the log files

"""




def get_log_files():

    log_path = 'logs/'

    if not os.path.exists(log_path):
        print('No logs found')
        return
    
    log_files = os.listdir(log_path)

    run_files = []
    for log_file in log_files:
        runs = os.listdir(log_path + log_file)

        for run in runs:
            run_files = os.listdir(log_path + log_file + '/' + run)


    for file in run_files:
        print(file)

    return run_files


if __name__ == "__main__":
    main()