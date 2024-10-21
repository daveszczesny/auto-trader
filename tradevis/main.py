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

from views.logs import log

def main():

    pygame.init()
    pygame.font.init()
    WIDTH, HEIGHT = 800, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    pygame.display.set_caption("TradeVis")

    file_list_canvas = pygame.Surface((300, HEIGHT))
    info_panel_canvas = pygame.Surface((500, HEIGHT))

    log_struct = get_log_files_structure()

    log.generate_log_buttons(log_struct)
    


    while running:
        screen.fill((0, 0, 0))
        file_list_canvas.fill((20, 20, 20))
        info_panel_canvas.fill((50, 50, 50))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for btn in log.buttons:
                btn.update(event)

                if btn and btn.info_component is not None:
                    btn.info_component.update(event, info_panel_canvas)



        for btn in log.buttons:
            btn.draw(file_list_canvas, info_panel_canvas)

            if btn and btn.info_component is not None:
                    btn.info_component.draw(info_panel_canvas)

        screen.blit(file_list_canvas, (0, 0))
        screen.blit(info_panel_canvas, (300, 0))
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

def get_log_files_structure() -> dict:
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