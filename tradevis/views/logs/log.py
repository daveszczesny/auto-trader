
from collections import deque
from tradevis.shapes.shapes import Button, InfoComponent

import random

buttons = []

def generate_log_buttons(logs_: dict) -> list:

    print(f' this should be two: {len(logs_)}')

    counter = 0
    for i, log in enumerate(logs_):
        # Runs
        for j, run in enumerate(logs_[log]):
            print(f' Runs: {len(logs_[log])}')
            # Log files
            for k, log_file in enumerate(logs_[log][run]):
                print(f' Log files: {len(logs_[log][run])}')
                btn = Button(
                    x = 0,
                    y = 50 + (counter * 80),
                    width = 300,
                    height = 75,
                    color = (255, 255, 255),
                    text = f'{log} / {run} / {log_file}'
                )
                btn.onclick = lambda b=btn: button_onclick(b, b.text)

                counter += 1
                buttons.append(btn)

def button_onclick(btn, text):

    for button in buttons:
        if button.info_component:
            button.info_component = None
            break

    btn.info_component = InfoComponent(
        x = 0,
        y = 0,
        width = 500,
        height = 800,
        background_color = (255, 255, 255),
        text_color = (0, 0, 0),
        text = get_log_details(text)
    )


def get_log_details(log_file: str):
    log_path = 'logs/'

    log_, run_, file_ = log_file.split(' / ')
    
    with open(f'{log_path}{log_}/{run_}/{file_}', 'r') as f:
        log_details = deque(f, maxlen=6)

    data = '\n'.join(log_details)

    return data

