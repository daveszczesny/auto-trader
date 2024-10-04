import os
from datetime import datetime


DEBUG: str = "DEBUG"
TEST: str = "TEST"
PROD: str = "PROD"


class Logger:
    root_dir: str = "logs/"
    root_run: str = root_dir + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    def __init__(self, mode: str = 'debug'):
        self.run: int = 1
        self.mode = mode.upper()

        # Create the root directory if it doesn't exist
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        
        # Create the root run directory if it doesn't exist
        if not os.path.exists(self.root_run):
            os.makedirs(self.root_run)
        else:
            # If the root run directory exists, determine the run number
            self.run = len(os.listdir(self.root_run)) + 1
        
        # Create the directory for the current run
        os.makedirs(f"{self.root_run}{self.run:03d}/")

        # Set the current log file path
        self.current_file = f"{self.root_run}{self.run:03d}/log_" + str(len(os.listdir(f"{self.root_run}{self.run:03d}/")))

    def log_debug(self, message: str):
        if self.mode != DEBUG:
            return
        with open(self.current_file + ".txt", "a") as file:
            file.write(f"{message}\n")

    def log_test(self, message: str):
        if self.mode != TEST:
            return
        if os.path.exists(self.current_file + ".csv"):
            with open(self.current_file + ".csv", "a") as file:
                file.write(f"{message}\n")

    def create_new_log_file(self):
        self.current_file = f"{self.root_run}{self.run:03d}/log_" + str(len(os.listdir(f"{self.root_run}{self.run:03d}/")))

        if self.mode == TEST:
            self.create_csv("step, action, trades open, trade size, price, low, high, balance, unrealized profit")

    def create_csv(self, header: str):
        # Run at the start of a new csv file
        open(self.current_file + ".csv", "w").close()
        with open(self.current_file + ".csv", "a") as file:
            file.write(f"{header}\n")

