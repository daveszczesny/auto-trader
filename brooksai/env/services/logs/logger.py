import os
from datetime import datetime

class Logger:
    root_dir: str = "logs/"
    root_run: str = root_dir + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    def __init__(self):
        self.run: int = 1

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

    def log(self, message: str):
        with open(self.current_file + ".txt", "a") as file:
            file.write(f"{message}\n")
    
    def create_new_log_file(self):
        self.current_file = f"{self.root_run}{self.run:03d}/log_" + str(len(os.listdir(f"{self.root_run}{self.run:03d}/")))