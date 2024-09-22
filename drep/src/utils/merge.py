
import os
from src.servies.logger import logger
from datetime import datetime, timedelta

PATH_RESOURCES = '../environments/resources/'

def merge_csv_files(from_directory: str, destination_file: str | None, start_date: str, end_date: str):
    """
    Merge csv files into one file
    """

    # Set default filename
    if destination_file is None:
        destination_file = 'data.csv'

    if os.path.exists(PATH_RESOURCES + destination_file):
        os.remove(PATH_RESOURCES + destination_file)
    
    # create file
    open(PATH_RESOURCES + destination_file, 'w').close()

    logger.log_info("Merging csv files...")

    total_files_to_merge: int = len(os.listdir(from_directory))
    files_merged: int = 0
    files_skipped: int = 0

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    first_file: bool = True

    current_dt = start_dt

    while current_dt <= end_dt:
        for hour in range(24):
            filename: str = f"{current_dt.year}_{current_dt.month}_{current_dt.day:02d}_{hour:02d}.csv"
            filepath: str = f"{from_directory}/{filename}"

            if not os.path.exists(filepath):
                files_skipped += 1
                continue

            lines = []
            with open(filepath, 'r') as file:
                lines = file.readlines()
                
                # Remove the first line which is the header
                lines = lines if first_file else lines[1:]
                first_file = False

                lines_to_write = []

                for line in lines:
                    # Skip lines with no values
                    if ',,,' not in line:
                        lines_to_write.append(line)

            with open(PATH_RESOURCES + destination_file, 'a') as file:
                for line in lines_to_write:
                    file.write(line)
                files_merged += 1

            
            logger.log_state(f"Merged {files_merged} files out of {total_files_to_merge}, "
                             f"Skipped: {files_skipped} files")
        current_dt += timedelta(days=1)
    print('\n')

