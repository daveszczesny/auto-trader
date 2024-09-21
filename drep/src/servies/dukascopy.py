import os
from time import sleep
from datetime import datetime, timedelta
from urllib.error import HTTPError
from urllib.request import urlretrieve

from src.servies.logger import logger
from src.utils.constants import FOREX_URL, MAX_RETRIES, FAILED_DOWNLOADS_FILE


TOTAL_FILES: int = 0
FILES_DOWNLOADED: int = 0
START_TIME: datetime | None = None

def download_bi5_file_between_dates(directory: str, start_date: str, end_date: str):
    """
    Download bi5 files between two dates
    """

    global TOTAL_FILES, FILES_DOWNLOADED, START_TIME

    # Clear the failed downloads file
    open(FAILED_DOWNLOADS_FILE, 'w').close()

    if not directory:
        raise ValueError("Directory not provided")

    if not os.path.exists(directory):
        os.makedirs(directory)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    current_dt = start_dt

    TOTAL_FILES = ((end_dt - start_dt).days + 1) * 24
    START_TIME = datetime.now()

    while current_dt <= end_dt:
        download_file(directory, current_dt.year, current_dt.month, current_dt.day)
        current_dt += timedelta(days=1)

    # Retry failed downloads
    failed_downloads = open(FAILED_DOWNLOADS_FILE).read().split('\n')

    if not failed_downloads:
        return

    print('\n')
    logger.log_info(f"Retrying {len(failed_downloads)} failed downloads")

    for url in failed_downloads:
        download_file_from_url(directory, url)

    print('\n')



def download_file(directory: str, year: int, month: int, day: int):
    """
    Download bi5 file from Dukascopy given a year, month and day
    """
    global FILES_DOWNLOADED

    month: str = f"{month-1:02d}"
    day: str = f"{day:02d}"

    URL: str = f'{FOREX_URL}/{year}/{month}/{day}'

    for hour in range(24):
        HOUR_URL = f'{URL}/{hour:02d}h_ticks.bi5'

        FILES_DOWNLOADED += download_file_from_url(directory, HOUR_URL)
        logger.log_state(f"Downloaded {FILES_DOWNLOADED} files out of {TOTAL_FILES}. "
                         f"Time taken: {(datetime.now() - START_TIME).seconds} seconds")


def download_file_from_url(directory: str, url: str) -> int:
    """
    Download a file from a given URL
    """

    # url format
    # https://datafeed.dukascopy.com/datafeed/EURUSD/2020/00/00/00h_ticks.bi5'

    if not url:
        return 0

    save_file: str = url.split('EURUSD/')[1].replace('/', '_').replace('h_ticks.bi5', '.bi5')

    for attempt in range(MAX_RETRIES):

        try:
            urlretrieve(url, f"{directory}/" + save_file)
            return 1
        except HTTPError as e:
            logger.log_warning(f"\nFailed to download file from {url} on attempt {attempt + 1} / {MAX_RETRIES}")
            sleep(50)
        except Exception as e:
            logger.log_error(f"\nError downloading file. {url}. Exception is {repr(e)}")
            _update_failed_downloads(url)
            break

    return 0


def _update_failed_downloads(url: str):
    """
    Update the list of failed downloads
    """
    logger.log_error(f"\nFailed to download file from {url}")
    open(FAILED_DOWNLOADS_FILE, 'a').write(url + '\n')