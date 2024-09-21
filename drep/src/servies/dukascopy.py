import os
from time import sleep
from datetime import datetime, timedelta
from urllib.error import HTTPError
from urllib.request import urlretrieve

from src.servies.logger import logger
from src.utils.constants import FOREX_URL, MAX_RETRIES, FAILED_DOWNLOADS_FILE


def download_bi5_file_between_dates(directory: str, start_date: str, end_date: str):
    """
    Download bi5 files between two dates
    """

    # Clear the failed downloads file
    open(FAILED_DOWNLOADS_FILE, 'w').close()

    if not os.path.exists(directory):
        os.makedirs(directory)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    current_dt = start_dt

    total_files: int = ((end_dt - start_dt).days + 1) * 24
    files_downloaded = 0
    start_time = datetime.now()

    while current_dt <= end_dt:
        files_downloaded += download_file(directory, current_dt.year, current_dt.month, current_dt.day)
        
        time_taken =  (datetime.now() - start_time).total_seconds()

        logger.log_state(f"Downloaded {files_downloaded} files out of {total_files}. "
                         f"Time taken: {time_taken} seconds")
        current_dt += timedelta(days=1)

    # Retry failed downloads
    failed_downloads = open(FAILED_DOWNLOADS_FILE).read().split('\n')

    if not failed_downloads:
        return

    logger.log_info(f"Retrying {len(failed_downloads)} failed downloads")

    for url in failed_downloads:
        download_file_from_url(directory, url)
    


def download_file(directory: str, year: int, month: int, day: int) -> int:
    """
    Download bi5 file from Dukascopy given a year, month and day
    """

    month: str = f"{month-1:02d}"
    day: str = f"{day:02d}"

    URL: str = f'{FOREX_URL}/{year}/{month}/{day}'

    files_downloaded: int = 0

    for hour in range(24):
        HOUR_URL = f'{URL}/{hour:02d}h_ticks.bi5'

        files_downloaded += download_file_from_url(directory, HOUR_URL)

    return files_downloaded


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
            logger.log_warning(f"Failed to download file from {url} on attempt {attempt + 1} / {MAX_RETRIES}")
            sleep(50)
        except Exception as e:
            logger.log_error(f"Error downloading file. {url}. Exception is {repr(e)}")
            _update_failed_downloads(url)
            break

    return 0


def _update_failed_downloads(url: str):
    """
    Update the list of failed downloads
    """
    logger.log_error(f"Failed to download file from {url}")
    open(FAILED_DOWNLOADS_FILE, 'a').write(url + '\n')