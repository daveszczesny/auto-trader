import os
import struct
import lzma
import pandas as pd
from datetime import datetime
from src.servies.logger import logger
from src.utils.constants import FAILED_CONVERSIONS_FILE

def bi5_to_csv(from_directory: str, directory: str, aggregation: str = '1min'):

    files_to_convert: list | None = os.listdir(from_directory)

    if not files_to_convert:
        logger.log_error("No files to convert found in data directory")
        return
    
    # Clear the failed conversions file
    open(FAILED_CONVERSIONS_FILE, 'w').close()

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    files_converted: int = 0

    start_time = datetime.now()
    for file in files_to_convert:
        if not file.endswith('.bi5'):
            continue

        try:
            _bi5_to_csv(from_directory, directory, file, aggregation=aggregation)
            files_converted += 1
            logger.log_state(f"Converted {files_converted} files out of {len(files_to_convert)}")
        except Exception as _:
            # We are already handling exception within _bi5_to_csv function
            pass

    time_taken = datetime.now() - start_time
    logger.log_info(f"Time taken to convert {files_converted} files: {(int) (time_taken.total_seconds())} seconds")

    failed_conversions = open(FAILED_CONVERSIONS_FILE).read().split('\n')
    if failed_conversions:
        logger.log_error(f"Failed to convert {len(failed_conversions)} files. Retrying failed conversions")
        for filename in failed_conversions:
            if not filename:
                continue
            try:
                _bi5_to_csv(from_directory, directory, filename, aggregation=aggregation)
            except Exception as _:
                logger.log_error(f"Failed to convert {filename} again.")



def _bi5_to_csv(from_directory: str, directory: str, filename: str, aggregation: str = '1min'):
    """
    Decode bi5 file and save as csv
    """

    if aggregation not in ['1min', '5min', '15min', '30min', '1H']:
        raise ValueError("Invalid aggregation parameter. Must be one of '1min', '5min', '15min', '30min', '1H'")

    year, month, day, hour = filename.split('_')
    hour = hour.split('.')[0]

    chunk_size = struct.calcsize('>3i2f')
    data = []

    try:
        with lzma.open(f"{from_directory}/{filename}") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                data.append(struct.unpack('>3i2f', chunk))

        df = pd.DataFrame(data)
        df.columns = ['timestamp', 'ask', 'bid', 'ask_volume', 'bid_volume']
        df.ask = df.ask / 100_000
        df.bid = df.bid / 100_000

        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
        df.set_index('timestamp', inplace=True)

        # Resample the data based on the aggregation parameter
        df_resampled = df.resample(aggregation).agg({
            'ask': ['first', 'max', 'min', 'last'],
            'bid': ['first', 'max', 'min', 'last'],
            'ask_volume': 'sum',
            'bid_volume': 'sum'
        }).reset_index()

        # Flatten the MultiIndex columns
        df_resampled.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_resampled.columns.values]

        # Rename the columns appropriately
        df_resampled.rename(columns={
            'ask_first': 'ask_open',
            'ask_max': 'ask_high',
            'ask_min': 'ask_low',
            'ask_last': 'ask_close',
            'bid_first': 'bid_open',
            'bid_max': 'bid_high',
            'bid_min': 'bid_low',
            'bid_last': 'bid_close'
        }, inplace=True)

        # format timestamp
        date_hour: str = f"{year}-{int(month) + 1}-{day}T{hour}:"
        df_resampled['timestamp'] = date_hour + df_resampled['timestamp'].dt.strftime('%M')

        # Save the resampled data to CSV
        df_resampled.to_csv(f'{directory}/{year}_{int(month) + 1}_{day}_{hour}.csv', index=False)

    except Exception as e:
        logger.log_exception(f"Failed to convert {filename}", exec_info=e)
        _update_failed_conversions(filename)


def _update_failed_conversions(filename: str):
    """
    Update failed conversions
    """
    open(FAILED_CONVERSIONS_FILE, 'a').write(filename + '\n')