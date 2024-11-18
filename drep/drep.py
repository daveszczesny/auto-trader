"""
“Forex Historical Data Feed :: Dukascopy Bank SA.” Dukascopy Bank SA, 2024, www.dukascopy.com/swiss/english/fx-market-tools/historical-data/
"""



import argparse
import os
import shutil
from drep.src.servies.logger import logger
from drep.src.servies.dukascopy import download_bi5_file_between_dates
from drep.src.utils.convert import bi5_to_csv
from drep.src.utils.merge import merge_csv_files
from drep.src.servies.indicators import csv_to_dataframe, dataframe_to_csv,\
    add_indicator, remove_indicator


os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Process some arguments.')

# Parse arguments for download, convert, merge and clean
parser.add_argument('-d', '--download', type=str,\
                    help='Download data to the specified folder')
parser.add_argument('-s', '--start', type=str,\
                    help='Start date in YYYY-MM-DD format')
parser.add_argument('-e', '--end', type=str,\
                    help='End date in YYYY-MM-DD format')
parser.add_argument('-c', '--convert', type=str,\
                    help='Convert data to the specified folder')
parser.add_argument('-m', '--merge', action='store_true',\
                    help='Merge data')
parser.add_argument('-mf', '--mergefile', type=str,\
                    help='Merge data to the specified filename in the resources folder')
parser.add_argument('-C', '--clean', action='store_true',\
                    help='Clean up download folders')

# Parser arguments for adding, removing indicators
parser.add_argument('-a', '--add', type=str, action='append', help='Add indicator, i.e. EMA_200')
parser.add_argument('-r', '--remove', type=str, action='append', help='Remove indicator')

args = parser.parse_args()


if args.download:
    download_bi5_file_between_dates(args.download, args.start, args.end)

if args.convert:
    bi5_to_csv(args.download, args.convert)

if args.merge:
    if args.mergefile:
        merge_csv_files(args.convert, args.mergefile, args.start, args.end)
    else:
        merge_csv_files(args.convert, None, args.start, args.end)

if args.clean:
    logger.log_info("Cleaning up download folders")
    if os.path.exists(args.download):
        shutil.rmtree(args.download)
    if os.path.exists(args.convert):
        shutil.rmtree(args.convert)
    logger.log_info("Clean up complete")

if args.add:
    data = csv_to_dataframe(args.mergefile)

    for indicator in args.add:
        data = add_indicator(data, indicator)

    dataframe_to_csv(data, args.mergefile)

if args.remove:
    data = csv_to_dataframe(args.mergefile)
    for indicator in args.remove:
        data = remove_indicator(data, indicator)

    dataframe_to_csv(data, args.mergefile)
