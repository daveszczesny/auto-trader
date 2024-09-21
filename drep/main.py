import argparse
from src.servies.dukascopy import download_bi5_file_between_dates
from src.utils.convert import bi5_to_csv
from src.utils.merge import merge_csv_files

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-d', '--download', type=str, help='Download data to the specified folder')
    parser.add_argument('-s', '--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('-e', '--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('-c', '--convert', type=str, help='Convert data to the specified folder')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge data')

    args = parser.parse_args()

    if args.download:
        download_bi5_file_between_dates(args.download, args.start, args.end)
    
    if args.convert:
        bi5_to_csv(args.download, args.convert)
    
    if args.merge:
        merge_csv_files(args.convert, args.start, args.end)

if __name__ == '__main__':
    main()