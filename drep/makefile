# print help
help:
	@echo "Usage: make setup-venv"
	@echo ""
	@echo "Usage: make run [download_dir=<download directory>] [convert_dir=<convert directory>] [start_date=<start date>] [end_date=<end date>] [merge=<true/false>] [merge_file=<merge file name>] [cleanup=<true/false>] [add_indicator=<add indicator>] [remove_indicator=<remove indicator>]"
	@echo ""
	@echo "Arguments:"
	@echo "  download_dir		Folder to download data"
	@echo "  start_date			Start date in YYYY-MM-DD format"
	@echo "  end_date			End date in YYYY-MM-DD format"
	@echo "  convert_dir		Folder to save converted CSV files"
	@echo "  merge				Merge data (true/false)"
	@echo "  merge_file			File name to merge"
	@echo "  cleanup			Clean up download directorys (true/false)"
	@echo "  add_indicator		Add indicator to the data"
	@echo "  remove_indicator	Remove indicator from the data"
	@echo ""
	@echo "Examples:"
	@echo "make run download_dir=download convert_dir=convert start_date=2021-01-01 end_date=2021-01-31 merge=true merge_file=merged.csv cleanup=true add_indicator=EMA_200 remove_indicator=EMA_200"

setup-venv:
	rm -rf venv
	python3 -m venv venv
	venv/bin/pip3 install -r requirements/requirements.txt


# Define default variables
download_dir ?= # download folder
convert_dir ?= # convert folder
start_date ?= # start date
end_date ?= # end date
merge ?= false # merge
merge_file ?= # merge file name
cleanup ?= false # clean up

add_indicator ?= # add indicator
remove_indicator ?= # remove indicator

run:
	venv/bin/python3 drep.py \
		${if ${download_dir}, --download ${download_dir}, } \
		${if ${convert_dir}, --convert ${convert_dir}, } \
		${if ${start_date}, --start ${start_date}, } \
		${if ${end_date}, --end ${end_date}, } \
		$(if $(filter true, $(merge)),--merge) \
		${if ${merge_file}, --mergefile ${merge_file}, } \
		$(if $(filter true, $(cleanup)),--clean) \
		${if ${add_indicator}, --add ${add_indicator}, } \
		${if ${remove_indicator}, --remove ${remove_indicator}, }

