# print help
help:
	@echo "Usage: make setup-venv"
	@echo ""
	@echo "Usage: make run [df=<download_folder>] [cf=<convert_folder>] [s=<YYYY-MM-DD>] [e=<YYYY-MM-DD>] [m=true]"
	@echo ""
	@echo "Arguments:"
	@echo "  df   Folder to download data"
	@echo "  s    Start date in YYYY-MM-DD format"
	@echo "  e    End date in YYYY-MM-DD format"
	@echo "  cf   Folder to save converted CSV files"
	@echo "  m    Merge data (true/false)"
	@echo ""
	@echo "Examples:"
	@echo "  make run df=data cf=csv_data s=2022-01-01 e=2022-01-31 m=true"

setup-venv:
	rm -rf venv
	python3 -m venv venv
	venv/bin/pip3 install -r requirements/requirements.txt


# Define default variables
df ?= # download folder
cf ?= # convert folder
s ?= # start date
e ?= # end date
m ?= false # merge

run:
	venv/bin/python3 main.py \
		${if ${df}, --download ${df}, } \
		${if ${cf}, --convert ${cf}, } \
		${if ${s}, --start ${s}, } \
		${if ${e}, --end ${e}, } \
		${if ${m}, --merge, }



