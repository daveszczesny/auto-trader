
setup-venv:
	rm -rf venv
	python3 -m venv venv
	sh venv/bin/activate
	venv/bin/pip install -r requirements/requirements.txt
	venv/bin/pip install git+https://github.com/DLR-RM/stable-baselines3

train:
	venv/bin/python brooksai/train_agent.py

lint:
	venv/bin/pylint brooksai
	venv/bin/pylint drep