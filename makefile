# Mac Commands

setup-venv:
	rm -rf venv
	python3 -m venv venv
	sh venv/bin/activate
	venv/bin/pip install -r requirements/requirements.txt
	venv/bin/pip install git+https://github.com/DLR-RM/stable-baselines3
	venv/bin/pip install pylint
	venv/bin/pip install pytest
	venv/bin/pip install torch torchvision tensorboard

clean-venv:
	rm -rf venv

train:
	PYTHONPATH=${shell pwd} venv/bin/python brooksai/train_agent.py

lint:
	venv/bin/pylint --fail-under=9 brooksai
	venv/bin/pylint --fail-under=9 drep

test-unit:
	PYTHONPATH=${shell pwd} venv/bin/pytest brooksai/tests/unit

setup-venv-tradevis:
	rm -rf tradevis/venv
	python3 -m venv tradevis/venv
	sh tradevis/venv/bin/activate
	tradevis/venv/bin/pip install -r requirements/requirements-tradevis.txt

run-tradevis:
	PYTHONPATH=${shell pwd} tradevis/venv/bin/python tradevis/main.py

# Windows Commands

setup-venv-w:
	rmdir /s /q venv
	python -m venv venv
	venv\Scripts\activate.bat
	venv\Scripts\pip install -r requirements/requirements.txt
	venv\Scripts\pip install git+https://github.com/DLR-RM/stable-baselines3
	venv\Scripts\pip install pylint
	venv\Scripts\pip install pytest
	venv\Scripts\pip install torch torchvision tensorboard

clean-venv-w:
	rmdir /s /q venv

train-w:
	set PYTHONPATH=%cd% && venv\Scripts\python brooksai\train_agent.py

lint-w:
	venv\Scripts\pylint --fail-under=9 brooksai
	venv\Scripts\pylint --fail-under=9 drep

test-unit-w:
	set PYTHONPATH=%cd% && venv\Scripts\pytest brooksai\tests\unit
