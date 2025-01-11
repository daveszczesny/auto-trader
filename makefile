# Mac Commands

setup-venv:
	rm -rf venv
	python3.10 -m venv venv
	. venv/bin/activate
	venv/bin/pip install -r requirements/requirements.txt
	venv/bin/pip install -r requirements/requirements_test.txt

	venv/bin/pip install -r brookyapi/requirements.txt

clean-venv:
	rm -rf venv

train:
	PYTHONPATH=${shell pwd} venv/bin/python brooksai/train_agent.py

tune:
	PYTHONPATH=${shell pwd} venv/bin/python brooksai/tuner.py

lint:
	venv/bin/pylint --fail-under=9 brooksai
	venv/bin/pylint --fail-under=9 brookyapi
	venv/bin/pylint --fail-under=9 drep

test-unit:
	PYTHONPATH=${shell pwd} venv/bin/pytest brooksai/tests/unit
	PYTHONPATH=${shell pwd} venv/bin/pytest brookyapi/tests/unit

test-integration:
	PYTHONPATH=${shell pwd} venv/bin/pytest brookyapi/tests/integration

test-e2e:
	PYTHONPATH=${shell pwd} venv/bin/pytest brookyapi/tests/e2e

test:
	PYTHONPATH=${shell pwd} venv/bin/pytest brooksai/tests/
	PYTHONPATH=${shell pwd} venv/bin/pytest brookyapi/tests/

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
	set PYTHONPATH=%cd% && venv\Scripts\pytest brookyapi\tests\unit

test-integration-w:
	set PYTHONPATH=%cd% && venv\Scripts\pytest brookypi\tests\integration

test-e2e-w:
	set PYTHONPATH=%cd% && venv\Scripts\pytest brookyapi\tests\e2e

test-w:
	set PYTHONPATH=%cd% && venv\Scripts\pytest brooksai\tests
