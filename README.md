# Welcome to Auto Trader. The home of an Autonomous AI Forex Trader

## Repository Introduction
1. **drep**: A tool that downloads and prepares our data for usage.
    - Try `cd drep` and `make help` to get started.
2. **brooksai**: The home of our little Trader, named after Al Brooks, a famous trader.
3. **ctrader**: A bot for our broker to link our awesome AI with the harsh reality of the markets.

## Follow the Progress
[JIRA Link](https://daveszczesny-college.atlassian.net/jira/software/projects/AT/boards/1/backlog)

## Getting Started

To get started with **drep**, `cd` into the package and read its README.

To get started with **brooksai**, the source code can be found under `brooksai`.

## Training
To start the training process, you will first need to install all the dependencies
```bash
# On Mac
make setup-venv

# On Windows
make setup-venv-w
```
After installing all dependencies, you will require the training data. Please enquire dszczesny1@universityofgalway.ie for it or use `drep` to retrieve it yourself.
Afterwards run the following to begin
```bash
# On Mac
make train

# On Windows
make train-w
```

## Credit

The forex data was retrieved from Dukascopy.

This project uses Stable Baselines3 for the machine learning algorithms and Gymnasium for the environments.
