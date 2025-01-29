# Welcome to Auto Trader. The home of an Autonomous AI Forex Trader

## Repository Introduction
1. **drep**: A tool that downloads and performs preprocessing for our data
    - Try `cd drep` and `make help` to get started.
2. **brooksai**: The home of our little Trader, named after Al Brooks, a famous trader.
    - Utilizes Gymnasium and Stable Baselines 3
3. **ctrader**: A bot for our broker to link our awesome AI with the harsh reality of the markets.
    - Read README.md for more information
4. **brookyapi**: The medium between brooksai and ctrader.
    - Read README.md for more information

## Follow the Progress
The majority of the progress is tracked via a Jira board. Any issues do post them on Github Issues section
<br/>[JIRA Link](https://daveszczesny-college.atlassian.net/jira/software/projects/AT/boards/1/backlog)


## Project Contributions

Contributions are not permitted until summer 2025 due to College restrictions.

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

To get started with **brookyapi**, the source code is under `brookyapi`, however, it links in with `brooksai` and the `root` directory.
```bash
# Deploy Cloud Run locally, [NOTE] you will require a .env file to do this. Reach out to code owner for this
sh local_deploy.sh
```

More on **brookyapi** under its own README.md

## Future Ambitions

- More Forex symbol support
- More time control support (i.e. 5 minute, 15 minute)
- Ability to trade multiple symbols concurrently
- Use more indicators / ability to learn new indicators at any time

## Credit

“Forex Historical Data Feed :: Dukascopy Bank SA.” Dukascopy Bank SA, 2024, www.dukascopy.com/swiss/english/fx-market-tools/historical-data/.

@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}

@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}