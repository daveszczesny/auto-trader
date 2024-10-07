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
To train the AI there are two different options
1. Using the CPU. If you don't have a CUDA supported GPU, you can simply just run the application as is
2. Using CUDA
To check whether your GPU is supports CUDA check [CUDA-GPUS](https://developer.nvidia.com/cuda-gpus)
If it does not, use the CPU
If your GPU does support CUDA, download version 12.4 [CUDA](https://developer.nvidia.com/cuda-12-4-0-download-archive)
Using windows you will automatically download pytorch via this command that runs when you run `make setup-venv-w`
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
After this ensure torch can see your GPU
```python
import torch
print(torch.cuda.is_available())
```
If false, ensure you have set the environment variables for CUDA.


## Credit

The forex data was retrieved from Dukascopy.

This project uses Stable Baselines3 for the machine learning algorithms and Gymnasium for the environments.
