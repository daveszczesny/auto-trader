import os
import pandas as pd
from utils.constants import DATASET_NAME

class GameManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GameManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.dataset = pd.DataFrame()
        self.current_element = None
        self.game_turn = 0
        self.leverage = 500
        self.dataset = self.read_dataset()
        self.current_element = self.dataset.iloc[0]

    def read_dataset(self) -> pd.DataFrame:
        """
        Read the dataset from the resources folder
        """
        if os.path.exists(f'./resources/{DATASET_NAME}'):
            return pd.read_csv(f'./resources/{DATASET_NAME}')
        else:
            print(os.path.abspath(f'./resources/{DATASET_NAME}'))
            raise FileNotFoundError("Dataset not found")

    def get_next_element_from_dataset(self) -> pd.Series:
        """
        Get the next element from the dataset
        """
        self.current_element = self.dataset.iloc[self.game_turn]
        return self.current_element

    def get_current_element(self) -> pd.Series:
        """
        Get the current element
        """
        return self.current_element

# Usage
game_manager = GameManager()