"""
The game will start by the environment retrieving the next dataset from its resources.
This will be given to the agent
The agent must make a decision and reply with [buy, sell, do nothing]
The environment will then continue to the next dataset and repeat the process

"""

import random
import sys
from trade import Trade
from agent import Agent
from gamemanager import game_manager
from random_bot import RandomBot

YEAR_5 = 2_628_000
YEAR = 525_600
MONTH = 43_800
DAY = 1_440

def main():

    robo = RandomBot()

    robo.current_data = game_manager.get_current_element()

    for i in range(200_000, 400_000):

        sys.stdout.write(f"\rIteration: {i}, Account balance: {robo.agent.account_balance}")
        sys.stdout.flush()

        if robo.agent.account_balance <= 0:
            print("\n\nAgent has run out of money")
            break

        
        robo.make_decision()
        robo.current_data = game_manager.get_current_element()

        game_manager.game_turn += 1
        game_manager.get_next_element_from_dataset()


    robo.close_all_trades()
    robo.write_logs()

    print(f"\n\nFinal account balance: {robo.agent.account_balance}")


if __name__ == "__main__":
    main()

