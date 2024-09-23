
from gamemanager import game_manager
from utils.constants import TimeConversion

def main():

    for _ in range(TimeConversion.YEAR_DAYS):
        
        # bot decision
        # bot.make_decision()

        game_manager.game_turn += 1
        game_manager.get_next_element_from_dataset()

if __name__ == "__main__":
    main()

