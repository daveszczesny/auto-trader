import logging
import sys


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

class Logger:

    def get_logger(self):
        return _logger

    def log_info(self, msg: str):
        print('\n')
        _logger.info(msg)

    def log_warning(self, msg: str):
        _logger.warning(msg)

    def log_error(self, msg: str):
        _logger.error(msg)

    def log_exception(self, msg: str, exc_info: Exception):
        _logger.exception(msg, exc_info=exc_info)

    def log_state(self, msg: str):
        """
        Log the current state of the program inline
        """
        sys.stdout.write(f"\r{msg}")
        sys.stdout.flush()
        
logger = Logger()