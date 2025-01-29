
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('Yaml Parser')

class ConfigManager:
    _instance = None

    def __new__(cls, config_file='brooksai/config.yaml'):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = None
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, key, default=None, verbose: int = 0):
        keys = key.split('.')
        value = self.config

        if verbose > 0:
            logger.info(f'Full config at start {self.config}')

        for k in keys:
            if verbose > 0:
                logger.info(f"Accessing key: {k}, Current value: {value}")
            if isinstance(value, dict):
                if k in value:
                    value = value[k]
                else:
                    if verbose > 0:
                        logger.info(f"Key '{k}' not found. Returning default: {default}")
                    return default
            else:
                if verbose > 0:
                    logger.info(f"Value is no longer a dictionary. Key '{k}' invalid. Returning default: {default}")
                return default
        if verbose > 0:
            print(f"Final retrieved value: {value}")
        return value
