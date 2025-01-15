import yaml

class ConfigManager:
    _instance = None

    def __init__(self):
        self.config = None

    def __new__(cls, config_file='brooksai/config.yaml'):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
