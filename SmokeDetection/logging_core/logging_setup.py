import os
import json
import logging
import logging.config

def setup_logging(default_level=logging.INFO):
    """Set up logging configuration."""
    project_dir = os.path.dirname(os.getcwd())
    logging_config_path = os.path.join(project_dir, 'logging_core', 'logging_config.json')
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)