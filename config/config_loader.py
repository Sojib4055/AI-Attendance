import yaml
import os

def load_config(path: str = None):
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
