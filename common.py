import yaml
import os
import pickle

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yml')

with open(CONFIG_PATH, "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    DB_PATH = str(config_yaml['paths']['db_path'])
    MODEL_PATH = str(config_yaml['paths']["model_path"])
    RANDOM_STATE = int(config_yaml["ml"]["random_state"])

#SQLite requires the absolute path
DB_PATH = os.path.abspath(DB_PATH)

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model