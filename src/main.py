import os
import shutil
import time

from classifier import Classifier
from utils import Config

model_id = time.strftime("%Y%m%d_%H%M%S")
config_path = r"src\config.json"
# Setup config
config = Config(config_path)
classifier = Classifier(config, model_id=model_id)
classifier.mkdir()
# Backup the config
shutil.copyfile(
    config_path, os.path.join(config.path.train.description, f"config_{model_id}.json")
)
# classifier.preprocess()
classifier.load_data()
classifier.train()
