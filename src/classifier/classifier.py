import os
import time

import torch

from .data_loader import DataLoader
from .preprocess import Preprocess
from .train import Train


class Classifier:
    def __init__(
        self,
        config,
        model_id,
    ):
        """
        Initialize the classifier
        """
        self.config = config
        self.model_id = model_id

    def _get_folder_list(self):
        """
        Get the folder list in the path.
        """
        folder_list = []
        for key in dir(self.config.path):
            if key != "input" and not key.startswith("__"):
                value = getattr(self.config.path, key)
                if hasattr(value, "__dict__"):
                    for sub_key in dir(value):
                        if not sub_key.startswith("__"):
                            folder_list.append(getattr(value, sub_key))
                else:
                    folder_list.append(value)
        return folder_list

    def mkdir(self):
        """
        Make directories for the output.
        """
        folder_list = self._get_folder_list()
        [os.makedirs(folder, exist_ok=True) for folder in folder_list]

    def preprocess(self):
        """
        Preprocess the data.
        """
        preprocess = Preprocess(self.config)
        preprocess.run()

    def load_data(self):
        """
        Load the data.
        """
        data_loader = DataLoader(self.config, self.model_id)
        self.label_structure, self.data = data_loader.run()

    def train(self):
        """
        Train the model.
        """
        train = Train(self.config, self.label_structure, self.data, self.model_id)
        train.run()

    def predict(self):
        """
        Predict the data.
        """
        pass
