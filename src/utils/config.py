import json


class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_config()
        self._set_attributes(self, self.data)

    def _load_config(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _set_attributes(self, obj, data):
        for key, value in data.items():
            if isinstance(value, dict):
                sub_obj = type("ConfigNode", (object,), {})()
                self._set_attributes(sub_obj, value)
                setattr(obj, key, sub_obj)
            else:
                setattr(obj, key, value)

    def __getattr__(self, name):
        return self.data.get(name)
