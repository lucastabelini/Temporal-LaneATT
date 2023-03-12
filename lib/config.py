from io import StringIO

import torch
from ruamel.yaml import YAML

import lib.models as models
import lib.datasets as datasets


class yaml_tuple(tuple):
    yaml_tag = "!tuple"

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(constructor.construct_sequence(node))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_sequence(cls.yaml_tag, node, flow_style="block")


class Config:
    def __init__(self, config_path, merge_options=None):
        self.config = {}
        self.config_str = ""
        self.yaml = None
        self.load(config_path)
        if merge_options is not None:
            self._merge(merge_options)

    def _merge(self, options):
        for item in options:
            key, value = self._get_key_value_from_str(item)
            self._set_dot_attribute(key, value)

    def _get_key_value_from_str(self, data):
        data = data.split("=")
        key = data[0]
        value = "=".join(data[1:])
        return key, value

    def _set_dot_attribute(self, key, value):
        keys = key.split(".")
        return self._set_recursively(self.config, keys, value)

    def _set_recursively(self, container, keys, value):
        if len(keys) == 1:
            value = type(container[keys[0]])(value)  # cast to correct type
            container[keys[0]] = value
            return container
        else:
            return self._set_recursively(container[keys[0]], keys[1:], value)

    def load(self, path):
        self.yaml = YAML()
        self.yaml.register_class(yaml_tuple)

        with open(path, "r") as file:
            self.config = self.yaml.load(file)

    def dump(self, file):
        return self.yaml.dump(self.config, file)

    def __repr__(self):
        string_stream = StringIO()
        self.yaml.dump(self.config, string_stream)
        output_str = string_stream.getvalue()
        string_stream.close()
        return output_str

    def get_dataset(self, split):
        return getattr(datasets, self.config["datasets"][split]["type"])(
            **self.config["datasets"][split]["parameters"]
        )

    def get_model(self, **kwargs):
        name = self.config["model"]["name"]
        parameters = self.config["model"]["parameters"]
        return getattr(models, name)(**parameters, **kwargs)

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config["optimizer"]["name"])(
            model_parameters, **self.config["optimizer"]["parameters"]
        )

    def get_lr_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"]["name"])(
            optimizer, **self.config["lr_scheduler"]["parameters"]
        )

    def get_loss_parameters(self):
        return self.config["loss_parameters"]

    def get_train_parameters(self):
        return self.config["train_parameters"]

    def get_test_parameters(self):
        return self.config["test_parameters"]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
