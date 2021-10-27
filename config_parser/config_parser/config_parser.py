import configparser
import json

from os.path import isfile
from collections import OrderedDict
from typing import Union, Any


class ConfigParser:
    def __init__(self, config_path: str) -> None:
        """
        Constructor

        Parameters:
        -----------
            config_path: str = path of the config file
        """
        self.config_path = config_path
        assert isfile(self.config_path), "Insert valid config path."
        self.load_config()

    @staticmethod
    def safe_cast(val: Any, to_type: type) -> Any:
        try:
            return to_type(val)
        except (ValueError, TypeError):
            return val

    @staticmethod
    def inference_type(dictionary: dict, key: str, value: str) -> None:
        """
        Inference type from read value using json formatting

        Parameters:
        -----------
            dictionary: dict = dictonary in which will be saved the casted value
            key: str = key value of the dictonary
            value: str = value read from config file
        """
        dictionary[key] = json.loads(value)

    def load_config(self) -> None:
        """
        Load config from files
        """
        config = configparser.ConfigParser()
        config.read(self.config_path)
        for section in config.sections():
            if section != "__docker__" and section != "__udocker__":
                setattr(self, section, {})
                for k, v in config[section].items():
                    if isinstance(v, OrderedDict) or isinstance(v, dict):
                        for k2, v2 in v.items():
                            ConfigParser.inference_type(self[k], k2, v2)
                    else:
                        ConfigParser.inference_type(self[section], k, v)

    def flatten(self, d) -> dict:
        """
        Recursive function that compacts a dictonary

        Paramaters:
        -----------
            d = dictonary to compact
        """
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                val = [val]
            if isinstance(val, list):
                for subdict in val:
                    deeper = self.flatten(subdict).items()
                    out.update({key + "_" + key2: val2 for key2, val2 in deeper})
            else:
                out[key] = val
        return out

    def compact_config(self) -> dict:
        """
        Compacts nested config dict to single dict
        """
        out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                val = [val]
            if isinstance(val, list):
                for subdict in val:
                    deeper = self.flatten(subdict).items()
                    out.update({key + "_" + key2: val2 for key2, val2 in deeper})
            else:
                out[key] = val
        return out

    def get_config(self) -> dict:
        """
        Return all configs.
        """
        out = {}
        for key, val in self.__dict__.items():
            if key != "config_path":
                out[key] = val

        return out

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item: Any, default: Any) -> Any:
        """
        Get dict functin wrapper
        """
        return getattr(self, item, default)
