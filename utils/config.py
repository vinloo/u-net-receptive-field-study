import json
from dotmap import DotMap
from sys import exit

def load_config(config_name):
    try:
        config = json.load(open(f"configurations/{config_name}.json"))
        config = DotMap(config, _dynamic=False)
    except json.decoder.JSONDecodeError:
        print("Invalid config file")
        exit(1)

    try:
        assert type(config.name) == str, "name must be a string"
        assert type(config.grayscale) == bool, "grayscale must be a boolean"
        assert type(config.depth) == int, "depth must be an integer"
        assert type(config.channels) == list, "channels must be a list"
        assert type(config.conv_k) == int, "conv_k must be an integer"
        assert type(config.conv_p) in [str, int], "conv_p must be a string or an integer"
        assert type(config.conv_s) == int, "conv_s must be an integer"
        assert type(config.pool_k) == int, "pool_k must be an integer"
        assert type(config.deconv_k) == int, "deconv_k must be an integer"
        assert type(config.deconv_p) == int, "deconv_p must be an integer"
        assert type(config.deconv_s) == int, "deconv_s must be an integer"
        
        assert config.depth > 0, "depth must be greater than 0"
        assert config.conv_k > 0, "conv_k must be greater than 0"
        assert config.conv_s > 0, "conv_s must be greater than 0"
        assert config.pool_k > 0, "pool_k must be greater than 0"
        assert config.deconv_k > 0, "deconv_k must be greater than 0"
        assert config.deconv_s > 0, "deconv_s must be greater than 0"
        
        if type(config.conv_p) == str:
            assert config.conv_p == "same", "conv_p must be 'same' or an integer"
        else:
            assert config.conv_p >= 0, "conv_p must be 'same' or at least 0"

        assert len(config.channels) == config.depth + 1, "channels must have length depth + 1"

    except AttributeError as e:
        print(f"'{config_name}' is missing attribute '{e.name}'")
        exit(1)
    except AssertionError as e:
        print(e)
        exit(1)

    return config
