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
        assert type(config.grayscale) is bool, "config.grayscale is of the wrong type, expected bool"

        assert type(config.enc1.conv1.k) is int, "config.enc1.conv1.k is of the wrong type, expected int"
        assert type(config.enc1.conv1.s) is int, "config.enc1.conv1.s is of the wrong type, expected int"
        assert type(config.enc1.conv1.p) is int or config.enc1.conv1.p == "same", "config.enc1.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc1.conv2.k) is int, "config.enc1.conv2.k is of the wrong type, expected int" 
        assert type(config.enc1.conv2.s) is int, "config.enc1.conv2.s is of the wrong type, expected int"
        assert type(config.enc1.conv2.p) is int or config.enc1.conv2.p == "same", "config.enc1.conv2.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc1.pool_k) is int, "config.enc1.pool_k is of the wrong type, expected int"

        assert type(config.enc2.conv1.k) is int, "config.enc2.conv1.k is of the wrong type, expected int"
        assert type(config.enc2.conv1.s) is int, "config.enc2.conv1.s is of the wrong type, expected int"
        assert type(config.enc2.conv1.p) is int or config.enc2.conv1.p == "same", "config.enc2.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc2.conv2.k) is int, "config.enc2.conv2.k is of the wrong type, expected int"
        assert type(config.enc2.conv2.s) is int, "config.enc2.conv2.s is of the wrong type, expected int"
        assert type(config.enc2.conv2.p) is int or config.enc2.conv2.p == "same", "config.enc2.conv2.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc2.pool_k) is int, "config.enc2.pool_k is of the wrong type, expected int"

        assert type(config.enc3.conv1.k) is int, "config.enc3.conv1.k is of the wrong type, expected int"
        assert type(config.enc3.conv1.s) is int, "config.enc3.conv1.s is of the wrong type, expected int"
        assert type(config.enc3.conv1.p) is int or config.enc3.conv1.p == "same", "config.enc3.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc3.conv2.k) is int, "config.enc3.conv2.k is of the wrong type, expected int"
        assert type(config.enc3.conv2.s) is int, "config.enc3.conv2.s is of the wrong type, expected int"
        assert type(config.enc3.conv2.p) is int or config.enc3.conv2.p == "same", "config.enc3.conv2.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc3.pool_k) is int, "config.enc3.pool_k is of the wrong type, expected int"

        assert type(config.enc4.conv1.k) is int, "config.enc4.conv1.k is of the wrong type, expected int"
        assert type(config.enc4.conv1.s) is int, "config.enc4.conv1.s is of the wrong type, expected int"
        assert type(config.enc4.conv1.p) is int or config.enc4.conv1.p == "same", "config.enc4.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc4.conv2.k) is int, "config.enc4.conv2.k is of the wrong type, expected int"
        assert type(config.enc4.conv2.s) is int, "config.enc4.conv2.s is of the wrong type, expected int"
        assert type(config.enc4.conv2.p) is int or config.enc4.conv2.p == "same", "config.enc4.conv2.p is of the wrong type, expected int or value \"same\""
        assert type(config.enc4.pool_k) is int, "config.enc4.pool_k is of the wrong type, expected int"

        assert type(config.b.conv1.k) is int, "config.b.conv1.k is of the wrong type, expected int"
        assert type(config.b.conv1.s) is int, "config.b.conv1.s is of the wrong type, expected int"
        assert type(config.b.conv1.p) is int or config.b.conv1.p == "same", "config.b.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.b.conv2.k) is int, "config.b.conv2.k is of the wrong type, expected int"
        assert type(config.b.conv2.s) is int, "config.b.conv2.s is of the wrong type, expected int"
        assert type(config.b.conv2.p) is int or config.b.conv2.p == "same", "config.b.conv2.p is of the wrong type, expected int or value \"same\""

        assert type(config.dec1.up.k) is int, "config.dec1.up.k is of the wrong type, expected int"
        assert type(config.dec1.up.s) is int, "config.dec1.up.s is of the wrong type, expected int"
        # assert type(config.dec1.up.p) is int, "config.dec1.up.p is of the wrong type, expected int"
        assert type(config.dec1.conv1.k) is int, "config.dec1.conv1.k is of the wrong type, expected int"
        assert type(config.dec1.conv1.s) is int, "config.dec1.conv1.s is of the wrong type, expected int"
        assert type(config.dec1.conv1.p) is int or config.dec1.conv1.p == "same", "config.dec1.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.dec1.conv2.k) is int, "config.dec1.conv2.k is of the wrong type, expected int"
        assert type(config.dec1.conv2.s) is int, "config.dec1.conv2.s is of the wrong type, expected int"
        assert type(config.dec1.conv2.p) is int or config.dec1.conv2.p == "same", "config.dec1.conv2.p is of the wrong type, expected int or value \"same\""

        assert type(config.dec2.up.k) is int, "config.dec2.up.k is of the wrong type, expected int"
        assert type(config.dec2.up.s) is int, "config.dec2.up.s is of the wrong type, expected int"
        # assert type(config.dec2.up.p) is int, "config.dec2.up.p is of the wrong type, expected int"
        assert type(config.dec2.conv1.k) is int, "config.dec2.conv1.k is of the wrong type, expected int"
        assert type(config.dec2.conv1.s) is int, "config.dec2.conv1.s is of the wrong type, expected int"
        assert type(config.dec2.conv1.p) is int or config.dec2.conv1.p == "same", "config.dec2.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.dec2.conv2.k) is int, "config.dec2.conv2.k is of the wrong type, expected int"
        assert type(config.dec2.conv2.s) is int, "config.dec2.conv2.s is of the wrong type, expected int"
        assert type(config.dec2.conv2.p) is int or config.dec2.conv2.p == "same", "config.dec2.conv2.p is of the wrong type, expected int or value \"same\""

        assert type(config.dec3.up.k) is int, "config.dec3.up.k is of the wrong type, expected int"
        assert type(config.dec3.up.s) is int, "config.dec3.up.s is of the wrong type, expected int"
        # assert type(config.dec3.up.p) is int, "config.dec3.up.p is of the wrong type, expected int"
        assert type(config.dec3.conv1.k) is int, "config.dec3.conv1.k is of the wrong type, expected int"
        assert type(config.dec3.conv1.s) is int, "config.dec3.conv1.s is of the wrong type, expected int"
        assert type(config.dec3.conv1.p) is int or config.dec3.conv1.p == "same", "config.dec3.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.dec3.conv2.k) is int, "config.dec3.conv2.k is of the wrong type, expected int"
        assert type(config.dec3.conv2.s) is int, "config.dec3.conv2.s is of the wrong type, expected int"
        assert type(config.dec3.conv2.p) is int or config.dec3.conv2.p == "same", "config.dec3.conv2.p is of the wrong type, expected int or value \"same\""

        assert type(config.dec4.up.k) is int, "config.dec4.up.k is of the wrong type, expected int"
        assert type(config.dec4.up.s) is int, "config.dec4.up.s is of the wrong type, expected int"
        # assert type(config.dec4.up.p) is int, "config.dec4.up.p is of the wrong type, expected int"
        assert type(config.dec4.conv1.k) is int, "config.dec4.conv1.k is of the wrong type, expected int"
        assert type(config.dec4.conv1.s) is int, "config.dec4.conv1.s is of the wrong type, expected int"
        assert type(config.dec4.conv1.p) is int or config.dec4.conv1.p == "same", "config.dec4.conv1.p is of the wrong type, expected int or value \"same\""
        assert type(config.dec4.conv2.k) is int, "config.dec4.conv2.k is of the wrong type, expected int"
        assert type(config.dec4.conv2.s) is int, "config.dec4.conv2.s is of the wrong type, expected int"
        assert type(config.dec4.conv2.p) is int or config.dec4.conv2.p == "same", "config.dec4.conv2.p is of the wrong type, expected int or value \"same\""

        assert type(config.out.k) is int, "config.out.k is of the wrong type, expected int"
        assert type(config.out.s) is int, "config.out.s is of the wrong type, expected int"
        assert type(config.out.p) is int or config.out.p == "same", "config.out.p is of the wrong type, expected int or value \"same\""

    except AttributeError as e:
        print(f"'{config_name}' is missing attribute '{e.name}'")
        exit(1)
    except AssertionError as e:
        print(e)
        exit(1)

    return config
