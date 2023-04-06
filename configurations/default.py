from dotmap import DotMap

config = DotMap({
    "grayscale": True,
    "enc1": {
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "pool_k": 2
    },
    "enc2": {
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "pool_k": 2
    },
    "enc3": {
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "pool_k": 2
    },
    "enc4": {
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "pool_k": 2
    },
    "b": {
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        }
    },
    "dec1": {
        "up": {
            "k": 2,
            "p": 0,
            "s": 2
        },
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        }
    },
    "dec2": {
        "up": {
            "k": 2,
            "p": 0,
            "s": 2
        },
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        }
    },
    "dec3": {
        "up": {
            "k": 2,
            "p": 0,
            "s": 2
        },
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        }
    },
    "dec4": {
        "up": {
            "k": 2,
            "p": 0,
            "s": 2
        },
        "conv1": {
            "k": 3,
            "p": 1,
            "s": 1
        },
        "conv2": {
            "k": 3,
            "p": 1,
            "s": 1
        }
    },
    "out": {
        "k": 1,
        "p": 0,
        "s": 1
    }
})