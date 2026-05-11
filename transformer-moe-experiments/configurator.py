"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import os
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if arg.startswith('--'):
        # assume it's a --key=value argument
        if '=' not in arg:
            raise ValueError(f"Expected --key=value format, got: {arg!r}")
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok (None defaults accept any type)
            # allow safe numeric coercion: int→float always, float→int only if whole
            current = globals()[key]
            if current is not None and type(attempt) != type(current):
                if isinstance(current, float) and isinstance(attempt, int):
                    attempt = float(attempt)
                elif isinstance(current, int) and isinstance(attempt, float):
                    if attempt == int(attempt):
                        attempt = int(attempt)
                    else:
                        raise TypeError(
                            f"Type mismatch for --{key}: expected int, "
                            f"got non-integer float (value: {val!r})"
                        )
                else:
                    raise TypeError(
                        f"Type mismatch for --{key}: expected {type(current).__name__}, "
                        f"got {type(attempt).__name__} (value: {val!r})"
                    )
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
    else:
        # assume it's the name of a config file
        config_file = arg
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file not found: {config_file!r}\n"
                f"(Did you forget to expand template variables like {{width}}?)"
            )
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
