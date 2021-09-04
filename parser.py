import argparse
from distutils.util import strtobool
import logging

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)


def parse_args_and_update_config(configuration):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # parser.add_argument('--some-known-arg', type=int, default=0,
    #                     help='Description of arg')
    # ...

    # split args into known and unknown args
    args, unknown = parser.parse_known_args()
    other_args = {
        (remove_prefix(key, '--'), val)
        for (key, val) in zip(unknown[::2], unknown[1::2])
    }
    args.other_args = other_args

    # handle known args (included above with parser.add_argument)
    for arg_name, arg_value in vars(args).items():
        if arg_name in ['other_args', 'ignored-hyperparams-go-here']:
            continue
        configuration[arg_name] = arg_value

    # handle other args (included as runtime args that match the config variables)
    for arg_name, arg_value in args.other_args:
        update_param(configuration, arg_name, arg_value)
    return configuration
