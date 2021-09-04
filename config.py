import configparser
import copy
import os

config = configparser.ConfigParser()
config.optionxform=str
config.read('./config.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]

def save_config(directory):
    with open(os.path.join(directory, 'config.conf'), 'w') as configfile:
        config.write(configfile)
