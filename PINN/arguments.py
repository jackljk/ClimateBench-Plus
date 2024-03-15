import argparse
import yaml

def arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-c', '--config', help='Path to config file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()

    # Access the values of the arguments
    config_file = args.config
    verbose_mode = args.verbose

    # Read the YAML config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config, verbose_mode

