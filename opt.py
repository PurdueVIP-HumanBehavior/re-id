defaultkey = "default"

import argparse
from detectors import options as detopt
from distancemetrics import options as distopt
from loaders import options as loadopt

parser = argparse.ArgumentParser(description="multi-camera re-id system")
parser.add_argument("-d", "--detector", default=defaultkey, choices=detopt.keys())
parser.add_argument("-r", "--distance", default=defaultkey, choices=distopt.keys())
parser.add_argument("-l", "--loader", default=defaultkey, choices=loadopt.keys())