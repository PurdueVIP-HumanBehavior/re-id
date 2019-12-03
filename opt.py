defaultkey = "default"

import argparse
from detectors import options as detopt
from distancemetrics import options as distopt
from loaders import options as loadopt
from galleries import options as galopt
from vectorgenerator import options as vecopt

parser = argparse.ArgumentParser(description="multi-camera re-id system")
parser.add_argument("-d", "--detector", default=defaultkey, choices=detopt.keys())
parser.add_argument("-r", "--distance", default=defaultkey, choices=distopt.keys())
parser.add_argument("-l", "--loader", default=defaultkey, choices=loadopt.keys())
parser.add_argument("-g", "--gallery", default=defaultkey, choices=galopt.keys())
parser.add_argument("-v", "--vectgen", default=defaultkey, choices=vecopt.keys())
parser.add_argument("-i", "--interval", default=2, type=int)

args = parser.parse_args()