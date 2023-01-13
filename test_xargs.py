from time import sleep
from random import random

import argparse

parser = argparse.ArgumentParser(description='Please set input folder and flags for your test run')
parser.add_argument('-i','--input',type=int, default=1,help='input')

args = parser.parse_args()

if __name__ == "__main__":
    sleep(random())
    print(args.input**2)
