#!/usr/bin/env python
"""Data preprocessing command. """

import os.path as op
import sys

import requests
import numpy as np


PARAMETERS_URL = 'https://raw.githubusercontent.com/rbesenczi/my_ols/refs/heads/main/my_ols/parameters.txt'
"""Download URL for preprocessing parameters. """


def load_parameters(url=None):
    """Download preprocessing parameters from the given url or local file. """

    if url is None:
        url = PARAMETERS_URL

    if op.exists(url):
        with open(url, 'rt') as f:
            config = f.read().strip().split('\n')
    else:
        resp   = requests.get(url)
        config = resp.text.split('\n')

    param1 = int(config[0])
    param2 = int(config[1])
    param3 = int(config[2])
    return param1, param2, param3


def run_preprocessing(data):
    """Apply preprocessing rules to the given data. """
    p1, p2, p3 = load_parameters()
    return data * p1 * p2 / p3


def main(args=None):
    """Main routine - loads input file, runs preprocessing, saves the result.
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) != 2:
        print(f'Usage: {__name__} infile outfile')
        return 1

    infile  = args[0]
    outfile = args[1]

    data   = np.loadtxt(infile)
    result = run_preprocessing(data)

    np.savetxt(outfile, result)

    return 0


if __name__ == '__main__':
    sys.exit(main())
