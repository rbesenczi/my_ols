#!/usr/bin/env python
"""OLS regression command. """


import numpy as np
import sys


def ols(data, model):
    """Perform ordinary-least-squares regression. """

    fit   = np.linalg.pinv(model) @ data
    error = data - (model @ fit)
    return fit, error


def main(args=None):
    """Main routine - loads input data and model, fits the model to the data,
    saves the result.
    """

    if args is None:
        args = sys.argv[1:]

    if len(args) != 4:
        print(f'Usage: {__name__} data_in design_in fit_out error_out')
        return 1

    datafile   = args[0]
    designfile = args[1]
    fitfile    = args[2]
    errfile    = args[3]

    # load data and model
    data  = np.loadtxt(datafile)
    model = np.loadtxt(designfile)

    # perform OLS regression
    fit, error = ols(data, model)

    # save parameter estimates and residuals
    np.savetxt(fitfile, fit)
    np.savetxt(errfile, error)

    return 0


if __name__ == '__main__':
    sys.exit(main())
