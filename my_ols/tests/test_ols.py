#!/usr/bin/env python


import numpy as np

import my_ols.main as main

from my_ols.tests import tempdir


test_model = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).T
test_data  = (test_model[:, 0] * 5) + (test_model[:, 1] * 20)


test_model_deficient = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                                 [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                                 [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).T


def test_ols():
    fit, err = main.ols(test_data, test_model)
    assert np.isclose(fit, [5, 20]).all()
    assert np.isclose(err, 0)      .all()

def test_ols_singular():
    fit, err = main.ols(test_data, test_model_deficient)
    assert np.isclose(fit, [2.5, 2.5, 20]).all()
    assert np.isclose(err, 0)             .all()


def test_ols2():
    fit, err = main.ols(2 * test_data, test_model)
    assert np.isclose(fit, [10, 40]).all()
    assert np.isclose(err, 0)       .all()


def test_main():
    with tempdir():
        np.savetxt('data.txt', test_data)
        np.savetxt('model.txt', test_model)

        args = ('data.txt', 'model.txt', 'pes.txt', 'residuals.txt')
        main.main(args)

        fit = np.loadtxt('pes.txt')
        err = np.loadtxt('residuals.txt')

        assert np.isclose(fit, [5, 20]).all()
        assert np.isclose(err, 0)      .all()
