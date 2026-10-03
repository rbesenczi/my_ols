#!/usr/bin/env python


import numpy as np

from unittest import mock

import my_ols.preprocess as preproc

from my_ols.tests import tempdir


def test_load_parameters():

    with tempdir():
        with open('parameters.txt', 'wt') as f:
            f.write('1\n2\n3')
        with mock.patch('my_ols.preprocess.PARAMETERS_URL', 'parameters.txt'):
            assert preproc.load_parameters() == (1, 2, 3)


def test_run_preprocessing():

    # Each test is a tuple of (input data, input parameters, expected result)
    tests = [
        (10,                     (10, 10, 10), 100),
        (10,                     (1,  2,  4),  5),
        (20,                     (1,  2,  5),  8),
        (np.array([10, 20, 30]), (10, 10, 10), np.array([100, 200, 300])),
    ]

    for data, params, expect in tests:
        with mock.patch('my_ols.preprocess.load_parameters', return_value=params):
            result = preproc.run_preprocessing(data)
            assert np.all(np.isclose(result, expect))


def test_main():

    data   = np.array([10,  20,  30])
    expect = np.array([100, 200, 300])
    params = (10, 10, 10)

    with tempdir():
        np.savetxt('input.txt', data)

        with mock.patch('my_ols.preprocess.load_parameters',
                        return_value=params):
            preproc.main(('input.txt', 'output.txt'))

        result = np.loadtxt('output.txt')
        assert np.all(np.isclose(result, expect))
