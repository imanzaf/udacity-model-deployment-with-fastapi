'''
Define fixtures in Namespace
'''

import pytest


def pytest_configure():
    '''
    configure function to set up empty variables
    '''
    pytest.X = None
    pytest.y = None

    pytest.lgbm = None

    pytest.preds = None
