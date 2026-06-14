"""Smoke tests for the code responsible for pd.Series type definition
"""

import pytest
import pandas as pd
import numpy as np
from storybear.datagal.plotters.base import infer_kind


def generate_data():
    PROT_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYXBZJ'
    DNA_ALPHABET = 'ACGTU'
    return [
        ('numeric', pd.Series([1, 2, 3, 4, 5, 6, 7, np.nan])),
        ('numeric', pd.Series([np.nan, 1, 2, 3, 4, 5, 6, 7, np.nan])),
        ('protein', pd.Series(
            ["".join(np.random.choice(list(PROT_ALPHABET), 5)) for i in range(5)] + [np.nan]
        )),
        ('dna', pd.Series(
            ["".join(np.random.choice(list(DNA_ALPHABET), 5)) for i in range(5)] + [np.nan]
        )),
        ('text', pd.Series(
            ["".join(np.random.choice(list("ABGCDEfgh123,!"), 5)) for i in range(5)] + [np.nan]
        )),
    ]
    
def test_check_infer_kind():
    data = generate_data()
    for k, v in data:
        assert infer_kind(v) == k