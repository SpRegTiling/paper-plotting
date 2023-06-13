import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def rand_jitter(arr):
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


def create_chart_grid(charts, row_width):
    charts_merged = None
    charts_row = None

    col = 0
    for i in range(0, len(charts)):
        col = i % row_width
        if col == 0:
            charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
            charts_row = None

        charts_row = charts[i] if charts_row is None else charts_row | charts[i]

    if col:
        charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
    return charts_merged

