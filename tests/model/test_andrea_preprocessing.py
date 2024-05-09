import pytest
from src.model import andrea_preprocessing as ap
import pandas as pd


def test_preprocess_livability():
    df = pd.read_csv("./data/raw/livability_dataset.csv")
    actual = ap.preprocess_livability(df)['Year'].values
    expected = '2017'
    assert expected in actual

def test_preprocess_crimes():
    df = pd.read_csv("./data/raw/recorded_crimes.csv")
    actual = ap.preprocess_crimes(df)['Perioden'].values
    expected = '2013'
    assert expected not in actual

