import pytest
from src.model import borislav_preprocessing as bp


class TestGetYearsToPredict(object):

    def test_get_years_to_predict_should_return_correct_value(self, mocker):
        actual = bp.demo()
        expected = True
        assert actual == expected
