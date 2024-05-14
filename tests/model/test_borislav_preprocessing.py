import pytest
from testasddsa.model import borislav_preprocessing as bp


class TestGetYearsToPredict(object):

    def test_get_years_to_predict_should_return_correct_value(self):
        actual = bp.demo()
        expected = True
        assert actual == expected
