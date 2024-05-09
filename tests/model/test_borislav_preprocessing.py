import pytest
from src.model import borislav_preprocessing as bp
import pandas as pd

data = {'Neighbourhood': ['Bavel', 'City', 'Random', 'Text', 'For', 'This', 'Column', 'Instead', 'Neighbourhood'],
        'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        'temp': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        'Grade': [6, 5, 5, 9, 10, 5, 8, 3, 5]}

test_df = pd.DataFrame(data)
invalid_column_name = 'invalid'


class TestGetYearsToPredict(object):

    def test_get_years_to_predict_should_return_correct_value(self, mocker):
        future_year = (max(test_df['year'].values) + 1)
        mocker.patch.object(bp, 'MAX_YEAR_INCLUSIVE', future_year)
        actual = bp.get_years_to_predict(test_df, 'year')
        expected = future_year
        assert actual == expected, "Get Years To Predict doesn't produce the expected result."

    def test_get_years_to_predict_should_raise_exception_for_empty_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            bp.get_years_to_predict(obj, str)

    def test_get_years_to_predict_should_raise_exception_for_empty_as_input_2(self):
        obj = None
        with pytest.raises(ValueError):
            bp.get_years_to_predict(pd.DataFrame(), obj)


class TestSplitColumns(object):

    def test_split_columns_should_return_correct_value(self, mocker):
        actual = bp.split_columns(test_df.columns, ['Grade'])
        expected = ['Grade_scaled'], ['Neighbourhood_scaled', 'year_scaled', 'temp_scaled']
        assert actual == expected, "Get Years To Predict doesn't produce the expected result."

    def test_split_columns_should_raise_exception_for_empty_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            bp.split_columns(obj, [0])

    def test_split_columns_should_raise_exception_for_empty_as_input_2(self):
        obj = None
        with pytest.raises(ValueError):
            bp.split_columns([0], obj)

    def test_split_columns_should_raise_exception_for_empty_as_input_1_1(self):
        obj = []
        with pytest.raises(ValueError):
            bp.split_columns(obj, [0])

    def test_split_columns_should_raise_exception_for_empty_as_input_2_1(self):
        obj = []
        with pytest.raises(ValueError):
            bp.split_columns([0], obj)


class TestScaleValues(object):

    def test_scale_values_should_return_correct_value_for_max(self, mocker):
        actual = max(bp.scale_values(test_df, 'Grade', bp.MIN_SCALE, bp.MAX_SCALE)['Grade_scaled'])
        expected = bp.MAX_SCALE - 1
        assert actual == expected, "Get Scale Values doesn't produce the expected result."

    def test_scale_values_should_return_correct_value_for_min(self, mocker):
        actual = min(bp.scale_values(test_df, 'Grade', bp.MIN_SCALE, bp.MAX_SCALE)['Grade_scaled'])
        expected = bp.MIN_SCALE - 1
        assert actual == expected, "Get Scale Values doesn't produce the expected result."

    def test_scale_values_should_raise_exception_for_empty_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            bp.scale_values(obj, '', 2, 1)

    def test_scale_values_should_raise_exception_for_empty_as_input_2(self):
        obj = None
        with pytest.raises(ValueError):
            bp.scale_values(pd.DataFrame(), obj, 2, 1)

    def test_scale_values_should_raise_exception_for_empty_as_input_3(self):
        obj = None
        with pytest.raises(ValueError):
            bp.scale_values(pd.DataFrame(), '', obj, 1)

    def test_scale_values_should_raise_exception_for_empty_as_input_4(self):
        obj = None
        with pytest.raises(ValueError):
            bp.scale_values(pd.DataFrame(), '', 2, obj)

    def test_scale_values_should_raise_exception_for_invalid_mix_max_values_as_input(self):
        with pytest.raises(ValueError):
            bp.scale_values(pd.DataFrame(), '', 2, 1)


class TestCalculateIndexBasedOnGoodBadCols(object):

    def test_calculate_index_based_on_good_bad_cols_should_return_correct_value_for_max(self, mocker):
        actual = len(bp.calculate_index_based_on_good_bad_cols(test_df, ['Grade'], ['temp'], False).columns)
        expected = len(test_df.columns) + 1 + 3  # 3 is the debug columns
        assert actual == expected, "Get Scale Values doesn't produce the expected result."

    def test_calculate_index_based_on_good_bad_cols_should_raise_exception_for_empty_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            bp.calculate_index_based_on_good_bad_cols(obj, [0], [0], is_scaled=False)

    def test_calculate_index_based_on_good_bad_cols_should_raise_exception_for_empty_as_input_2(self):
        obj = None
        with pytest.raises(ValueError):
            bp.calculate_index_based_on_good_bad_cols(pd.DataFrame(), obj, [0], is_scaled=False)

    def test_calculate_index_based_on_good_bad_cols_should_raise_exception_for_empty_as_input_3(self):
        obj = None
        with pytest.raises(ValueError):
            bp.calculate_index_based_on_good_bad_cols(pd.DataFrame(), [0], obj, is_scaled=False)

    def test_calculate_index_based_on_good_bad_cols_should_raise_exception_for_empty_as_input_3_1(self):
        obj = None
        with pytest.raises(ValueError):
            bp.calculate_index_based_on_good_bad_cols(pd.DataFrame(), obj, [0], is_scaled=False)

    def test_calculate_index_based_on_good_bad_cols_should_raise_exception_for_empty_as_input_3_2(self):
        obj = []
        with pytest.raises(ValueError):
            bp.calculate_index_based_on_good_bad_cols(pd.DataFrame(), [0], obj, is_scaled=False)
