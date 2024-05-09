import pytest
from src.utils import preprocess_helper as pph
import pandas as pd

data = {'Name': ['Tom', 'Nick', 'Krish', 'Nick', 'Krishna', 'Krish', 'Mario', 'Mario', 'Mario'],
        'Year': [2019, 2019, 2019, 2020, 2021, 2021, 2019, 2020, 2021],
        'Grade': [6, 5, 5, 9, 10, 5, 8, 3, 5]}

test_df = pd.DataFrame(data)
invalid_column_name = 'invalid'


class TestCreatePivot(object):

    def test_create_pivot_columns_should_return_correct_column_count3(self):
        expected = len(set(data['Year']))
        actual = len(pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade').columns)
        assert actual == expected, "Create Pivot returns wrong column count result."

    def test_create_pivot_rows_should_return_correct_row_count3(self):
        expected = len(set(data['Name']))
        actual = len(pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade'))
        assert actual == expected, "Create Pivot returns wrong row count result."

    def test_create_pivot_invalid_index_column_should_raise_exception(self):
        with pytest.raises(ValueError):
            pph.create_pivot_from_df(test_df, invalid_column_name, 'Year', 'Grade')

    def test_create_pivot_invalid_columns_column_should_raise_exception(self):
        with pytest.raises(ValueError):
            pph.create_pivot_from_df(test_df, 'Name', invalid_column_name, 'Grade')

    def test_create_pivot_invalid_value_column_should_raise_exception(self):
        with pytest.raises(ValueError):
            pph.create_pivot_from_df(test_df, 'Name', 'Year', invalid_column_name)


class TestComputeMovingAverage(object):

    def test_compute_moving_average_invalid_value_at_index_should_raise_exception(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        with pytest.raises(ValueError):
            pph.compute_moving_average(obj.iloc[0].values, 2)

    def test_compute_moving_average_invalid_greater_index_should_raise_exception(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        with pytest.raises(ValueError):
            pph.compute_moving_average(obj.iloc[0].values, 22)

    def test_compute_moving_average_invalid_negative_index_should_raise_exception(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        with pytest.raises(ValueError):
            pph.compute_moving_average(obj.iloc[0].values, -22)

    def test_compute_moving_average_should_return_correct_5(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 5.0
        actual = pph.compute_moving_average(obj.iloc[0].values, 1)
        assert actual == expected, "compute_moving_average returns wrong value at index."

    def test_compute_moving_average_should_return_correct__first_value5(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 5.0
        actual = pph.compute_moving_average(obj.iloc[1].values, 0)
        assert actual == expected, "compute_moving_average returns wrong value at index."

    def test_compute_moving_average_should_return_correct_second_value5(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 5.0
        actual = pph.compute_moving_average(obj.iloc[1].values, 1)
        assert actual == expected, "compute_moving_average returns wrong value at index."

    def test_compute_moving_average_should_return_correct_third_value10(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 10.0
        pph.compute_moving_average(obj.iloc[1].values, 1)
        actual = obj.iloc[1].values[2]
        assert actual == expected, "compute_moving_average returns wrong value at index."

    def test_compute_moving_average_should_return_correct_third_value3(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 3.0
        actual = pph.compute_moving_average(obj.iloc[4].values, 2)
        assert actual == expected, "compute_moving_average returns wrong value at index."


class TestFillNaN(object):

    def test_fill_nan_should_return_correct_value_at_position_1_1(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 5
        actual = pph.fill_nan_values(obj.reset_index()).iloc[1, 1]
        assert actual == expected, "Fill NaN doesn't calculate the expected result."

    def test_fill_nan_should_return_correct_value_at_position_1_2(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 5
        actual = pph.fill_nan_values(obj.reset_index()).iloc[1, 2]
        assert actual == expected, "Fill NaN doesn't calculate the expected result."

    def test_fill_nan_should_return_correct_value_at_position_4_3(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 3
        actual = pph.fill_nan_values(obj.reset_index()).iloc[4, 3]
        assert actual == expected, "Fill NaN doesn't calculate the expected result."

    def test_fill_nan_should_return_correct_value_at_position_4_2(self):
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        expected = 3
        actual = pph.fill_nan_values(obj.reset_index()).iloc[4, 2]
        assert actual == expected, "Fill NaN doesn't calculate the expected result."

    def test_fill_nan_should_raise_exception_for_none_as_input(self):
        obj = None
        with pytest.raises(ValueError):
            pph.fill_nan_values(obj)


class TestPredictFutureYearlyData(object):

    def test_predict_future_yearly_data_should_return_correct_columns_count(self):
        future_years = [2022, 2023]
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        obj = pph.fill_nan_values(obj.reset_index()).melt(
            id_vars=['Name'],
            var_name='Year',
            value_vars=obj.columns,
            value_name='Grade')
        expected = len(obj['Year'].unique()) + len(future_years)
        future_data = pph.predict_future_yearly_data(obj, future_years, 'Name', 'Year', 'Grade')
        actual = len(pd.concat([obj, future_data])['Year'].unique())

        assert actual == expected, "Predict Future Yearly Data doesn't calculate the expected result."

    def test_predict_future_yearly_data_should_return_same_columns_count(self):
        future_years = []
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        obj = pph.fill_nan_values(obj.reset_index()).melt(
            id_vars=['Name'],
            var_name='Year',
            value_vars=obj.columns,
            value_name='Grade')
        expected = len(obj['Year'].unique()) + len(future_years)
        future_data = pph.predict_future_yearly_data(obj, future_years, 'Name', 'Year', 'Grade')
        actual = len(pd.concat([obj, future_data])['Year'].unique())

        assert actual == expected, "Predict Future Yearly Data doesn't calculate the expected result."

    def test_predict_future_yearly_data_should_raise_exception_for_none_as_input(self):
        future_years = []
        obj = None
        with pytest.raises(ValueError):
            pph.fill_nan_values(pph.predict_future_yearly_data(obj, future_years, 'Name', 'Year', 'Grade'))

    def test_predict_future_yearly_data_should_raise_exception_for_invalid_column_names_1(self):
        future_years = []
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        obj = pph.fill_nan_values(obj.reset_index()).melt(
            id_vars=['Name'],
            var_name='Year',
            value_vars=obj.columns,
            value_name='Grade')
        with pytest.raises(ValueError):
            pph.fill_nan_values(pph.predict_future_yearly_data(obj, future_years, invalid_column_name, 'Year', 'Grade'))

    def test_predict_future_yearly_data_should_raise_exception_for_invalid_column_names_2(self):
        future_years = []
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        obj = pph.fill_nan_values(obj.reset_index()).melt(
            id_vars=['Name'],
            var_name='Year',
            value_vars=obj.columns,
            value_name='Grade')
        with pytest.raises(ValueError):
            pph.fill_nan_values(pph.predict_future_yearly_data(obj, future_years, 'Name', invalid_column_name, 'Grade'))

    def test_predict_future_yearly_data_should_raise_exception_for_invalid_column_names_3(self):
        future_years = []
        obj = pph.create_pivot_from_df(test_df, 'Name', 'Year', 'Grade')
        obj = pph.fill_nan_values(obj.reset_index()).melt(
            id_vars=['Name'],
            var_name='Year',
            value_vars=obj.columns,
            value_name='Grade')
        with pytest.raises(ValueError):
            pph.fill_nan_values(pph.predict_future_yearly_data(obj, future_years, 'Name', 'Year', invalid_column_name))
