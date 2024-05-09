import pandas as pd
import statsmodels.api as sm
import numpy as np


def create_pivot_from_df(df: pd.DataFrame, index: str, columns: str, values: str):
    """
    Transforms (pandas.DataFrame) object into (pandas.DataFrame) pivot object. The data is aggregated based on the index
    and columns parameters.

    :param df: (pandas.DataFrame) which will be used for the creation of the pivot.
    :param index: text value which will be used as index parameter for the pivot.
    :param columns: text value which will be used as columns parameter for the pivot.
    :param values: text value which will be used as values parameter for the pivot.
    :return:
        (pandas.DataFrame) pivot like object.
    """
    if index not in df.columns or columns not in df.columns or values not in df.columns:
        raise ValueError('Invalid column data passed.')
    return df.groupby([index, columns])[values] \
        .mean() \
        .reset_index() \
        .pivot(index=index, columns=columns, values=values)


def compute_moving_average(arr: np.array, idx: int):
    """
        Computes the average value of missing np.nan value at idx from the array.
    :param arr: (np.array) object containing (np.nan) missing values.
    :param idx: (int) index of the (np.nan) value.
    :return:
        The mean of the first and last values between the given index.
    """
    if idx < 0 or idx > len(arr):
        raise ValueError('Index cannot be negative number or with value higher than the length of the array.')
    if not np.isnan(arr[idx]):
        raise ValueError('Value at index is not NaN')
    first = 0
    last = 0
    if idx > 0:
        before = arr[:idx][~np.isnan(arr[:idx])]
        first = before[-1] if len(before) != 0 else 0
    if idx < len(arr) - 1:
        after = arr[idx:][~np.isnan(arr[idx:])]
        last = after[0] if len(after) != 0 else 0
    return (first + last) / 2


def fill_nan_values(pivot: pd.DataFrame):
    """
        Fills (np.nan) values from pivot (pd.DataFrame) object.
    :param pivot: (pd.DataFrame) pivot object containing (np.nan) values.
    :return:
        Returns (pd.DataFrame) pivot object without (np.nan) values.
    """
    if pivot is None:
        raise ValueError('DataFrame cannot be None. Cannot fill NaN values of NoneType.')
    pivot_numpy = pivot.to_numpy()
    for i, arr in enumerate(pivot_numpy):
        temp = [arr[0]]
        arr = arr[1:].astype(float)
        for j, el in enumerate(arr):
            t = el if not np.isnan(el) else compute_moving_average(arr, j)
            temp.append(t)
        pivot_numpy[i] = temp

    return pd.DataFrame(pivot_numpy,
                        index=pivot.index,
                        columns=pivot.columns)


def predict_future_yearly_data(pivot: pd.DataFrame, future_years: list, type: str, index: str, value: str):
    """
    Predicts future values for the given param future_years by using ARIMA (Autoregressive Integrated Moving Average)
    model.

    :param pivot: (pd.DataFrame) preprocessed pivot object containing the values for each param type
    :param future_years: (list) containing the future years for which a prediction will be made
    :param type: variable used as a column name for extracting unique values and filtering values from the param pivot
    :param index: usually called `year` this parameter will be used as a column name for getting the yearly based data from the param pivot
    :param value: usually the column name from the pivot table used as a `value`, because the param pivot should be in a pivot like (pd.DataFrame) object format
    :return:
        (pd.DataFrame) object containing the future data generated with the same format as the passed param pivot.
    """
    if pivot is None:
        raise ValueError('DataFrame cannot be None. Cannot fill NaN values of NoneType.')
    if index not in pivot.columns or type not in pivot.columns or value not in pivot.columns:
        raise ValueError('Invalid column data passed.')

    if len(future_years) == 0:
        return pivot
    future_data = {type: [], index: [], value: []}

    for c_neighbourhood in pivot[type].unique():
        temp_df = pivot.loc[pivot[type] == c_neighbourhood].sort_values([type, index])
        temp_df.index = pd.to_datetime(temp_df[index], format='%Y')

        new_df = pd.DataFrame(temp_df[value].resample('y').mean()).interpolate(method='linear')

        mod = sm.tsa.arima.ARIMA(new_df,
                                 order=(1, 1, 1))
        results = mod.fit()

        forecast = results.get_forecast(steps=len(future_years)).conf_int().mean(axis=1)

        for idx, year in enumerate(future_years):
            future_data[type].append(c_neighbourhood)
            future_data[index].append(year)
            future_data[value].append(forecast[idx])
    return pd.DataFrame(future_data)
