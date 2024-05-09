from src.utils import preprocess_helper as pph
import pandas as pd
import numpy as np

MIN_YEAR_INCLUSIVE = 2014
MAX_YEAR_INCLUSIVE = 2020
MIN_SCALE, MAX_SCALE = 1, 21


def preprocess_nuisance(df):
    """
     Preprocesses the dataframe by applying different filters.
    :param df:  (pd.dataFrame) object.
    :return:
    Returns filtered (pd.DataFrame) object.
    """
    print("Preprocessing nuisance data...")
    df['year'] = df['Perioden'].str[:4].astype(int)
    df['Neighbourhood'] = df['WijkenEnBuurten']
    df['nuisance'] = df['GeregistreerdeOverlast_1'].astype(float)

    return df.loc[
        (df['Perioden'].str.len() > 4) &
        (df['year'] >= MIN_YEAR_INCLUSIVE) & (df['year'] <= MAX_YEAR_INCLUSIVE) &
        (df['Overlast'].str.contains(pat='Totaal'))
        ].groupby(['year', 'Neighbourhood'])['nuisance'].mean().reset_index()


def preprocess_houses(df):
    """
   Preprocesses the dataframe by predicting data for future years using the global max/min year inclusive
   variables.
   :param df: (pd.dataFrame) object.
   :return:
   Returns equalized with the others (pd.DataFrame) object.
   """
    print("Preprocessing house data...")
    df['year'] = df['time_period'].astype(str).str[:4].astype(int)
    df = df[['Neighbourhood', 'year', 'frequency', 'moving_type']].loc[~pd.isna(df['Neighbourhood'])]

    future_years = get_years_to_predict(df, 'year')
    future_data = pph.predict_future_yearly_data(df.reset_index(), future_years, 'Neighbourhood', 'year',
                                                 'frequency')

    types = df['moving_type'].unique()

    df['moving_type'] = df['moving_type'].apply(lambda x: np.squeeze(np.where(types == x)))
    future_moving_type_data = pph.predict_future_yearly_data(
        df.reset_index(), future_years, 'Neighbourhood', 'year',
        'moving_type')
    future_data['moving_type'] = future_moving_type_data['moving_type'].astype(int).apply(lambda x: types[x])
    df['moving_type'] = df['moving_type'].astype(int).apply(lambda x: types[x])

    processed_df = pd.concat([df, future_data]).groupby(['year', 'Neighbourhood', 'moving_type'])[
        'frequency'].mean().reset_index()

    return pd.pivot_table(processed_df.loc[processed_df['year'] >= MIN_YEAR_INCLUSIVE], index=['Neighbourhood', 'year'],
                          columns='moving_type',
                          values='frequency',
                          fill_value=0).reset_index()


def preprocess_green_index(df):
    """
    Fills missing values through the years and preprocesses the dataframe by predicting data for future years using the
     global max/min year inclusive
    variables.
    :param df: (pd.dataFrame) object.
    :return:
    Returns equalized with the others (pd.DataFrame) object without missing values.
    """
    print("Preprocessing green index data...")
    df['green_score'] = df['green_score'].astype(float)
    pivot = pph.create_pivot_from_df(df, 'Neighbourhood', 'year', 'green_score')
    pivot = pph.fill_nan_values(pivot.reset_index()).melt(
        id_vars=['Neighbourhood'],
        var_name='year',
        value_vars=pivot.columns,
        value_name='green_score')

    future_years = get_years_to_predict(pivot, 'year')
    future_data = pph.predict_future_yearly_data(pivot, future_years, 'Neighbourhood', 'year', 'green_score')

    processed_df = pd.concat([pivot, future_data])

    return processed_df.loc[processed_df['year'] >= MIN_YEAR_INCLUSIVE]


def get_years_to_predict(pivot: pd.DataFrame, column: str):
    """
     Calculates the yars which have to be predicted in order to equalize the dataframes time_periods base on the global
     variables _max_year and min_year inclusive.
    :param pivot: (pd.DataFrame) pivot like object.
    :param column: (str) column name from the passed param pivot.
    :return:
        Returns (list) containing the years which have to be predicted.
    """
    if pivot is None or column is None:
        raise ValueError('Invalid parameters detected. Cannot predict Years with these input parameters.')
    current_max = max(pivot[column])
    return list(range(current_max + 1, MAX_YEAR_INCLUSIVE + 1))


def merge_datasets(green_index: pd.DataFrame, move_houses: pd.DataFrame, public_nuisance: pd.DataFrame,
                   recorded_crimes: pd.DataFrame, livability_index: pd.DataFrame):
    """
    This method merges the given gren_index, move_houses, public_nuisance and recorder_crimes
    :param green_index: (pd.DataFrame) object containing a column name Neighbourhood and year
    :param move_houses: (pd.DataFrame) object containing a column name Neighbourhood and year
    :param public_nuisance: (pd.DataFrame) object containing a column name Neighbourhood and year
    :param recorded_crimes: (pd.DataFrame) object containing a column name WijkenEnBuurten and Perioden
    :return:
    Returns a merged (pd.DataFrame) with 'Neighbourhood' and 'year' as additional columns except the floating point ones
    """
    merged = green_index.merge(move_houses,
                               left_on='Neighbourhood',
                               right_on='Neighbourhood').merge(public_nuisance,
                                                               left_on='Neighbourhood',
                                                               right_on='Neighbourhood').merge(
        recorded_crimes,
        left_on='Neighbourhood',
        right_on='WijkenEnBuurten').merge(
        livability_index,
        left_on='Neighbourhood',
        right_on='Neighbourhoods')

    merged['year_y'] = merged['year_y'].astype(int)
    merged['year'] = merged['year'].astype(int)
    merged['year_x'] = merged['year_x'].astype(int)
    merged['Perioden'] = merged['Perioden'].astype(int)
    merged['Year'] = merged['Year'].astype(int)

    merged['GeregistreerdeMisdrijven_1'] = merged['GeregistreerdeMisdrijven_1'].astype(float)
    merged = merged.loc[(merged['year'] == merged['year_y']) & (merged['year_x'] == merged['year_y'])
                        & (merged['Perioden'] == merged['year_y']) & (merged['Year'] == merged['year_y'])].drop(
        ['year_y', 'year_x', 'WijkenEnBuurten', 'Perioden', 'Neighbourhoods', 'Year'],
        axis=1).reset_index(drop=True)

    return merged


def calculate_index_based_on_good_bad_cols(df, good_columns, bad_columns, is_scaled):
    """
    Caalculates the ratio between the sum of the good factors from the total sum of all factors as a percentage:
    Formula:
    (sum(good_factors)/sum(all_factors))*100
    :param df: (pd.DataFrame) containing the specified good and bad columns & their scaled version
    :param good_columns: (list) specifying the good columns from the (pd.DataFrame)
    :param bad_columns: (list) specifying the bad columns from the (pd.DataFrame)
    :param is_scaled: (boolean) identifying which values to get (scaled or not)
    :return:
    (pd.DataFrame) containing a livability index as additional column and without the scaled version of the original columns.
    Additionally, the (pd.DataFrame) contains _debug information to explain the calculations made,
    """
    if df is None or good_columns is None or bad_columns is None or len(good_columns) == 0 or len(bad_columns) == 0:
        raise ValueError('Invalid parameters detected. Cannot Calculate Index with these input parameters.')
    data = {'Neighbourhood': [], 'year': [], 'sum_debug': [], 'livability_score': [],
            'good_sum_debug': [], 'bad_sum_debug': []}
    for nidx, el in enumerate(df['Neighbourhood'].unique()):
        years = df.loc[df['Neighbourhood'] == el, 'year']

        for idx, year in enumerate(years):
            data['Neighbourhood'].append(el)
            data['year'].append(year)
            good_value = 0
            for e in good_columns:
                if is_scaled:
                    e = e[:-7]
                value = df.loc[
                    (df['Neighbourhood'] == el) & (df['year'] == year), e].values.squeeze()
                good_value += value
                if e not in data:
                    data[e] = []
                data[e].append(float(value))
            bad_value = 0
            for e in bad_columns:
                if is_scaled:
                    e = e[:-7]
                value = df.loc[
                    (df['Neighbourhood'] == el) & (df['year'] == year), e].values.squeeze()
                bad_value += value
                if e not in data:
                    data[e] = []
                data[e].append(float(value))

            good_value = good_value / len(good_columns)
            bad_value = bad_value / len(bad_columns)
            sum_all = good_value + bad_value
            data['sum_debug'].append(sum_all)
            score = (good_value / sum_all) * 100
            data['livability_score'].append(score)

            data['good_sum_debug'].append(good_value)
            data['bad_sum_debug'].append(bad_value)
    return pd.DataFrame(data)


def split_columns(df_columns, good_columns):
    """
    Based on the good_columns this method returns the bad ones from the total columns in a dataframe. Additionally, the method adds _suffix to the column name
    :param df_columns: (list) containing all columns of a (pd.DataFrame)
    :param good_columns: (list) containing the good columns of a (pd.DataFrame)
    :return:
    (list) containing the good and bad columns.
    """
    if df_columns is None or good_columns is None or len(good_columns) == 0 or len(df_columns) == 0:
        raise ValueError('Invalid parameters detected. Cannot split columns with these input parameters.')
    suffix = 'scaled'
    good_cols = []
    bad_cols = []
    for el in df_columns:
        transformed_column = '{0}_{1}'.format(el, suffix)
        if el in good_columns:
            good_cols.append(transformed_column)
        else:
            bad_cols.append(transformed_column)

    return good_cols, bad_cols


def scale_values(c_df: pd.DataFrame, column: str, min_scaled: int, max_scaled: int):
    """
    This method scales the given values based on the min_scaled and max_scaled parameters
    :param c_df: (pd.DataFrame) containing the values which need to be scaled
    :param column: (str) specifying which column should be scaled from the (pd.DataFrame)
    :param min_scaled: (int) specifying the min_scale value
    :param max_scaled: (int) specifying the max_scale value
    :return:
      The (pd.DataFrame) object passed aas input parameter, but the specified column is scaled and named with a suffix _scaled.
    """
    if c_df is None or column is None or min_scaled is None or max_scaled is None or max_scaled < min_scaled:
        raise ValueError('Invalid parameters detected. Cannot split columns with these input parameters.')
    min_value = min(c_df[column].values)
    max_value = max(c_df[column].values)
    temp = c_df.copy()
    temp[f'{column}_scaled'] = ((temp[column] - min_value) / (max_value - min_value)) * (
            max_scaled - min_scaled)
    return temp


def calculate_custom_livability_index(df):
    """
    Accepts a DataFrame & Calculates a custom livability index based on the specified column names
    Example:
    ['green_score', 'Vestiging', 'Verhuizing binnen gridcel', 'Verhuizing']
    :param df: (pd.DataFrame) object containing the specified columns and their scaled versions
    :return:
     (pd.DataFrame) object given as an input parameter without the scaled columns, but with a livability_score column
    """
    scaled_columns = []
    for idx, column_type in enumerate(df.dtypes):
        current_column = df.columns[idx]
        if (column_type == float or column_type == int) and \
                ('year' not in current_column and current_column != 'Livability index'):
            df = scale_values(df, df.columns[idx], MIN_SCALE, MAX_SCALE)
            scaled_columns.append(current_column)

    good_cols, bad_cols = split_columns(scaled_columns,
                                        ['green_score', 'Vestiging', 'Verhuizing binnen gridcel', 'Verhuizing'])
    result = calculate_index_based_on_good_bad_cols(df, good_cols,
                                                    bad_cols, is_scaled=True)

    result_columns = result.columns

    return result[result_columns[np.where(~result_columns.str.endswith('_debug'))]]
