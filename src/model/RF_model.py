from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

import src.helper as helper
import src.constants as constants
import joblib


def train_random_forest_model(df):
    """
    Trains Random Forest Regression Model and saves it into .pickle file.
    :param df (pd.DAtaFrame containing model's data.
    :return:
    NoneType
    """
    df['moving_out'] = df['Vertrek']
    df['moving_in'] = (df['Vestiging'] + df['Verhuizing binnen gridcel'] + df['Verhuizing']) / 3
    df = df.loc[df['Neighbourhood'] != 'Hoogeind']

    neighborhoods = df['Neighbourhood'].unique()
    data = {}

    for neighborhood in neighborhoods:
        data[neighborhood] = []
        df_neighborhood = df[df['Neighbourhood'] == neighborhood]
        X = df_neighborhood[
            ['green_score', 'nuisance', 'GeregistreerdeMisdrijven_1',
             'moving_in', 'moving_out']]
        y = df_neighborhood['Livability index'].to_numpy()

        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        model_name = f'{neighborhood}{constants.RF_FOREST_MODEL_SUFFIX}'
        joblib.dump(model, helper.generate_path(helper.generate_path(constants.PATH, constants.MODEL_DIRECTORY),
                                                model_name))
        print(f'Saving Random Forest Model for {neighborhood}.')
        importances = model.feature_importances_
        data[neighborhood].append(importances)

        split = 5
        X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        data[neighborhood].append(preds)
        data[neighborhood].append(y_test)
        data[neighborhood].append(mean_squared_error(y_test, preds))

    final_df = pd.DataFrame(data).T
    final_df.columns = ['Feature importance', 'pred', 'actual', 'mse']
