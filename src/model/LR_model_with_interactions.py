import statsmodels.formula.api as smf
import pandas as pd
import warnings

import src.helper as helper
import src.constants as constants

warnings.filterwarnings("ignore")


def train_lr_interaction(df):
    """
    Trains Linear Regression Model with Interaction Terms and saves it into .pickle file.
    :param df (pd.DAtaFrame containing model's data.
    :return:
    NoneType
    """

    df['moving_out'] = df['Vertrek']
    df['moving_in'] = (df['Vestiging'] + df['Verhuizing binnen gridcel'] + df['Verhuizing']) / 3
    df['liv'] = df['Livability index']

    df.drop(['Livability index', 'Vestiging', 'Vertrek', 'Verhuizing binnen gridcel', 'Verhuizing'], axis=1,
            inplace=True)
    df = pd.get_dummies(df)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')

    formula = df.columns[5] + ' ~ '

    for idx, column in enumerate(df.columns):
        if idx != 5:
            formula += column + ' + '

    for idx, column in enumerate(df.columns):
        if idx != 5:
            formula += column + ':'

    formula = formula[:-1]

    lr_model = smf.ols(formula=formula, data=df).fit()

    lr_model.save(helper.generate_path(helper.generate_path(constants.PATH, constants.MODEL_DIRECTORY),
                                       constants.LR_INTERACTION_MODEL_NAME))
    print(f'Saving Linear Regression with Interaction Terms.')
