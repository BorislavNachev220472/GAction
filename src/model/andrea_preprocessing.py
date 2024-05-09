from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

MAPPING = {
    'Onbekend': 1,
    'Zwak': 2,
    'Onvoldoende': 3,
    'Ruim onvoldoende': 4,
    'Voldoende': 5,
    'Ruim voldoende': 6,
    'Goed': 7,
    'Zeer goed': 8,
    'Uitstekend': 9
}


def preprocess_livability(df):
    """
    Preprocesses livability data by performing the following steps:
    1. Maps values in the '2002', '2008', '2012', '2014', '2016', '2018', and '2020' columns using the 'MAPPING' dictionary.
    2. Drops the '2002', '2008', and '2012' columns from the DataFrame.
    3. Creates a subset of the '2014', '2016', '2018', and '2020' columns.
    4. Uses SimpleImputer to fill in missing values in the subset with the mean of the respective column.
    5. Assigns the imputed values to the '2015', '2017', and '2019' columns in the original DataFrame.
    6. Rounds the values in the '2015', '2017', and '2019' columns.
    7. Converts the rounded values in the '2015', '2017', and '2019' columns to integers.
    8. Rearranges the columns in the DataFrame to ['Neighbourhoods', '2014', '2015', '2016', '2017', '2018', '2019', '2020'].
    9. Melts the DataFrame to have a single 'Year' column and a 'Livability index' column.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing the livability data.

    Returns:
        - processed_df (pandas.DataFrame): Preprocessed DataFrame with the following columns:
            - 'Neighbourhoods': Name of the neighborhood.
            - 'Year': Year of the livability index.
            - 'Livability index': Livability index value.

    Example Usage:
        processed_data = preprocess_livability(input_df)
    """

    print("Preprocessing livability data...")
    df['2002'] = df['2002'].map(MAPPING)
    df['2008'] = df['2008'].map(MAPPING)
    df['2012'] = df['2012'].map(MAPPING)
    df['2014'] = df['2014'].map(MAPPING)
    df['2016'] = df['2016'].map(MAPPING)
    df['2018'] = df['2018'].map(MAPPING)
    df['2020'] = df['2020'].map(MAPPING)
    # dropping unnecessary columns
    df = df.drop(['2002', '2008', '2012'], axis=1)
    # creating a subset of the score per years
    subset = df[['2014', '2016', '2018', '2020']]
    # initializng the imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # fiting the imputer to the known data
    imputer.fit(subset)
    # Transforming the subset data to fill in the missing values
    subset_imputed = imputer.transform(subset)
    # Creating a new DataFrame with the imputed values
    imputed_df = pd.DataFrame(subset_imputed, columns=['2014', '2016', '2018', '2020'])
    # list of the years we want to impute
    years_to_impute = ['2015', '2017', '2019']
    # Iterating over the years and assigning the imputed values back to the original df

    for year in years_to_impute:
        df[year] = imputed_df.mean(axis=1)
    # rounding the score
    df['2019'] = df['2019'].round()
    df['2017'] = df['2017'].round()
    df['2015'] = df['2015'].round()
    df['2019'] = df['2019'].astype(int)
    df['2017'] = df['2017'].astype(int)
    df['2015'] = df['2015'].astype(int)
    # rearanging the columns
    column_order = ['Neighbourhoods', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    return df.reindex(columns=column_order).melt(id_vars=["Neighbourhoods"],
                                                 var_name="Year",
                                                 value_name="Livability index")


def preprocess_crimes(df):
    """
    Preprocesses crime data by performing the following steps:
    1. Extracts the year from the 'Perioden' column and removes any leading/trailing whitespace.
    2. Groups the DataFrame by 'Perioden', 'WijkenEnBuurten', and 'SoortMisdrijf', and calculates the sum of 'GeregistreerdeMisdrijven_1' column for each group.
    3. Resets the index of the grouped DataFrame.
    4. Filters out rows where 'GeregistreerdeMisdrijven_1' is not equal to 0.
    5. Filters out rows where 'WijkenEnBuurten' is not equal to 'Breda'.
    6. Filters out rows where 'SoortMisdrijf' is not equal to 'Totaal misdrijven'.
    7. Sorts the DataFrame based on the 'Perioden' column in ascending order.
    8. Filters out rows where 'Perioden' is greater than '2013'.
    9. Returns the preprocessed DataFrame.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing the crime data.

    Returns:
        - processed_df (pandas.DataFrame): Preprocessed DataFrame with the following columns:
            - 'Perioden': Year of the crime data.
            - 'WijkenEnBuurten': Name of the neighborhood or district.
            - 'SoortMisdrijf': Type of crime.
            - 'GeregistreerdeMisdrijven_1': Total number of registered crimes.

    Example Usage:
        processed_data = preprocess_crimes(input_df)
    """
    print("Preprocessing crime data...")
    df['Perioden'] = df['Perioden'].str[:4].str.strip()
    grouped_df = df.groupby(['Perioden', 'WijkenEnBuurten', 'SoortMisdrijf']).sum()['GeregistreerdeMisdrijven_1']
    grouped_df = grouped_df.reset_index()
    grouped_df2 = grouped_df[grouped_df['GeregistreerdeMisdrijven_1'] != 0]
    grouped_df2 = grouped_df2[grouped_df2['WijkenEnBuurten'] != 'Breda']
    grouped_df2 = grouped_df2[grouped_df2['SoortMisdrijf'] == 'Totaal misdrijven']
    grouped_df2 = grouped_df2.sort_values('Perioden')
    grouped_df2 = grouped_df2.loc[grouped_df['Perioden'] > str(2013)]
    grouped_df2 = grouped_df2.drop(['SoortMisdrijf'], axis=1)
    return grouped_df2
