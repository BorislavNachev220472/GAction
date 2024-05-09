from sqlalchemy import text, MetaData, Table, select
import pandas as pd
import numpy as np
from .decorators import debounced, injectable
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import plotly.express as px
import re
import json
import topojson
import os
import src.constants as constants


def read_tables(engine):
    """
    Reads tables from the database and returns a dictionary.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy database engine.

    Returns:
        dict: A dictionary mapping table names to pandas DataFrames.
    """
    tables = {}

    metadata = MetaData()
    metadata.reflect(bind=engine)

    table_names = metadata.tables.keys()

    for table_name in table_names:
        df = pd.read_sql_table(table_name, engine)
        tables[table_name] = df

    return tables


def get_metadata(engine):
    """
    Retrieves metadata from the database.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy database engine.

    Returns:
        sqlalchemy.MetaData: The metadata object representing the database schema.
    """

    metadata = MetaData()
    metadata.reflect(bind=engine)

    return metadata


@injectable
def table_reader(engine, table_name, **kwargs):
    """
    Injectable: Reads a specific table from the database.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy database engine.
        table_name (str): The name of the table to read.
        **kwargs: Additional keyword arguments to pass to the `exec_query` function.

    Returns:
        tuple: A tuple containing the SQLAlchemy Table object representing the table schema and
               a pandas DataFrame containing the table data.
    """
    metadata = get_metadata(engine)
    x_q = exec_query(engine, **kwargs)

    table = Table(table_name, metadata)

    data = {}
    for el in table.columns:
        stmt = select(el)
        records = x_q(stmt)
        data[el.name] = np.squeeze(records)

    return table, pd.DataFrame(data)


@debounced(sec=30)
def check_connection(engine, desc=""):
    """
    debounced: Checks the database connection every 30 seconds.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy database engine.
        desc (str, optional): An optional description to include in the success message.
    """
    test_query = text("SELECT 1")

    with engine.connect() as conn:
        result = conn.execute(test_query)

        if result.scalar() == 1:
            print(desc or "Connection successful!")
        else:
            print("Connection failed!")
            exit(-1)


@injectable
def exec_query(engine, q: str, v=True, desc=""):
    """
    Injectable: Executes a database query.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy database engine.
        q (str): The query string to execute.
        v (bool, optional): A boolean flag indicating whether to perform connection checking (default is True).
        desc (str, optional): An optional description to include in the connection check success message.

    Returns:
        list: A list of records resulting from the query.
    """
    if v:
        check_connection(engine, desc)

    with engine.connect() as conn:
        if isinstance(q, str):
            q = text(q)
        q_res = conn.execute(q)

        res = q_res.fetchall()

        return res


def data_typing(
    df,
    integers=[], floats=[],
    strings=[],
    categoricals=[], dates={}, only_years=[],
    only_months=[], only_days=[]
):
    """
    Performs data typing on a DataFrame.

    Parameters:
        df (pandas.DataFrame): The pandas DataFrame to perform data typing on.
        integers (list, optional): A list of column names to be typed as integers.
        floats (list, optional): A list of column names to be typed as floats.
        strings (list, optional): A list of column names to be typed as strings.
        categoricals (list, optional): A list of column names to be typed as categoricals.
        dates (dict, optional): A dictionary mapping date column names to their format strings (e.g., {'date_column': '%Y-%m-%d'}).
        only_years (list, optional): A list of date column names to extract only the year.
        only_months (list, optional): A list of date column names to extract only the month.
        only_days (list, optional): A list of date column names to extract only the day.

    Returns:
        pandas.DataFrame: The modified DataFrame with the specified column types applied.
    """
    df = df.copy()

    for col in df:
        if col in integers:
            df[col] = pd.to_numeric(
                df[col], downcast='integer', errors="coerce")
            continue

        if col in floats:
            df[col] = pd.to_numeric(df[col].str.replace(
                ',', '.'), downcast='float', errors="coerce")
            continue

        if col in categoricals:
            df[col] = df[col].astype('category')
            continue

        if col in strings:
            df[col] = df[col].astype('string')
            continue

        if col in dates.keys():
            df[col] = pd.to_datetime(df[col], format=dates.get(col))

            if col in only_years:
                df[col] = df[col].dt.to_period('Y')

            if col in only_months:
                df[col] = df[col].dt.to_period('M')

            if col in only_days:
                df[col] = df[col].dt.to_period('D')
            continue

    return df


@injectable
def visualize_map(grid, filepath, num_col, cat_col, cat=None, quantiles=[0.02, 0.98], figsize=(10, 10), *args, **kwargs):
    ext_pattern = re.compile('^(?:\S+)*\.(\S+)$')
    ext = ext_pattern.match(filepath).group(1)

    if ext == 'csv':
        gdf = pd.read_csv(filepath, *args, **kwargs)
        gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
        gdf_data = gpd.GeoDataFrame(gdf, geometry='geometry')
    else:
        gdf_data = gpd.read_file(filepath)

    gdf_data.set_crs('epsg:4326')

    if cat and cat_col:
        gdf_data = gdf_data.loc[gdf_data[cat_col] == cat]

    q_1 = quantiles[0]
    q_2 = quantiles[1]

    q_low = gdf_data[num_col].quantile(q_1) if quantiles else gdf_data[num_col]
    q_high = gdf_data[num_col].quantile(
        q_2) if quantiles else gdf_data[num_col]

    _, ax = plt.subplots(figsize=figsize)

    gdf_data = gdf_data.loc[gdf_data[num_col] > q_low]
    gdf_data = gdf_data.loc[gdf_data[num_col] < q_high]

    gdf_data.plot(ax=ax, column=num_col, cmap='Reds', legend=True)
    grid.exterior.plot(
        ax=ax, edgecolor='gray',
        facecolor='none', linewidth=0.08
    )

    plt.show()


def load_breda_gdf():
    gdf = gpd.read_file(os.path.abspath(os.path.join(
        constants.PATH, constants.RAW_DIRECTORY, constants.NEIGHBOURHOOD_DATASET_NAME)))
    gdf = gdf.loc[gdf.gemeentenaam == 'Breda']
    gdf = clean_GeoDataFrame(gdf)
    gdf = gdf.drop_duplicates('buurtnaam')
    return gdf


@injectable
def make_st_map(df, gdf, col_color, locations, score, hover_data=[
    'Neighbourhoods',
    'green_score',
    'GeregistreerdeOverlast_1',
    'moving_frequency',
    'GeregistreerdeMisdrijven_1',
    'Livability index',
]):
    fig = px.choropleth(
        data_frame=df,
        geojson=json.loads(gdf.to_json()), color=col_color,
        locations=locations,
        featureidkey="properties.buurtnaam",
        projection='mercator',
        color_continuous_scale="Viridis",
        animation_frame='year',
        animation_group='moving_type',
        labels={
            'Neighbourhoods': "neighbourhood",
            'green_score': 'green score',
            'GeregistreerdeOverlast_1': "nuisance",
            'moving_frequency': 'moving frequency',
            'GeregistreerdeMisdrijven_1': 'crime score',
            'Livability index': 'liveability score'
        },
        hover_data=hover_data
    )

    fig.update_geos(fitbounds="geojson", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        coloraxis_colorbar=dict(
            orientation='h',
            x=0.5,
            y=-0.1,
            thickness=10,
            len=1,
            title=dict(side="bottom", text=score)
        )
    )

    return fig


def clean_GeoDataFrame(gdf, epsg=4326, geo_col=None, district_col=None):
    district_col = district_col or 'buurtnaam'
    geo_col = geo_col or 'geometry'

    try:
        gdf = gdf.to_crs(epsg=epsg)
    except ValueError:
        gdf = gdf.set_crs(epsg=epsg)
        gdf = gdf.to_crs(epsg=epsg)

    gdf = gdf.filter([geo_col, district_col])

    gdf[district_col] = gdf[district_col].map(str.strip)
    gdf = gdf.drop_duplicates(district_col)

    gdf = gdf.loc[gdf[district_col] != '']

    return gdf


def export_topoJSON(to_path, gdf, epsg=4326, geo_col=None, district_col=None):
    district_col = district_col or 'buurtnaam'
    geo_col = geo_col or 'geometry'

    clean_gdf = clean_GeoDataFrame(gdf, epsg, geo_col, district_col)

    clean_gdf['area'] = clean_gdf[geo_col].area
    clean_gdf['length'] = clean_gdf[geo_col].length

    result = clean_gdf.to_json()
    result = topojson.Topology(result).output

    with open(to_path, 'w') as fd:
        json.dump(result, fd)
