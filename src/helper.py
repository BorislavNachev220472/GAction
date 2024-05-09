from sqlalchemy import create_engine
import xml.etree.ElementTree as ET
from shapely.geometry import Point
from bs4 import BeautifulSoup
import geopandas as gpd
import pandas as pd
import shapely.wkt
import osmnx as ox
import numpy as np
import requests
import re
import os

# Local Utils
from src.utils import table_reader
import src.constants as constants


def generate_path(path_to_directory: str, file_name: str):
    """
    Generates the absolute PATH to a file based on the given parameters
    :param path_to_directory: (str) relative PATH to the folder containing the :param file_name
    :param file_name: (str) the name of the file which is in the param path_to_directory
    :return:
        (str) The combination of the path_to_directory and file_name.
    """
    if path_to_directory is None or file_name is None or path_to_directory.strip() == "" or file_name.strip() == "":
        raise ValueError('Invalid parameters detected. Cannot generate PATH with these input parameters.')
    return f'{path_to_directory}/{file_name}'


def prepare_data_directory():
    """
    Prepares the data directory by creating the required folders if they do not exist.
    :return:
        NoneType
    """
    print("Preparing data directory...")
    exist = os.path.exists(constants.PATH)
    if not exist:
        os.makedirs(constants.PATH)
    for folder in constants.DATA_FOLDERS:
        if not os.path.exists(generate_path(constants.PATH, folder)):
            os.makedirs(generate_path(constants.PATH, folder))


def save_df(df: pd.DataFrame, df_name: str, is_preprocessed: bool, **kwargs):
    """
    Saves the passed dataframe to its corresponding directory depending on its state. The state is defined by the
    :param is_preprocessed.
    :param df: (pd.DataFrame) object.
    :param df_name: (str) the name of the dataframe
    :param is_preprocessed: (bool) the state of the dataframe
    :return:
        NoneType
    """
    if df is None or df_name is None:
        raise ValueError('Invalid parameters detected. Cannot save DataFrame with these input parameters.')
    is_final = kwargs.get('is_final', False)
    if is_final == True:
        df_path = generate_path(constants.PATH, constants.FINAL_DIRECTORY)
        df.to_csv(generate_path(df_path, df_name), sep=constants.SEPARATOR, index=constants.INCLUDE_INDEX)
    if is_preprocessed:
        df_path = generate_path(constants.PATH, constants.PREPROCESSED_DIRECTORY)
        df.to_csv(generate_path(df_path, df_name), sep=constants.SEPARATOR, index=constants.INCLUDE_INDEX)
        print(f"Saving {constants.PREPROCESSED_DIRECTORY} data...")
    else:
        df_path = generate_path(constants.PATH, constants.RAW_DIRECTORY)
        df.to_csv(generate_path(df_path, df_name), sep=constants.SEPARATOR, index=constants.INCLUDE_INDEX)
    return df_path


def get_full_value(arr, value: str):
    """
    Transforming the given value to it's full definition using the metadata.
    :param arr: (list) containing the 'Key'(short definition) and 'Title'(full definition)
    which wil be used to transform the param value.
    :param value: (str) parameter which will be transformed to it's full definition.
    :return:
        The full definition of the param value
    """
    if arr is None or value is None:
        raise ValueError('Invalid parameters detected. Cannot Get Full Value with these input parameters.')
    arr = arr[1]['value']
    result = ''
    for elm in arr:
        if value in elm['Key']:
            result = elm['Title']
            break
    return result


def download_metadata(code: str, columns_to_transform):
    """
    Downloads metadata based on the input parameters and saves the fetched information to (dict).
    :param code: (str) containing the corresponding code of city from the Netherlands
    :param columns_to_transform: (list) containing the column names which need to be transformed.
    :return:
    (dict) containing key:value pairs with the metadata in the format specified below.
    Example:
     {'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
            'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                    'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]}
    """
    if code is None or columns_to_transform is None or len(columns_to_transform) == 0 or len(code) == 0:
        raise ValueError('Invalid parameters detected. Cannot Download MetaData with these input parameters.')
    metadata = {}
    for key in columns_to_transform:
        url = f'https://dataderden.cbs.nl/ODataApi/odata/{code}/{key}'
        r = requests.get(url)
        value = r.json()
        if key not in metadata:
            metadata[key] = [value]
        metadata[key].append(value)
    return metadata


def download_data_from_police(link_to_fetch: str, code: str, columns_to_transform, desc: str = ""):
    """
    Downloads data from the https://dataderden.cbs.nl website based on the given parameters using pagination
    :param link_to_fetch: (str) link to the first pagination link to the dataset which will be downloaded.
    :param code: (str) city code for which the data will be downloaded
    :param columns_to_transform: (list) containing the columns which have to be transformed by using the metadata.
    :param desc: (str) parameter used to log the state of the function.
    :return:
    Returns (pd.DataFrame) with the data downloaded from https://dataderden.cbs.nl.
    """
    if code is None or columns_to_transform is None or link_to_fetch is None or len(columns_to_transform) == 0 \
            or len(code) == 0 or len(link_to_fetch) == 0:
        raise ValueError('Invalid parameters detected. Cannot Download Data From Police with these input parameters.')

    print(desc)
    data = {}
    metadata = download_metadata(code, columns_to_transform)

    while True:
        r = requests.get(link_to_fetch)
        is_next = False

        a = ET.fromstring(r.text)
        for elem in a.iter():
            if 'content' in elem.tag:
                all_descendants = list(elem.iter())
                for el in all_descendants:
                    if el.text is None:
                        el.text = '0'
                    if el.text.strip():
                        key = el.tag.split("}")[1]
                        value = el.text
                        if key in columns_to_transform:
                            value = get_full_value(metadata[key], value)
                        if key not in data:
                            data[key] = []
                        data[key].append(value)
            if 'link' in elem.tag and 'next' in elem.attrib['rel']:
                is_next = True
                link_to_fetch = elem.attrib['href']
        print(link_to_fetch)
        if not is_next:
            break
    return pd.DataFrame(data)


def download_file(url: str, file_name: str, desc: str = ""):
    """
    Downloads and saves files to the file system based on the given parameters.
    :param url: (str) containing the url to the file
    :param file_name: (str) the file_name which the downloaded file will have.
    :param desc: (str) parameter used to log the state of the function.
    :return:
     NoneType
    """
    if url is None or file_name is None or len(url) == 0 or len(file_name) == 0:
        raise ValueError('Invalid parameters detected. Cannot Download File with these input parameters.')

    print(desc)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        raw_path = generate_path(constants.PATH, constants.RAW_DIRECTORY)
        with open(generate_path(raw_path, file_name), 'wb') as f:
            f.write(bytes(response.content))
    else:
        raise ConnectionAbortedError(f"Error accessing the {file_name} file. The file might was renamed/removed/moved.")


def get_articles(url: str):
    """
    Method which uses the library 'BeautifulSoup' to read and extract information from the raw .html file.
    :param url:  (str) containing the url to the website.
    :return:
    Returns the extracted information as a ResultSet of PageElements.
    """
    if url is None or len(url) == 0:
        raise ValueError('Invalid parameters detected. Cannot Download MetaData with these input parameters.')
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(bytes(response.content), 'html.parser')
        articles = soup.find_all('article')
        return articles
    else:
        raise ConnectionAbortedError(f"Error accessing the {url}.")


def get_neighbourhood_on_coordinates(neighbourhood, points, df: pd.DataFrame, points_column_names):
    """
    Checks if the given param points are within the param neighbourhood shape and sets the corresponding Neighbourhood
    to the positive results.
    :param neighbourhood: (list) containing (shapley.POLYGON) the shape data and (name) for each neighbourhood.
    :param points: (list) containing the points which will be connected to Neighbourhood.
    :param df: (pd.DataFrame) the whole dataframe in which the corresponding Neighbourhood will be set to the
    'Neighbourhood' column
    :param points_column_names: (list) containing the column names corresponding to of the latitude and longitude.
    :return:
    NoneType. The data is being changed using the reference of the param df
    """
    if neighbourhood is None or len(neighbourhood) == 0 or points is None or len(
            points) == 0 or df is None or points_column_names is None or len(points_column_names) == 0:
        raise ValueError('Invalid parameters detected. Cannot Map Coordinates with these input parameters.')
    for idx, coordinates in enumerate(points):
        if coordinates.within(neighbourhood[0]):
            df.loc[
                (df[points_column_names[0]] == coordinates.x) & (df[points_column_names[1]] == coordinates.y),
                'Neighbourhood'
            ] = neighbourhood[1]


def get_neighbourhood_on_geometry(neighbourhood, df: pd.DataFrame):
    """
     Checks if the given param neighbourhood is within the 'geometry' values in the param df. If so, the method sets
     the corresponding Neighbourhood to the positive results
    :param neighbourhood: (list) containing (shapley.POLYGON) the shape data and (name) for each neighbourhood.
    :param df: (pd.DataFrame) the whole dataframe in which the corresponding Neighbourhood will be set to the
    'Neighbourhood' column
    :return:
    NoneType. The data is being changed using the reference of the param df
    """
    if neighbourhood is None or len(neighbourhood) == 0 or df is None:
        raise ValueError('Invalid parameters detected. Cannot Map Geometry with these input parameters.')
    for house in df['geometry'].values:
        if type(house) == str:
            house = shapely.wkt.loads(house)
        if house.within(neighbourhood[0]):
            df.loc[df['geometry_temp'] == str(house), 'Neighbourhood'] = neighbourhood[1]


def map_geometries_to_neighbourhoods(coordinate_dfs, coordinate_columns_dfs, geometry_dfs):
    """
    This function maps many dataframes containing coordinates or geometries to their corresponding Neighbourhoods
    :param coordinate_dfs: (list) containing the dataframes with coordinates. The name of the coordinate columns is
    strictly defined
    :param coordinate_columns_dfs: (list) containing inner (list) of the column names for the coordinate dataframe
    :param geometry_dfs: (list) containing the dataframes with coordinates. The name of the geometry column is
    strictly defined: 'geometry'
    :return:
        NoneType. The dataframes are being changed by reference.
    """
    if coordinate_dfs is None or len(coordinate_dfs) == 0 or coordinate_columns_dfs is None or len(
            coordinate_columns_dfs) == 0 or geometry_dfs is None or len(geometry_dfs) == 0:
        raise ValueError('Invalid parameters detected. Cannot Map Geometries with these input parameters.')

    print('Mapping neighbourhoods based on coordinate and geometry data...')
    neighbourhoods = gpd.read_file(generate_path(generate_path(constants.PATH, constants.RAW_DIRECTORY),
                                                 constants.NEIGHBOURHOOD_DATASET_NAME))
    neighbourhoods = neighbourhoods.to_crs(epsg=4326)

    points = []
    for idx in range(len(coordinate_dfs)):
        temp = [Point(el[0], el[1]) for el in coordinate_dfs[idx][coordinate_columns_dfs[idx]].drop_duplicates().values]
        points.append(temp)
        coordinate_dfs[idx]['Neighbourhood'] = ''
        print(f'Preparing Coordinate dataset {idx}...')

    for idx in range(len(geometry_dfs)):
        geometry_dfs[idx]['geometry_temp'] = geometry_dfs[idx]['geometry'].map(lambda e: str(e))
        geometry_dfs[idx]['Neighbourhood'] = ''
        print(f'Optimising Geometry dataset {idx}...')

    for neighbourhood in neighbourhoods[neighbourhoods['gemeentenaam'] == 'Breda'][['geometry', 'buurtnaam']].values:
        for idx in range(len(coordinate_dfs)):
            get_neighbourhood_on_coordinates(neighbourhood, points[idx], coordinate_dfs[idx],
                                             coordinate_columns_dfs[idx])

        for idx in range(len(geometry_dfs)):
            get_neighbourhood_on_geometry(neighbourhood, geometry_dfs[idx])
        print(f'Pre-processing has been done for {neighbourhood[1]} district.')

    for idx in range(len(geometry_dfs)):
        geometry_dfs[idx].drop('geometry_temp', axis=1, inplace=True)
        print(f'Cleaning Geometry dataset {idx}...')

    neighbourhoods = neighbourhoods.loc[neighbourhoods['gemeentenaam'] == 'Breda']
    neighbourhoods.to_file(generate_path(generate_path(constants.PATH, constants.PREPROCESSED_DIRECTORY),
                                         constants.NEIGHBOURHOOD_DATASET_NAME), driver='GPKG')


def download_lights_data():
    """
    This function is used to encapsulate the downloading steps of the lights dataset.
    :return:
    NoneType
    """
    print("Downloading Lights data...")
    return pd.read_csv(constants.LIGHTS_DATASET_LINK)


def download_police_data():
    """
    This function is used to encapsulate the downloading steps of the police dataset.
    :return:
    NoneType
    """
    print("Downloading Police data...")
    patrols_data = {'names': [], 'contacts': [], 'locations': []}
    index = 1
    articles = get_articles(constants.POLICE_DATASET_LINK.format(''))

    while len(articles) != 0:

        for el in articles:
            name = el.find_all('h3')[0].text.strip()
            contact_info = el.find_all('dd')[0].text.strip()
            location_info = el.find_all('dd')[1].text.strip()
            patrols_data['names'].append(name)
            patrols_data['contacts'].append(contact_info)
            patrols_data['locations'].append(location_info)

        index += 1
        articles = get_articles(constants.POLICE_DATASET_LINK.format(f'page={index}&'))

    return pd.DataFrame(patrols_data)


def download_livability_data():
    """
    This function is used to encapsulate the downloading steps of the livability dataset.
    :return:
    NoneType
    """
    print("Downloading Livability index...")
    response = requests.get(constants.LIVABILITY_DATASET_LINK)
    txt = response.text

    columns = re.findall('<span\s*id="[\w+-]+">(?P<column_data>\d+)</span>', txt)

    rows_index = re.findall('<td class="tab-Buurt-name">(?P<district_data>[\w -]+)</td>', txt)
    rows_values = re.findall(
        '<td>[\s\t\n]*<span class="[\w\s-]+">\d*</span>[\s\t\n]*(?P<field_data>[\w -]+)[\s\t\n]*</td>',
        txt)

    return pd.DataFrame(np.array(rows_values).reshape((len(rows_index), len(columns))),
                        columns=columns,
                        index=rows_index).reset_index(names='Neighbourhoods')


def download_green_index_data():
    """
    This function is used to encapsulate the downloading steps of the green index dataset.
    :return:
    NoneType
    """
    print("Connecting to remote server...")
    print(constants.CONNECTION_STRING)
    engine = create_engine(constants.CONNECTION_STRING)

    read_table = table_reader(engine)
    green_index_table, green_index_data = read_table(constants.GREEN_INDEX_TABLE_NAME,
                                                     desc="Downloading Green Index...")

    green_index_data['latitude'] = green_index_data['latitude'].str.replace(',', '.').astype(float)
    green_index_data['longitude'] = green_index_data['longitude'].str.replace(',', '.').astype(float)
    return green_index_data


def download_poi_data_from_config():
    """
    This function is used to encapsulate the downloading steps of the POI dataset.
    :return:
    NoneType
    """
    config_file_path = generate_path(
        generate_path(constants.PATH, constants.RAW_DIRECTORY),
        constants.POI_CONFIG_FILE_NAME)
    poi_df = pd.read_csv(config_file_path,
                         skiprows=constants.POI_CONFIG_HEADER_LENGTH)[constants.POI_CONFIG_COLUMN_NAME]
    tags = {'amenity': poi_df.tolist()}
    os.remove(config_file_path)
    return ox.geometries_from_place(constants.POI_LOCATION, tags=tags)
