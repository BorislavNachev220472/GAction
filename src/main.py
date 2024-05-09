import geopandas as gpd
import pandas as pd

# Local Utils
import src.helper as helper
import src.constants as constants
import src.model.andrea_preprocessing as ap
import src.model.borislav_preprocessing as bp
import src.model.RF_model as rf_model
import src.model.LR_model_with_interactions as lr_model


def download_and_preprocess_data():
    """
    This function is created so that it can be linked to Poetry Custom Script Execution. It encapsulates the logic for the initial workspace setup for this project.
    :return:
    NoneType
    """
    print("Project initialization...")
    helper.prepare_data_directory()

    lights_data = helper.download_lights_data()
    helper.save_df(lights_data, constants.LIGHTS_DATASET_NAME, is_preprocessed=False)

    patrols_data = helper.download_police_data()
    helper.save_df(patrols_data, constants.PATROL_DATASET_NAME, is_preprocessed=False)

    livability_index_data = helper.download_livability_data()
    helper.save_df(livability_index_data, constants.LIVABILITY_DATASET_NAME, is_preprocessed=False)

    green_index_data = helper.download_green_index_data()
    helper.save_df(green_index_data, constants.GREEN_INDEX_DATASET_NAME, is_preprocessed=False)

    recorded_crimes_data = helper.download_data_from_police(constants.CRIME_LINK_TO_FETCH, constants.CRIME_CODE,
                                                            constants.CRIME_COLUMNS_TO_TRANSFORM,
                                                            desc="Downloading Crime Data...")
    public_nuisance_data = helper.download_data_from_police(constants.POLICE_LINK_TO_FETCH,
                                                            constants.POLICE_CODE,
                                                            constants.POLICE_COLUMNS_TO_TRANSFORM,
                                                            desc="Downloading Nuisance Data...")

    helper.save_df(recorded_crimes_data, constants.RECORDED_CRIMES_DATASET_NAME, is_preprocessed=False)
    helper.save_df(public_nuisance_data, constants.PUBLIC_NUISANCE_DATASET_NAME, is_preprocessed=False)

    helper.download_file(constants.POI_CONFIG_FILE_LINK, constants.POI_CONFIG_FILE_NAME, desc="Downloading POI Data...")
    poiS = helper.download_poi_data_from_config()
    helper.save_df(poiS, constants.POI_DATASET_NAME, is_preprocessed=False)

    helper.download_file(constants.MOVE_HOUSES_DATASET_LINK, constants.MOVE_HOUSES_DATASET_NAME,
                         desc="Downloading Move Houses Data...")
    helper.download_file(constants.NEIGHBOURHOOD_DATASET_LINK, constants.NEIGHBOURHOOD_DATASET_NAME,
                         desc="Downloading Neighbourhood Data...")

    preprocess_downloaded_raw_data()
    train_model_on_new_data()


def preprocess_downloaded_raw_data():
    """
    This function is created so that it can be linked to Poetry Custom Script Execution. It encapsulates the logic for the preprocessing for this project.
    :return:
    NoneType
    """
    print('Reading RAW data...')
    lights_data = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                                                   constants.LIGHTS_DATASET_NAME))
    green_index_data = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                                                        constants.GREEN_INDEX_DATASET_NAME))
    poiS = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                                            constants.POI_DATASET_NAME))
    livability_index_data = pd.read_csv(
        helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                             constants.LIVABILITY_DATASET_NAME))
    recorded_crimes_data = pd.read_csv(
        helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                             constants.RECORDED_CRIMES_DATASET_NAME))
    public_nuisance_data = pd.read_csv(
        helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                             constants.PUBLIC_NUISANCE_DATASET_NAME))

    move_houses_data = gpd.read_file(helper.generate_path(helper.generate_path(constants.PATH, constants.RAW_DIRECTORY),
                                                          constants.MOVE_HOUSES_DATASET_NAME))
    aggregate_raw_data(lights_data, green_index_data, move_houses_data, poiS, livability_index_data,
                       recorded_crimes_data, public_nuisance_data)
    train_model_on_new_data()


def aggregate_raw_data(lights_data, green_index_data, move_houses_data, poiS, livability_index_data,
                       recorded_crimes_data,
                       public_nuisance_data):
    """
    This function encapsulates the logic for the preprocessing for each dataset passed as input parameters.
    :param lights_data: (pd.DataFrame) object.
    :param green_index_data: (pd.DataFrame) object.
    :param move_houses_data: (pd.DataFrame) object.
    :param poiS: (pd.DataFrame) object.
    :param livability_index_data: (pd.DataFrame) object.
    :param recorded_crimes_data: (pd.DataFrame) object.
    :param public_nuisance_data: (pd.DataFrame) object.
    :return:
    NoneType
    """
    print('Pre-processing the downloaded data...')

    helper.map_geometries_to_neighbourhoods([lights_data, green_index_data],
                                            [['X', 'Y'], ['longitude', 'latitude']],
                                            [move_houses_data, poiS])

    livability_index_data = ap.preprocess_livability(livability_index_data)
    recorded_crimes_data = ap.preprocess_crimes(recorded_crimes_data)

    public_nuisance_data = bp.preprocess_nuisance(public_nuisance_data)
    move_houses_data = bp.preprocess_houses(move_houses_data)
    green_index_data = bp.preprocess_green_index(green_index_data)

    helper.save_df(lights_data, constants.LIGHTS_DATASET_NAME, is_preprocessed=True)
    helper.save_df(poiS, constants.POI_DATASET_NAME, is_preprocessed=True)

    helper.save_df(livability_index_data, constants.LIVABILITY_DATASET_NAME, is_preprocessed=True)
    helper.save_df(recorded_crimes_data, constants.RECORDED_CRIMES_DATASET_NAME, is_preprocessed=True)

    helper.save_df(public_nuisance_data, constants.PUBLIC_NUISANCE_DATASET_NAME, is_preprocessed=True)
    helper.save_df(green_index_data, constants.GREEN_INDEX_DATASET_NAME, is_preprocessed=True)
    helper.save_df(move_houses_data, constants.MOVE_HOUSES_DATASET_PREPROCESSED_NAME, is_preprocessed=True)

    merged = bp.merge_datasets(green_index_data, move_houses_data, public_nuisance_data, recorded_crimes_data,
                               livability_index_data)
    helper.save_df(merged, constants.MERGED_DATA_NAME, is_preprocessed=True)

    merged = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.PREPROCESSED_DIRECTORY),
                                              constants.MERGED_DATA_NAME))
    helper.save_df(merged, constants.ORIGINAL_DATA_PER_YEAR_NAME, is_preprocessed=False, is_final=True)
    temp = merged.groupby('Neighbourhood').mean().reset_index().drop('year', axis=1)
    helper.save_df(temp, constants.ORIGINAL_DATA_PER_NEIGHBOURHOOD_NAME, is_preprocessed=False, is_final=True)

    full_df = bp.calculate_custom_livability_index(merged)
    full_df['Livability index'] = merged['Livability index'].astype(int).values
    helper.save_df(full_df, constants.FULL_DATA_PER_YEAR_NAME, is_preprocessed=False, is_final=True)
    temp = full_df.groupby('Neighbourhood').mean().reset_index().drop('year', axis=1)
    helper.save_df(temp, constants.FULL_DATA_PER_NEIGHBOURHOOD_NAME, is_preprocessed=False, is_final=True)

    final_df = bp.calculate_custom_livability_index(merged)
    helper.save_df(final_df, constants.CUSTOM_DATA_PER_YEAR_NAME, is_preprocessed=False, is_final=True)
    temp = final_df.groupby('Neighbourhood').mean().reset_index().drop('year', axis=1)
    helper.save_df(temp, constants.CUSTOM_DATA_PER_NEIGHBOURHOOD_NAME, is_preprocessed=False, is_final=True)


def train_model_on_new_data():
    """
    This function is created so that it can be linked to Poetry Custom Script Execution. It encapsulates the logic for the Model training process for this project.
    :return:
    NoneType
    """
    lr_df = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.FINAL_DIRECTORY),
                                             constants.ORIGINAL_DATA_PER_NEIGHBOURHOOD_NAME))
    lr_model.train_lr_interaction(lr_df)

    rf_df = pd.read_csv(helper.generate_path(helper.generate_path(constants.PATH, constants.FINAL_DIRECTORY),
                                             constants.ORIGINAL_DATA_PER_YEAR_NAME))
    rf_model.train_random_forest_model(rf_df)
