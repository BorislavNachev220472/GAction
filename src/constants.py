import os
from dotenv import load_dotenv

dirname = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.abspath(os.path.join(dirname, '../'))
env_file = os.path.abspath(os.path.join(ROOT, '.env'))

load_dotenv(dotenv_path=env_file)

DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

GREEN_INDEX_TABLE_NAME = 'green_index'
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

PATH = os.path.abspath(os.path.join(ROOT, 'data'))

RAW_DIRECTORY = 'raw'
PREPROCESSED_DIRECTORY = 'preprocessed'
FINAL_DIRECTORY = 'final'
MODEL_DIRECTORY = 'model'
DATA_FOLDERS = [RAW_DIRECTORY, PREPROCESSED_DIRECTORY, FINAL_DIRECTORY, MODEL_DIRECTORY]
SEPARATOR = ','
INCLUDE_INDEX = False
LIGHTS_DATASET_NAME = 'lights_dataset.csv'
PATROL_DATASET_NAME = 'police_dataset.csv'
LIVABILITY_DATASET_NAME = 'livability_dataset.csv'
GREEN_INDEX_DATASET_NAME = 'green_index_dataset.csv'
PUBLIC_NUISANCE_DATASET_NAME = 'public_nuisance.csv'
RECORDED_CRIMES_DATASET_NAME = 'recorded_crimes.csv'
NEIGHBOURHOOD_DATASET_NAME = 'wijkenbuurten_2022_v1.gpkg'

LIGHTS_DATASET_LINK = "https://opendata.arcgis.com/datasets/9f936466931d4bd389188b681159fce8_0.csv"
POLICE_DATASET_LINK = 'https://www.politie.nl/mijn-buurt/wijkagenten/lijst?{0}geoquery=Breda%2C+Breda%2C+Noord-Brabant&distance=10.0'
LIVABILITY_DATASET_LINK = 'https://www.leefbaarometer.nl/tabel.php?indicator=Leefbaarheidssituatie&schaal=Buurt&gemeente=GM0758'
NEIGHBOURHOOD_DATASET_LINK = 'https://service.pdok.nl/cbs/wijkenbuurten/2022/atom/downloads/wijkenbuurten_2022_v1.gpkg'

CRIME_CODE = '47022NED'
CRIME_COLUMNS_TO_TRANSFORM = ['SoortMisdrijf', 'WijkenEnBuurten', 'Perioden']
CRIME_LINK_TO_FETCH = f'https://dataderden.cbs.nl/ODataFeed/odata/{CRIME_CODE}/TypedDataSet?%24filter=((substring(WijkenEnBuurten%2C2%2C4)%20eq%20%270758%27))'

POLICE_CODE = '47024NED'
POLICE_COLUMNS_TO_TRANSFORM = ['Overlast', 'WijkenEnBuurten', 'Perioden']
POLICE_LINK_TO_FETCH = f'https://dataderden.cbs.nl/ODataFeed/odata/{POLICE_CODE}/UntypedDataSet?%24filter=((substring(WijkenEnBuurten%2C2%2C4)%20eq%20%270758%27))'

MOVE_HOUSES_DATASET_LINK = 'https://edubuas-my.sharepoint.com/:u:/g/personal/blerck_i_buas_nl/EQ5VQqyNQNNFmz92RyrXuDABObPXZ2qmIa--codTJq03jQ?e=nwbZFf&download=1'
MOVE_HOUSES_DATASET_NAME = 'move_houses.gpkg'
MOVE_HOUSES_DATASET_PREPROCESSED_NAME = 'move_houses_preprocessed.csv'

POI_CONFIG_FILE_LINK = 'https://edubuas-my.sharepoint.com/:x:/g/personal/220472_buas_nl/EcbqrdP-TEpGv5KJojUaHVYBlPHrZDAwv0ie8Ok5rM1WmQ?download=1'
POI_CONFIG_FILE_NAME = 'poi_configuration.csv'
POI_CONFIG_COLUMN_NAME = 'POI'
POI_CONFIG_HEADER_LENGTH = 2
POI_LOCATION = 'Breda, The Netherlands'
POI_DATASET_NAME = 'poi_dataset.csv'

MERGED_DATA_NAME = 'merged.csv'

FULL_DATA_PER_YEAR_NAME = 'full_data_per_year.csv'
FULL_DATA_PER_NEIGHBOURHOOD_NAME = 'full_data_per_neighbourhood.csv'
ORIGINAL_DATA_PER_YEAR_NAME = 'original_liv_data_per_year.csv'
ORIGINAL_DATA_PER_NEIGHBOURHOOD_NAME = 'original_liv_data_per_neighbourhood.csv'
CUSTOM_DATA_PER_NEIGHBOURHOOD_NAME = 'custom_liv_data_per_neighbourhood.csv'
CUSTOM_DATA_PER_YEAR_NAME = 'custom_liv_data_per_year.csv'

LR_INTERACTION_MODEL_NAME = "LR_model_with_interactions.pickle"
RF_FOREST_MODEL_SUFFIX = "_RandomForest_model.joblib"
