import bs4
import geopandas
import io

import pytest
import requests
from shapely import Polygon, Point
import xml.etree.ElementTree as ET

from src import helper
import src.constants
import pandas as pd


class TestGeneratePath(object):

    def test_generate_path_should_return_correct_value(self):
        expected = 'demo/file.csv'
        actual = helper.generate_path('demo', 'file.csv')
        assert actual == expected, "Generate Path doesn't produce the expected result."

    def test_generate_path_should_raise_exception_for_none_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            helper.generate_path(obj, 'demo')

    def test_generate_path_should_raise_exception_for_none_as_input_2(self):
        obj = None
        with pytest.raises(ValueError):
            helper.generate_path('demo', obj)

    def test_generate_path_should_raise_exception_for_empty_as_input_1(self):
        obj = ''
        with pytest.raises(ValueError):
            helper.generate_path(obj, 'demo')

    def test_generate_path_should_raise_exception_for_empty_as_input_2(self):
        obj = ''
        with pytest.raises(ValueError):
            helper.generate_path('demo', obj)


class TestPrepareDataDirectory(object):

    def test_generate_path_should_return_correct_value_for_makedirs(self, mocker):
        mocker.patch.object(src.constants, 'PATH', 'demo')
        mocker.patch.object(src.constants, 'DATA_FOLDERS', ['a', 'b', 'c', 'd'])
        mock_f = mocker.patch('os.makedirs', return_value=True)
        mocker.patch('os.path.exists', return_value=False)
        helper.prepare_data_directory()
        expected = len(src.constants.DATA_FOLDERS) + 1
        assert expected == mock_f.call_count, "Generate Path doesn't produce the expected result."

    def test_generate_path_should_return_correct_value_for_path_exists(self, mocker):
        mocker.patch.object(src.constants, 'PATH', 'demo')
        mocker.patch.object(src.constants, 'DATA_FOLDERS', ['a', 'b', 'c'])
        mocker.patch('os.makedirs', return_value=True)
        mock_f = mocker.patch('os.path.exists', return_value=False)
        helper.prepare_data_directory()
        expected = len(src.constants.DATA_FOLDERS) + 1
        assert expected == mock_f.call_count, "Generate Path doesn't produce the expected result."


class TestSaveDf(object):

    def test_save_df_should_return_correct_value_for_to_csv_call_count(self, mocker):
        df = pd.DataFrame()
        mock_f = mocker.patch('pandas.DataFrame.to_csv', return_value=True)
        helper.save_df(df, 'demo.csv', is_preprocessed=False)
        expected = 1
        assert expected == mock_f.call_count, "Save Df doesn't produce the expected result."

    def test_save_df_should_return_correct_value_for_raw_path_dir(self, mocker):
        df = pd.DataFrame()
        mocker.patch('pandas.DataFrame.to_csv', return_value=True)
        expected = helper.generate_path(helper.constants.PATH, helper.constants.RAW_DIRECTORY)
        actual = helper.save_df(df, 'demo.csv', is_preprocessed=False)
        assert expected == actual, "Save Df doesn't produce the expected result."

    def test_save_df_should_return_correct_value_for_preprocessed_path_dir(self, mocker):
        df = pd.DataFrame()
        mocker.patch('pandas.DataFrame.to_csv', return_value=True)
        expected = helper.generate_path(helper.constants.PATH, helper.constants.PREPROCESSED_DIRECTORY)
        actual = helper.save_df(df, 'demo.csv', is_preprocessed=True)
        assert expected == actual, "Save Df doesn't produce the expected result."

    def test_save_df_should_raise_exception_for_none_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            helper.save_df(obj, 'demo', is_preprocessed=True)

    def test_save_df_should_raise_exception_for_none_as_input_2(self):
        df = pd.DataFrame()
        obj = None
        with pytest.raises(ValueError):
            helper.save_df(df, obj, is_preprocessed=True)


class TestGetFullValue(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.arr = [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}]

    def test_get_full_value_should_return_correct_value(self, setup_and_teardown):
        expected = 'memo'
        actual = helper.get_full_value(self.arr, 'demo')
        assert expected == actual, "Get Full Value doesn't produce the expected result."

    def test_get_full_value_should_return_empty_value_for_invalid_input_param(self, setup_and_teardown):
        expected = ''
        actual = helper.get_full_value(self.arr, 'invalid')
        assert expected == actual, "Get Full Value doesn't produce the expected result."

    def test_get_full_value_should_raise_exception_for_none_as_input_1(self):
        obj = None
        with pytest.raises(ValueError):
            helper.get_full_value(obj, 'invalid')

    def test_get_full_value_should_raise_exception_for_none_as_input_2(self):
        arr = []
        obj = None
        with pytest.raises(ValueError):
            helper.get_full_value(arr, obj)


class TestDownloadMetadata(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.expected = {'Demo': [{'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                   'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                   'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]},
                                  {'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                   'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                   'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]}],
                         'Demo2': [{'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                    'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                    'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]},
                                   {'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                    'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                    'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]}],
                         'Demo3': [{'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                    'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                    'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]},
                                   {'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                                    'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                                    'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]}]}

        self.input = {'Demo': [{'metadata': '...'}, {'value': [{'Key': 'demo', 'Title': 'memo'}]}],
                      'Demo2': [{'metadata': '...'}, {'value': [{'Key': 'demo2', 'Title': 'memo2'}]}],
                      'Demo3': [{'metadata': '...'}, {'value': [{'Key': 'demo3', 'Title': 'memo3'}]}]}

    def test_download_metadata_should_return_correct_value(self, mocker, setup_and_teardown):
        mocker.patch('requests.get', return_value=requests.Response())
        mock_f = mocker.patch('requests.Response.json', return_value=self.input)
        expected = self.expected
        actual = helper.download_metadata('ignored', list(self.input.keys()))
        assert expected == actual, "Download Metadata doesn't produce the expected result."

    def test_download_metadata_should_return_correct_method_calls_count(self, mocker, setup_and_teardown):
        mocker.patch('requests.get', return_value=requests.Response())
        mock_f = mocker.patch('requests.Response.json', return_value=self.input)
        expected = len(self.input.keys())
        actual = helper.download_metadata('ignored', list(self.input.keys()))
        assert expected == mock_f.call_count, "Download Metadata doesn't produce the expected result."

    def test_download_metadata_should_raise_exception_for_none_as_input_1(self):
        arr = []
        obj = None
        with pytest.raises(ValueError):
            helper.download_metadata(obj, arr)

    def test_download_metadata_should_raise_exception_for_none_as_input_2(self):
        str = 'valid'
        obj = None
        with pytest.raises(ValueError):
            helper.download_metadata(str, obj)

    def test_download_metadata_should_raise_exception_for_empty_columns_list(self):
        str = 'valid'
        obj = []
        with pytest.raises(ValueError):
            helper.download_metadata(str, obj)


class TestDownloadDataFromPolice(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.input = ET.fromstring('''<?xml version="1.0" encoding="utf-8"?>
<feed xml:base="http://dataderden.cbs.nl/ODataFeed/OData/47022NED" xmlns="http://www.w3.org/2005/Atom" xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices" xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" xmlns:georss="http://www.georss.org/georss" xmlns:gml="http://www.opengis.net/gml">
  <id>https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet</id>
  <title type="text">TypedDataSet</title>
  <updated>2023-05-15T02:00:00+02:00</updated>
  <link rel="self" title="TypedDataSet" href="https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet" />
  <entry>
    <id>https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet(437512)</id>
    <category term="Cbs.OData.TData" scheme="http://schemas.microsoft.com/ado/2007/08/dataservices/scheme" />
    <link rel="self" href="https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet(437512)" />
    <title />
    <updated>2023-05-15T02:00:00+02:00</updated>
    <author>
      <name />
    </author>
    <content type="application/xml">
      <m:properties>
        <d:ID m:type="Edm.Int32">437512</d:ID>
        <d:SoortMisdrijf xml:space="preserve">0.0.0 </d:SoortMisdrijf>
        <d:WijkenEnBuurten xml:space="preserve">GM0758    </d:WijkenEnBuurten>
        <d:Perioden>2012MM01</d:Perioden>
        <d:GeregistreerdeMisdrijven_1 m:type="Edm.Int32">1382</d:GeregistreerdeMisdrijven_1>
      </m:properties>
    </content>
  </entry>
  <entry>
    <id>https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet(2887079)</id>
    <category term="Cbs.OData.TData" scheme="http://schemas.microsoft.com/ado/2007/08/dataservices/scheme" />
    <link rel="self" href="https://dataderden.cbs.nl/ODataFeed/OData/47022NED/TypedDataSet(2887079)" />
    <title />
    <updated>2023-05-15T02:00:00+02:00</updated>
    <author>
      <name />
    </author>
    <content type="application/xml">
      <m:properties>
        <d:ID m:type="Edm.Int32"></d:ID>
        <d:SoortMisdrijf xml:space="preserve">1.1.1 </d:SoortMisdrijf>
        <d:WijkenEnBuurten>BU07580004</d:WijkenEnBuurten>
        <d:Perioden>2017MM12</d:Perioden>
        <d:GeregistreerdeMisdrijven_1 m:type="Edm.Int32">1</d:GeregistreerdeMisdrijven_1>
      </m:properties>
    </content>
  </entry>
</feed>''')
        self.none = None
        self.empty = ''
        self.arr = [5]
        self.valid_str = 'valid'

    def test_download_data_from_police_should_return_correct_request_call_count(self, mocker, setup_and_teardown):
        mock_f = mocker.patch('requests.get', return_value=requests.Response())
        mocker.patch('requests.Response.text', return_value=str(self.input))
        mocker.patch('src.helper.download_metadata', return_value={'WijkenEnBuurten': ['demo']})
        mocker.patch('src.helper.get_full_value', return_value='DEMO')
        mocker.patch('xml.etree.ElementTree.fromstring', return_value=self.input)
        expected = 1
        actual = helper.download_data_from_police('ignored', 'ignored', ['WijkenEnBuurten'])
        assert expected == mock_f.call_count, "Download Data From Police doesn't produce the expected result."

    def test_download_data_from_police_should_return_correct_get_full_value_count(self, mocker, setup_and_teardown):
        mocker.patch('requests.get', return_value=requests.Response())
        mocker.patch('requests.Response.text', return_value=str(self.input))
        mocker.patch('src.helper.download_metadata', return_value={'WijkenEnBuurten': ['demo']})
        mock_f = mocker.patch('src.helper.get_full_value', return_value='DEMO')
        mocker.patch('xml.etree.ElementTree.fromstring', return_value=self.input)
        expected = 2
        actual = helper.download_data_from_police('ignored', 'ignored', ['WijkenEnBuurten'])
        assert expected == mock_f.call_count, "Download Data From Police doesn't produce the expected result."

    def test_download_data_from_police_should_return_correct_value(self, mocker, setup_and_teardown):
        mocker.patch('requests.get', return_value=requests.Response())
        mocker.patch('requests.Response.text', return_value=str(self.input))
        mocker.patch('src.helper.download_metadata', return_value={'WijkenEnBuurten': ['demo']})
        mock_f = mocker.patch('src.helper.get_full_value', return_value='DEMO')
        mocker.patch('xml.etree.ElementTree.fromstring', return_value=self.input)
        expected = 'DEMO'
        actual = helper.download_data_from_police('ignored', 'ignored', ['WijkenEnBuurten']).iloc[0]['WijkenEnBuurten']
        assert expected == actual, "Download Data From Police doesn't produce the expected result."

    def test_download_data_from_police_should_return_correct_value_zero_after_correction(self, mocker,
                                                                                         setup_and_teardown):
        mocker.patch('requests.get', return_value=requests.Response())
        mocker.patch('requests.Response.text', return_value=str(self.input))
        mocker.patch('src.helper.download_metadata', return_value={'WijkenEnBuurten': ['demo']})
        mock_f = mocker.patch('src.helper.get_full_value', return_value='DEMO')
        mocker.patch('xml.etree.ElementTree.fromstring', return_value=self.input)
        expected = '0'
        actual = helper.download_data_from_police('ignored', 'ignored', ['WijkenEnBuurten']).iloc[1]['ID']
        assert expected == actual, "Download Data From Police doesn't produce the expected result."

    def test_download_data_from_police_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.none, self.valid_str, self.arr)

    def test_download_data_from_police_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.valid_str, self.none, self.arr)

    def test_download_data_from_police_should_raise_exception_for_none_as_input_3(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.valid_str, self.valid_str, self.none)

    def test_download_data_from_police_should_raise_exception_for_none_as_input_1_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.empty, self.valid_str, self.arr)

    def test_download_data_from_police_should_raise_exception_for_none_as_input_2_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.valid_str, self.empty, self.arr)

    def test_download_data_from_police_should_raise_exception_for_none_as_input_3_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_data_from_police(self.valid_str, self.valid_str, columns_to_transform=[])


class TestDownloadFile(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.expected = 1
        self.none = None
        self.empty = ''
        self.valid_str = 'valid'

    def test_download_file_should_return_correct_value_for_file_open_write(self, mocker, setup_and_teardown):
        response = requests.Response()
        mocker.patch('requests.get', return_value=response)
        mocker.patch('requests.Response.content', return_value='demo'.encode())
        mocker.patch.object(response, 'status_code', 200)
        mock_f = mocker.patch('builtins.open', return_value=io.BytesIO('demo'.encode()))
        expected = self.expected
        helper.download_file(self.valid_str, 'demo.txt')
        actual = mock_f.call_count
        assert expected == actual, "Download File doesn't produce the expected result."

    def test_download_file_should_return_correct_value_for_request_get(self, mocker, setup_and_teardown):
        response = requests.Response()
        mock_f = mocker.patch('requests.get', return_value=response)
        mocker.patch('requests.Response.content', return_value='demo'.encode())
        mocker.patch.object(response, 'status_code', 200)
        mocker.patch('builtins.open', return_value=io.BytesIO('demo'.encode()))
        expected = self.expected
        helper.download_file(self.valid_str, 'demo.txt')
        actual = mock_f.call_count
        assert expected == actual, "Download File doesn't produce the expected result."

    def test_download_file_should_raise_exception_for_invalid_status_code(self, mocker, setup_and_teardown):
        response = requests.Response()
        mocker.patch('requests.get', return_value=response)
        mocker.patch.object(response, 'status_code', 403)
        with pytest.raises(ConnectionAbortedError):
            helper.download_file(self.valid_str, self.valid_str)

    def test_download_file_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_file(self.none, self.valid_str)

    def test_download_file_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_file(self.valid_str, self.none)

    def test_download_file_should_raise_exception_for_none_as_input_1_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_file(self.empty, self.valid_str)

    def test_download_file_should_raise_exception_for_none_as_input_2_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.download_file(self.valid_str, self.empty)


class TestGetArticles(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.expected = bs4.ResultSet(bs4.SoupStrainer(), ['demo', 'demo2'])
        self.input = '''<html>
<head>
</head>
<body>
<article>demo</article>
<article>demo2</article>
</body>
</html>'''
        self.none = None
        self.empty = ''
        self.valid_str = 'valid'

    def test_get_articles_should_return_correct_value_for_the_given_html(self, mocker, setup_and_teardown):
        response = requests.Response()
        mock_f = mocker.patch('requests.get', return_value=response)
        mocker.patch('requests.Response.content', return_value=self.input)
        mocker.patch.object(response, 'status_code', 200)
        expected = 1
        helper.get_articles('demo')
        actual = mock_f.call_count
        assert expected == actual, "Get Articles doesn't produce the expected result."

    def test_get_articles_should_return_correct_value_for_the_given_content(self, mocker, setup_and_teardown):
        response = requests.Response()
        mocker.patch('requests.get', return_value=response)
        mocker.patch('requests.Response.content', return_value=self.input)
        mocker.patch('requests.Response.content', return_value=self.input)
        mocker.patch('bs4.BeautifulSoup.find_all', return_value=self.expected)
        mocker.patch.object(response, 'status_code', 200)
        expected = self.expected
        actual = helper.get_articles('demo')
        assert expected == actual, "Get Articles doesn't produce the expected result."

    def test_download_file_should_raise_exception_for_invalid_status_code(self, mocker, setup_and_teardown):
        response = requests.Response()
        mocker.patch('requests.get', return_value=response)
        mocker.patch.object(response, 'status_code', 403)
        with pytest.raises(ConnectionAbortedError):
            helper.get_articles(self.valid_str)

    def test_download_file_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_articles(self.none)

    def test_download_file_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_articles(self.empty)


class TestGetNeighbourhoodOnCoordinates(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.neighbourhoods_df = pd.DataFrame(
            {'geometry': Polygon([(5, 5), (5, 13), (13, 13), (13, 5)]),
             'Neighbourhood Name': ['Bavel']})
        self.none = None
        self.empty_arr = []
        self.valid_str = 'valid'
        self.valid_arr = ['valid']

    def test_get_neighbourhood_on_coordinates_should_return_correct_value_for_the_given_points(self, mocker,
                                                                                               setup_and_teardown):
        coordinate_dfs = [pd.DataFrame({'x': [8, 11], 'y': [8, 11]})]
        coordinate_columns_dfs = [['x', 'y']]

        points = []
        for idx in range(len(coordinate_dfs)):
            temp = [Point(el[0], el[1]) for el in
                    coordinate_dfs[idx][coordinate_columns_dfs[idx]].drop_duplicates().values]
            points.append(temp)
            coordinate_dfs[idx]['Neighbourhood'] = ''

        expected = self.neighbourhoods_df.iloc[0][1]
        helper.get_neighbourhood_on_coordinates(self.neighbourhoods_df.iloc[0], points[0],
                                                coordinate_dfs[0], coordinate_columns_dfs[0])
        actual = coordinate_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Get Neighbourhood On Coordinates doesn't produce the expected result."

    def test_get_neighbourhood_on_coordinates_should_return_correct_empty_value_for_the_given_points(self, mocker,
                                                                                                     setup_and_teardown):
        coordinate_dfs = [pd.DataFrame({'x': [15], 'y': [15]})]
        coordinate_columns_dfs = [['x', 'y']]

        points = []
        for idx in range(len(coordinate_dfs)):
            temp = [Point(el[0], el[1]) for el in
                    coordinate_dfs[idx][coordinate_columns_dfs[idx]].drop_duplicates().values]
            points.append(temp)
            coordinate_dfs[idx]['Neighbourhood'] = ''

        expected = ''
        helper.get_neighbourhood_on_coordinates(self.neighbourhoods_df.iloc[0], points[0],
                                                coordinate_dfs[0], coordinate_columns_dfs[0])
        actual = coordinate_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Get Neighbourhood On Coordinates doesn't produce the expected result."

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.none, self.valid_arr, pd.DataFrame(), self.valid_arr)

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.valid_arr, self.none, pd.DataFrame(), self.valid_arr)

    def testget_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_3(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.valid_arr, self.valid_arr, self.none, self.valid_arr)

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_4(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.valid_arr, self.valid_arr, pd.DataFrame(), self.none)

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_1_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.empty_arr, self.valid_arr, pd.DataFrame(), self.valid_arr)

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_2_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.valid_arr, self.empty_arr, pd.DataFrame(), self.valid_arr)

    def test_get_neighbourhood_on_coordinates_should_raise_exception_for_none_as_input_4_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_coordinates(self.valid_arr, self.valid_arr, pd.DataFrame(), self.empty_arr)


class TestGetNeighbourhoodOnGeometry(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.neighbourhoods_df = pd.DataFrame(
            {'geometry': Polygon([(5, 5), (5, 13), (13, 13), (13, 5)]),
             'Neighbourhood Name': ['Bavel']})
        self.none = None
        self.empty_arr = []
        self.valid_str = 'valid'
        self.valid_arr = ['valid']

    def test_get_neighbourhood_on_geometry_should_return_correct_value_for_the_given_geometry(self, mocker,
                                                                                              setup_and_teardown):
        geometry_dfs = [pd.DataFrame({'geometry': [Polygon([(6, 6), (6, 12), (12, 12), (12, 6)])]})]

        for idx in range(len(geometry_dfs)):
            geometry_dfs[idx]['geometry_temp'] = geometry_dfs[idx]['geometry'].map(lambda e: str(e))
            geometry_dfs[idx]['Neighbourhood'] = ''
        expected = self.neighbourhoods_df.iloc[0][1]
        helper.get_neighbourhood_on_geometry(self.neighbourhoods_df.iloc[0], geometry_dfs[0])
        actual = geometry_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Get Neighbourhood On Geometry doesn't produce the expected result."

    def test_get_neighbourhood_on_geometry_should_return_correct_value_for_the_given_str_geometry(self, mocker,
                                                                                                  setup_and_teardown):
        geometry_dfs = [pd.DataFrame({'geometry': [str(Polygon([(6, 6), (6, 12), (12, 12), (12, 6)]))]})]

        for idx in range(len(geometry_dfs)):
            geometry_dfs[idx]['geometry_temp'] = geometry_dfs[idx]['geometry'].map(lambda e: str(e))
            geometry_dfs[idx]['Neighbourhood'] = ''
        expected = self.neighbourhoods_df.iloc[0][1]
        helper.get_neighbourhood_on_geometry(self.neighbourhoods_df.iloc[0], geometry_dfs[0])
        actual = geometry_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Get Neighbourhood On Geometry doesn't produce the expected result."

    def test_get_neighbourhood_on_geometry_should_return_correct_empty_value_for_the_given_geometry(self, mocker,
                                                                                                    setup_and_teardown):
        geometry_dfs = [pd.DataFrame({'geometry': [Polygon([(55, 55), (55, 13), (13, 13), (13, 55)])]})]

        for idx in range(len(geometry_dfs)):
            geometry_dfs[idx]['geometry_temp'] = geometry_dfs[idx]['geometry'].map(lambda e: str(e))
            geometry_dfs[idx]['Neighbourhood'] = ''
        expected = ''
        helper.get_neighbourhood_on_geometry(self.neighbourhoods_df.iloc[0], geometry_dfs[0])
        actual = geometry_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Get Neighbourhood On Geometry doesn't produce the expected result."

    def test_get_neighbourhood_on_geometry_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_geometry(self.none, pd.DataFrame())

    def test_get_neighbourhood_on_geometry_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_geometry(self.valid_arr, self.none)

    def test_get_neighbourhood_on_geometry_should_raise_exception_for_none_as_input_1_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.get_neighbourhood_on_geometry(self.empty_arr, pd.DataFrame())


class TestMapGeometriesToNeighbourhoods(object):

    @pytest.fixture
    def setup_and_teardown(self):
        self.neighbourhoods_df = geopandas.GeoDataFrame(
            {'geometry': Polygon([(5, 5), (5, 13), (13, 13), (13, 5)]),
             'buurtnaam': ['Bavel'],
             'gemeentenaam': ['Breda']}, crs=4326)
        self.none = None
        self.empty_arr = []
        self.valid_str = 'valid'
        self.valid_arr = ['valid']

    def test_map_geometries_to_neighbourhoods_should_return_correct_value_for_the_given_points(self, mocker,
                                                                                               setup_and_teardown):
        geometry_dfs = [pd.DataFrame({'geometry': [Polygon([(6, 6), (6, 12), (12, 12), (12, 6)])]})]
        coordinate_dfs = [pd.DataFrame({'x': [8, 11], 'y': [8, 11]})]

        mock_f = mocker.patch('geopandas.read_file', return_value=self.neighbourhoods_df)

        expected = self.neighbourhoods_df.iloc[0][1]
        helper.map_geometries_to_neighbourhoods(coordinate_dfs, [['x', 'y']], geometry_dfs)
        actual = geometry_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Map Geometries To Neighborhood doesn't produce the expected result."

    def test_map_geometries_to_neighbourhoods_should_return_correct_empty_value_for_the_given_points(self, mocker,
                                                                                                     setup_and_teardown):
        geometry_dfs = [pd.DataFrame({'geometry': [Polygon([(55, 55), (55, 13), (13, 13), (13, 55)])]})]
        coordinate_dfs = [pd.DataFrame({'x': [8, 11], 'y': [8, 11]})]

        mock_f = mocker.patch('geopandas.read_file', return_value=self.neighbourhoods_df)

        expected = ''
        helper.map_geometries_to_neighbourhoods(coordinate_dfs, [['x', 'y']], geometry_dfs)
        actual = geometry_dfs[0]['Neighbourhood'][0]
        assert expected == actual, "Map Geometries To Neighborhood doesn't produce the expected result."

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.none, self.valid_arr, self.valid_arr)

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_2(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.valid_arr, self.none, self.valid_arr)

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_3(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.valid_arr, self.valid_arr, self.none)

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_1_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.empty_arr, self.valid_arr, self.valid_arr)

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_2_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.valid_arr, self.empty_arr, self.valid_arr)

    def test_map_geometries_to_neighbourhoods_should_raise_exception_for_none_as_input_3_1(self, setup_and_teardown):
        with pytest.raises(ValueError):
            helper.map_geometries_to_neighbourhoods(self.valid_arr, self.valid_arr, self.empty_arr)
