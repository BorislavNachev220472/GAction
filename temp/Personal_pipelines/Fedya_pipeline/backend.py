import cv2
import pandas as pd
import numpy as np
import glob
import os
import skimage
from patchify import patchify, unpatchify
import networkx as nx
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import segmentation_models
from keras.models import model_from_json

#region ROI extraction

def find_start_end(array):
    '''
    Takes is an image as array and return start and end coordinates of the petri dish
    :param array: image as a numpy array
    :return: start and end coordinates of the petri dish
    '''
    st = np.argmin(array)
    end = np.argmax(array)

    return [st, end]


# %%
def derivative(arr, axis):
    '''
    Takes in an image as an array and return average derivative value on x or y-axis
    :param arr: image as an array
    :param axis: boolean showing whether to inspect x or y-axis
    :return: average derivative value on x or y-axis
    '''
    if axis:
        derivatives = np.gradient(arr, axis=1)
        average_derivatives_vertical = np.mean(derivatives, axis=0)
        return average_derivatives_vertical

    else:
        derivatives = np.gradient(arr, axis=0)
        average_derivatives_vertical = np.mean(derivatives, axis=1)
        return average_derivatives_vertical


# %%
def roi_extraction_image(path):
    '''
    Function takes in a path to an image and returns coordinates of the petri dish in it by finding average derivatives values, start and end coordinates of the petri dish by looking at the change of the derivative. Then crops the petri dish
    :param path: path to the image
    :return: cropped petri dish
    '''
    img = cv2.imread(path, 0)
    img = img[0:-1, 0:4000]

    average_derivatives_horizontal = derivative(img, 1)
    average_derivatives_vertical = derivative(img, 0)

    y = find_start_end(average_derivatives_vertical)
    x = find_start_end(average_derivatives_horizontal)

    width = x[0] - x[1]

    return img[y[1]:y[1]+width, x[1]:x[0]]

def roi_extraction_coords(path):
    '''
    Function takes in a path to an image and returns coordinates of the petri dish in it by finding average derivatives values, start and end coordinates of the petri dish by looking at the change of the derivative. These coordinates are used to crop masks
    :param path: path to an image
    :return: coordinates of the petri dish to crop masks
    '''
    img = cv2.imread(path, 0)
    img = img[0:-1, 0:4000]

    average_derivatives_horizontal = derivative(img, 1)
    average_derivatives_vertical = derivative(img, 0)

    y = find_start_end(average_derivatives_vertical)
    x = find_start_end(average_derivatives_horizontal)

    width = x[0] - x[1]

    return [y[1], y[1] + width, x[1], x[0]]

def width_of_the_plate_p(path):
    '''
    Function takes in a path to an image and returns coordinates of the petri dish in it by finding average derivatives values, start and end coordinates of the petri dish by looking at the change of the derivative. These coordinates are used to find the width / height of the petri dish in pixels and return it.
    :param path: path to an image
    :return: width / height of the petri dish
    '''
    img = cv2.imread(path, 0)
    img = img[0:-1, 0:4000]

    average_derivatives_horizontal = derivative(img, 1)
    average_derivatives_vertical = derivative(img, 0)

    y = find_start_end(average_derivatives_vertical)
    x = find_start_end(average_derivatives_horizontal)

    width = x[0] - x[1]

    return width

#endregion

#region Prepare data for prediction

def load_image(path, single=True):
    images = []

    if single:
        bounds = roi_extraction_coords(path)
        images.append(cv2.imread(path)[bounds[0]:bounds[1], bounds[2]:bounds[3]])

    # locally - '/Users/work_uni/Documents/GitHub/2023-24b-fai2-adsai-FedorChursin220904/Dataset/'

    for image in glob.glob('/home/y2b/2023-24b-fai2-adsai-FedorChursin220904/Dataset/' + path + '/*.tif'):
        bounds = roi_extraction_coords(image)
        images.append(cv2.imread(image)[bounds[0]:bounds[1], bounds[2]:bounds[3]])

    return images


def padder(image, patch_size):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[25, 48, 121])

    return padded_image


def single_image(img, scaling_factor, patch_size):
    dims = []
    X = []

    if scaling_factor != 1:
        img = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor)

    img = padder(img, patch_size)
    patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
    dims.append([img.shape, patches.shape])

    patches = patches.reshape(-1, patch_size, patch_size, 3)

    for patch in patches:
        X.append(patch)

    return patches, dims


def multiple_images(images, scaling_factor, patch_size):
    dims = []
    X = []

    for img in images:

        if scaling_factor != 1:
            img = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor)

        img = padder(img, patch_size)

        patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        dims.append([img.shape, patches.shape])

        patches = patches.reshape(-1, patch_size, patch_size, 3)

        for patch in patches:
            X.append(patch)

    return X, dims


def datagen_for_predictions(path, patch_size, scaling_factor, single=True):
    X = []
    dims = []

    if path in 'traintest':
        single = False

    images = load_image(path, single)

    if single:
        X, dims = single_image(images[0], scaling_factor, patch_size)
        X = np.array(X, dtype=float) / 255

        return X, dims

    else:
        X, dims = multiple_images(images, scaling_factor, patch_size)
        X = np.array(X, dtype=float) / 255

        return X, dims

#endregion

#region Get model predictions

def get_predictions(model, data):
    predictions = model.predict(data, verbose = 0)
    predictions = np.argmax(predictions, axis=3)

    predictions_root_only = (predictions == 3) | (predictions == 1)
    predictions_root_only = predictions_root_only > 0
    predictions_root_only = (predictions_root_only * 1).astype(np.uint8)

    return predictions_root_only

#endregion

#region Instance segmentation

def segment_plants(mask):
    mask = cv2.convertScaleAbs(mask)

    kernel = np.ones((5, 5), dtype='uint8')
    mask_dialated = cv2.dilate(mask, kernel, iterations=1)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dialated)

    ind = np.argsort(stats[:, -1])[::-1]
    stats_sorted = stats[ind]

    minimal_area = 0
    if len(stats_sorted) < 6:
        minimal_area = stats_sorted[-1][4]
    else:
        minimal_area = stats_sorted[5][4]

    labels = skimage.morphology.remove_small_objects(labels, minimal_area)
    labels = (labels * 255).astype(np.uint8)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(labels)

    return (labels, stats)

#endregion

#region Set up files

def create_output_dir():

    path = os.getcwd()

    if not os.path.exists(path + "/outputs"):
        os.makedirs(path + "/outputs")

    return  f"Outputs directory created at {path}"
def save_csv_data(data):
    create_output_dir()

    path = os.getcwd() + "/outputs/"
    data.to_csv(path + 'primary_root_lengths.csv')

    return f"CSV file with primary root  lengths created at {path}"

def save_masks(image, image_name):
    create_output_dir()

    path = os.getcwd() + "/outputs/"
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(path + image_name + "_root_mask.tif", image)

    return f"Mask saved at {path}"

def extract_plant_name(path):

    plant_name = path[path.rfind("/") + 1:]
    name_no_file_extension = plant_name[:plant_name.find(".")]

    return name_no_file_extension

def get_output_directory():

    path = os.getcwd() + "/outputs"

    return path


#endregion
def load_in_model(path_to_json = "", path_to_weights = ""):

    if path_to_json != "" and path_to_weights != "":

        json_file = open(path_to_json)
        json_model = json_file.read()
        json_file.close()

        model_with_weights = model_from_json(json_model)
        model_with_weights.load_weights(path_to_weights)

        return model_with_weights

    else:

        current_dir = os.getcwd()

        json_file = open(current_dir + "/model_256.json")
        json_model = json_file.read()
        json_file.close()

        model_with_weights = model_from_json(json_model)
        model_with_weights.load_weights(current_dir + "/transfer_learning_weights256.h5")

        return model_with_weights


def find_end_node(graph):
    """
    Finds the end nodes in a given graph structure.

    Parameters:
    graph (dict): A dictionary representing the graph with keys 'node-id-pyphenotyper' and 'node-id-dst'.

    Returns:
    list: A list of end nodes in the graph.
    """

    src = list(graph['node-id-pyphenotyper'])
    end_nodes = []

    for destination in list(graph['node-id-dst']):

        if destination not in src:
            end_nodes.append(destination)

    return (end_nodes)


def split_image_with_roots(image, num_splits):
    """
    Splits an image with roots into specified number of segments and returns segment widths and bounds
    Parameters:
    image (ndarray): The image to be split
    num_splits (int): Number of segments to split the image into
    Returns:
    tuple: A tuple containing widths of each segment and the bounding coordinates (x_start, x_end) of each segment
    """

    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the width of each rectangle
    split_width = width // num_splits

    # List to store the split images
    split_images = []
    bounds_x = []

    # Iterate over each split
    for i in range(num_splits):
        # Define the region of interest (ROI)
        x_start = i * split_width
        x_end = (i + 1) * split_width
        roi = image[:, x_start:x_end]

        # Check if the ROI contains a full root
        if np.any(roi == 1):
            # If yes, append the original ROI to the list
            split_images.append(roi)
            bounds_x.append([x_start, x_end])
        else:
            # If not, adjust the ROI until a full root is found
            for j in range(x_start, x_end):
                adjusted_roi = image[:, x_start:j]
                if np.all(adjusted_roi == 1):
                    split_images.append(adjusted_roi)
                    bounds_x.append([x_start, j])
                    break

    width = []
    for i, split_img in enumerate(split_images):
        width.append(split_img.shape[1])

    return width, bounds_x

def get_roots_bboxes(image, stats):
    """
    Extracts bounding boxes of root structures from an image, based on provided statistical data.

    Args:
    image (ndarray): The image containing root structures.
    stats (ndarray): Statistical data used for identifying root structures.

    Returns:
    list: Coordinates of root bounding boxes.
    """

    number_of_plants = stats.shape[0]

    roots_coordinates = []

    for i in range(1, number_of_plants):

        x, y, w, h, _ = stats[i]
        roots_coordinates.append([y,y+h,x,x+w])

    return roots_coordinates

def root_to_graph(image, root_coordinates):
    """
    Converts a root image segment into a graph representation. Extracts a sub-image defined by root coordinates and processes it into a graph structure.

    Args:
    image (ndarray): The complete image of roots.
    root_coordinates (list): Coordinates defining the sub-image to process.

    Returns:
    DataFrame: Graph representation of the root segment.
    """
    y, y_max, x, x_max = root_coordinates
    plant = image[y:y_max, x:x_max]
    plant_skeleton = skimage.morphology.skeletonize(plant)
    plant_branch = summarize(Skeleton(plant_skeleton))

    return plant_branch


def sort_measurements(measurements, order):
    '''
    Function returns measurements in a provided order (from left to right)
    Args:
        measurements: array with measurements of the primary roots
        order: order in which plants should be

    Returns: ordered measurements

    '''

    measurements_in_order = [0, 0, 0, 0, 0]

    for k in range(len(order)):
        measurements_in_order[order[k] - 1] = measurements[k]

    return measurements_in_order

def landmark_detection(image, masks):
    """
    Detects landmarks in an image given the masks. Optionally returns coordinates of the landmarks.

    Parameters:
    image (ndarray): The image to process.
    masks (ndarray): The masks for segmentation.
    coordinates (bool, optional): If True, returns coordinates of landmarks. Defaults to False.

    Returns:
    ndarray or list: Either the image with landmarks or a list of coordinates, based on 'coordinates' parameter.

    """

    segmented_plants, stats = segment_plants(masks)

    roots_bboxes = get_roots_bboxes(segmented_plants, stats)
    plants_branches = [root_to_graph(segmented_plants, roots_bboxes[i]) for i in range(len(roots_bboxes))]

    for i in range(len(plants_branches)):
        branch = plants_branches[i]
        y, y_max, x, x_max = roots_bboxes[i]

        end_nodes = find_end_node(branch)
        prim_root_end = max(end_nodes)

        for j in range(len(branch)):

            center_coordinates = (branch['image-coord-pyphenotyper-1'][j] + x, branch['image-coord-pyphenotyper-0'][j] + y)
            center_coordinates_end = (branch['image-coord-dst-1'][j] + x, branch['image-coord-dst-0'][j] + y)

            radius = 10

            color_start = (0, 0, 255)
            color_end = (0, 0, 255)

            if branch['node-id-pyphenotyper'][j] == 0:
                color_start = (0, 255, 0)

            if branch['node-id-dst'][j] == prim_root_end:
                color_end = (0, 0, 0)

            elif branch['node-id-dst'][j] in end_nodes:
                color_end = (255, 0, 0)

            thickness = 2

            image = cv2.circle(image, center_coordinates, radius, color_start, thickness)
            image = cv2.circle(image, center_coordinates_end, radius, color_end, thickness)

    return image


def find_root_length(branch):
    '''
    Performs graph analysis and finds length of the root from given start-node to end-node
    Args:
        branch: graph representation of the root

    Returns: returns length of a root

    '''

    G = nx.from_pandas_edgelist(branch, source='node-id-pyphenotyper', target='node-id-dst', edge_attr='branch-distance')

    connected_components = list(nx.connected_components(G))
    start = min(connected_components[0])
    dist = max(connected_components[0])

    length = nx.dijkstra_path_length(G, start, dist, weight='branch-distance')

    return length

def plants_mapping(bboxes):
    '''
    Function returns the order of boxes starting with the first one (lowest x - on the right) finishing with the last one (highest x - on the left)
    Args:
        bboxes: array with bounding boxes of the plants in the images

    Returns: order in which plants should be ordered in the final dataset

    '''

    sorted_bboxes = sorted(bboxes, key=lambda x: x[2])

    plants_order = [sorted_bboxes.index(elem) + 1 for elem in bboxes]

    return plants_order


def measure_all_roots(image, masks):
    """
    Measures length of all primary roots in a segment, using prediction images.

    Args:
    segment_to_predict (ndarray): The segment to analyze for root measurements.
    prediction_image (ndarray): The prediction image used for measuring roots.

    Returns:
    tuple: Root measurements and order in which measurements should be
    """

    all_length = []

    segmented_plants, stats = segment_plants(masks)
    width, bounds = split_image_with_roots(masks, 5)

    non_germinated = width.count(0)
    non_germinated_indexes = [i for i, e in enumerate(width) if e == 0]

    roots_bboxes = get_roots_bboxes(segmented_plants, stats)

    width, bounds = split_image_with_roots(masks, 5)

    plants_branches = [root_to_graph(segmented_plants, roots_bboxes[i]) for i in range(len(roots_bboxes))]

    for i in range(len(plants_branches)):
        branch = plants_branches[i]

        all_length.append(find_root_length(branch))

    measurements_ordered = sort_measurements(all_length, plants_mapping(roots_bboxes))

    for i in range(non_germinated):
        measurements_ordered.remove(min(measurements_ordered))

    for j in range(len(non_germinated_indexes)):
        measurements_ordered.insert(non_germinated_indexes[j], 0)

    for i, item in enumerate(width):
        if item == 0:
            measurements_ordered[i] = 0.0

    return measurements_ordered


def measure_all_roots_young(image, masks, label):
    """
    Measures the roots of young plants given an image, masks and a label. For young plants a different patch_size is used in order to get more accurate predictions

    Parameters:
    path (str): The file path to the image.
    label (int): The label indicating the type of plant.

    Returns:
    list: A list of measurements for the roots of the young plants.
    """

    all_length = []

    if label == 7:

        kernel = np.ones((5, 5), dtype='uint8')
        mask_dialated = cv2.dilate(masks, kernel, iterations=1)
        _, segmented_plants, stats, _ = cv2.connectedComponentsWithStats(mask_dialated, connectivity=8)
        roots_bboxes = get_roots_bboxes(segmented_plants, stats[0:6])

        width, bounds = split_image_with_roots(masks, 5)
    else:
        segmented_plants, stats = segment_plants(masks)

        roots_bboxes = get_roots_bboxes(segmented_plants, stats)

        width, bounds = split_image_with_roots(masks, 5)

    plants_branches = [root_to_graph(segmented_plants, roots_bboxes[i]) for i in range(len(roots_bboxes))]

    for i in range(len(plants_branches)):
        branch = plants_branches[i]

        all_length.append(find_root_length(branch))

    measurements_ordered = sort_measurements(all_length, plants_mapping(roots_bboxes))

    if width[0] == 0:
        measurements_ordered[0] = measurements_ordered[1]
        measurements_ordered[1] = 0.0
        measurements_ordered[2] = 0.0
        # measurements_ordered[3] = 0.0
        measurements_ordered[4] = 0.0
        for i, item in enumerate(width):
            if i != 0 and item == 0:
                measurements_ordered[i] = 0.0
    else:
        for i, item in enumerate(width):
            if item == 0:
                measurements_ordered[i] = 0.0

    return measurements_ordered

def measure_root_grown_plants(path):
    '''
    Function takes in path to an image  and returns length of the primary roots in the petri dish in a left to right order
    Args:
        path: path to an image

    Returns: roots measurements in order from left to right

    '''

    model = load_in_model()

    data_for_predictions, dims = (datagen_for_predictions(path, 256, 1, True))  # prepare the data for prediction
    predictions = get_predictions(model, data_for_predictions)  # get predictions

    item_number = 0
    segment_to_predict = data_for_predictions[121 * item_number:121 * (item_number + 1)]
    segment_to_predict = segment_to_predict.reshape(11, 11, 1, 256, 256, 3)
    segment_to_predict = unpatchify(segment_to_predict, (2816, 2816, 3))

    predictions_segment = predictions[121 * item_number:121 * (item_number + 1)]  # choose prediction
    prediction_reshaped = predictions_segment.reshape(11, 11, 1, 256, 256, 1)  # reshape the prediction
    prediction_image = unpatchify(prediction_reshaped, (2816, 2816, 1))  # unpatchify it
    save_masks(prediction_image, extract_plant_name(path))

    return measure_all_roots(segment_to_predict, prediction_image)


def measure_root_young_plants(path, label):
    '''
    Function takes in path to an image and its label and returns length of the primary roots in the petri dish in a left to right order
    Args:
        path: path to an image
        label: label of the image

    Returns: roots measurements in order from left to right

    '''

    model = load_in_model()

    data_for_predictions, dims = datagen_for_predictions(path, 32, 1, True)
    predictions = get_predictions(model, data_for_predictions)

    item_number = 0
    segment_to_predict = data_for_predictions[7569 * item_number:7569 * (item_number + 1)]
    segment_to_predict = segment_to_predict.reshape(87, 87, 1, 32, 32, 3)
    segment_to_predict = unpatchify(segment_to_predict, (2784, 2784, 3))

    predictions_segment = predictions[7569 * item_number:7569 * (item_number + 1)]
    prediction_reshaped = predictions_segment.reshape(87, 87, 1, 32, 32, 1)
    prediction_image = unpatchify(prediction_reshaped, (2784, 2784, 1))

    save_masks(prediction_image, extract_plant_name(path))

    return measure_all_roots_young(segment_to_predict, prediction_image[400:800], label)


def get_measurements(path_to_data):
    """
    Gets measurements for a specified number of pictures.

    Parameters:
    number_of_pics (int): The number of pictures to process.

    Returns:
    dict: A dictionary containing the plant IDs and their corresponding measurements.
    """



    kaggle_data = {
        'Plant ID': [],
        'Length (px)': []
    }

    for plant in tqdm(glob.glob(path_to_data + "/*")):


        measurements_in_order = [0, 0, 0, 0, 0]


        measurements_in_order = measure_root_grown_plants(plant)

        plant_name = extract_plant_name(plant)

        current_plant = plant_name + '_plant_'

        for j in range(len(measurements_in_order)):
            kaggle_data['Plant ID'].append(current_plant + str(j + 1))
            kaggle_data['Length (px)'].append(measurements_in_order[j])

    csv_df = pd.DataFrame(kaggle_data)

    save_csv_data(csv_df)


    return f"All the outputs are saved at - {get_output_directory()}"

