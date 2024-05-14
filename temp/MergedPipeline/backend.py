import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skan import draw
import networkx as nx

from metrics import f1,iou

def create_folder(folder_name):
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def find_start_end(array):
    '''
    Takes is an image as array and return start and end coordinates of the petri dish
    :param array: image as a numpy array
    :return: start and end coordinates of the petri dish
    '''
    st = np.argmin(array)
    end = np.argmax(array)

    return [st, end]



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

    height = x[0] - x[1]
    width = y[0] - y[1]
    return [y[1] + 185 , y[1] + height , x[1] + 15, x[1] + width - 20]

def set_outside_pixels_to_zero(image, min_x, max_x, min_y, max_y):

    average_pixel_value = np.mean(image) + 90
    roi_extracted_image = image
    roi_extracted_image[0:min_x] = average_pixel_value
    roi_extracted_image[max_x:] = average_pixel_value
    roi_extracted_image[:, 0: min_y] = average_pixel_value
    roi_extracted_image[:, max_y:] = average_pixel_value
    
    return roi_extracted_image

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


def padder(image, patch_size):
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)
    
    return padded_image

# Predict root and shoot mask
def predict_image(image, patch_size, segmentation_model, patching_model, shoot_model):

    padded_image = padder(image, patch_size)
    
    # Preparing patches for prediction
    patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    patches_reshaped = patches.reshape(-1, patch_size, patch_size, 1)
    
    # Predict with the root model (segmentation_model + patching_model for discontinuities)
    root_predictions = segmentation_model.predict(patches_reshaped / 255.0, verbose = 0)
    # Assuming patching_model is another model to refine the root predictions
    #root_predictions = patching_model.predict(root_predictions, verbose = 0)
    root_predictions_reshaped = root_predictions.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    root_mask = unpatchify(root_predictions_reshaped, padded_image.shape)
    root_mask = (root_mask > 0.5).astype(np.uint8)  # Thresholding to create a binary mask
    
    # Predict with the shoot model
    shoot_predictions = shoot_model.predict(patches_reshaped / 255.0, verbose = 0)
    shoot_predictions_reshaped = shoot_predictions.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    shoot_mask = unpatchify(shoot_predictions_reshaped, padded_image.shape)
    shoot_mask = (shoot_mask > 0.5).astype(np.uint8)  # Thresholding to create a binary mask

    # Manually removing some noise in the top and bottom of the shoot mask
    shoot_mask[:100, :] = 0 
    shoot_mask[-2000:, :] = 0
    return root_mask, shoot_mask

def save_model_predictions(input_folder, root_segmentation_model, patching_model, shoot_segmentation_model, padder, output_folder):
    """
    Saves predictions from root and shoot segmentation models for each image in the specified input folder.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - root_segmentation_model: Pre-loaded root segmentation model.
    - patching_model: Model or function for handling image patching.
    - shoot_segmentation_model: Pre-loaded shoot segmentation model.
    - padder: Function to pad images to the required size.
    """

    # Iterate through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".tif") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, 0)  # Assuming 0 for grayscale reading, adjust if necessary
            
            # Assuming predict_image is a function that takes an image and returns root and shoot masks
            root_mask, shoot_mask = predict_image(image, 256, root_segmentation_model, patching_model, shoot_segmentation_model)

            # Create a new folder for each image
            image_folder_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, image_folder_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Save the padded original image
            padded_image = padder(image, 256)
            cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_original_padded.png"), padded_image)

            # Save the root mask
            cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_root_mask.png"), root_mask * 255)

            # Save the shoot mask
            cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_shoot_mask.png"), shoot_mask * 255)


def overlay_root_shoot_masks(input_folder):
    for root, dirs, files in os.walk(input_folder):
        for subdir in dirs:
            folder_path = os.path.join(root, subdir)
            original_image_path = None
            root_mask_path = None
            shoot_mask_path = None

            # Identify the required files in each subfolder
            for file in os.listdir(folder_path):
                if file.endswith("_original_padded.png"):
                    original_image_path = os.path.join(folder_path, file)
                elif file.endswith("_root_mask.png"):
                    root_mask_path = os.path.join(folder_path, file)
                elif file.endswith("_shoot_mask.png"):
                    shoot_mask_path = os.path.join(folder_path, file)

            if original_image_path and root_mask_path and shoot_mask_path:
                # Load the original image
                original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
                original_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay

                # Load masks
                root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)
                shoot_mask = cv2.imread(shoot_mask_path, cv2.IMREAD_GRAYSCALE)

                # Create colored overlays
                red_overlay = np.zeros_like(original_colored)
                green_overlay = np.zeros_like(original_colored)

                # Assign colors to the masks (Red for root, Green for shoot)
                red_overlay[root_mask == 255] = [0, 0, 255]
                green_overlay[shoot_mask == 255] = [0, 255, 0]

                # Combine overlays with the original image
                combined_overlay = cv2.addWeighted(cv2.addWeighted(original_colored, 1, red_overlay, 0.5, 0), 1, green_overlay, 0.5, 0)

                # Save the overlayed image
                overlayed_image_path = os.path.join(folder_path, f"{subdir}_overlayed.png")
                cv2.imwrite(overlayed_image_path, combined_overlay)


def process_image_rois(directory_path):
    """
    Processes each root mask image in the specified directory, extracting and saving ROIs (with the original image size)
    and their area details into CSV files (without using the .append method which is deprecated in pandas 2.0).

    Parameters:
    - directory_path: The path to the directory containing the root mask images to process.
    """
    # Iterate through each subfolder in the specified directory
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a root mask image
            if file.endswith("_root_mask.png"):
                root_mask_path = os.path.join(subdir, file)
                root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)

                # Connected components labeling on the root mask
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(root_mask)

                # Initialize a list to store data about each ROI
                roi_data = []

                mask = np.zeros_like(root_mask)
                for label, value in enumerate([255 if (stats[l, cv2.CC_STAT_AREA] > 100 and
                                                       stats[l, cv2.CC_STAT_TOP] < 800 and
                                                       stats[l, cv2.CC_STAT_TOP] > 400) else 0
                                               for l in range(1, retval)], start=1):
                    mask[labels == label] = value

                final_image = cv2.bitwise_and(root_mask, root_mask, mask=mask)

                for i in range(5):
                    start_col = i * final_image.shape[1] // 5
                    end_col = (i + 1) * final_image.shape[1] // 5
                    roi = final_image[:, start_col:end_col]

                    # Calculate ROI area
                    roi_area = np.count_nonzero(roi)

                    # Add ROI data to the list
                    roi_data.append({'ROI': f'roi_{i+1}', 'Area': roi_area})

                    # Create a black image of the original size and add the ROI
                    original_size_image = np.zeros_like(root_mask)
                    original_size_image[:, start_col:end_col] = roi

                    # Save the new image
                    roi_filename = f"roi_{i+1}.png"
                    roi_path = os.path.join(subdir, roi_filename)
                    cv2.imwrite(roi_path, original_size_image)

                # Convert the list of dictionaries to a DataFrame
                plant_area = pd.DataFrame(roi_data)

                # Save the DataFrame to a CSV file in the same folder as the image
                plant_area_csv_path = os.path.join(subdir, file.replace("_root_mask.png", "_plant_area.csv"))
                plant_area.to_csv(plant_area_csv_path, index=False)


#Measurement functions#


# Helper Function
def process_single_skeleton(skeleton_branch_data, image):
    # Create a graph from the provided skeleton branch data
    G = nx.from_pandas_edgelist(skeleton_branch_data, source='node-id-pyphenotyper', target='node-id-dst', edge_attr='branch-distance')
    
    # Loop through unique skeleton IDs
    for skeleton_id in skeleton_branch_data['skeleton-id'].unique():
        root_tips = skeleton_branch_data[skeleton_branch_data['skeleton-id'] == skeleton_id]

        # Check if there are any root tips for this skeleton
        if not root_tips.empty:
            # Find the minimum and maximum rows based on source and destination coordinates
            min_row = root_tips.loc[root_tips['coord-pyphenotyper-0'].idxmin()]
            max_row = root_tips.loc[root_tips['coord-dst-0'].idxmax()]

            # Extract coordinates for the junction and root tip
            junction = (int(min_row['coord-pyphenotyper-0']), int(min_row['coord-pyphenotyper-1']))
            root_tip = (int(max_row['coord-dst-0']), int(max_row['coord-dst-1']))

            # Draw circles on the image to mark the junction and root tip
            cv2.circle(image, junction[::-1], 15, (0, 255, 0), 4)  # Green circle for junction
            cv2.circle(image, root_tip[::-1], 15, (0, 0, 255), 4)  # Red circle for root tip

            # Add edges from the skeleton data to the graph
            for _, row in skeleton_branch_data.iterrows():
                src_node = row['node-id-pyphenotyper']
                dst_node = row['node-id-dst']
                distance = row['branch-distance']
                G.add_edge(src_node, dst_node, branch_distance=distance)

            junction_node_id = min_row['node-id-pyphenotyper']
            root_tip_node_id = max_row['node-id-dst']

            # Calculate the length of the path from junction to root tip in the graph
            length = nx.dijkstra_path_length(G, junction_node_id, root_tip_node_id, weight='branch_distance')
            
            # Return the calculated length
            return length
        
# Helper Function
def process_multiple_skeletons(skeleton_branch_data, image):
    # Create a graph from the provided skeleton branch data
    G = nx.from_pandas_edgelist(skeleton_branch_data, source='node-id-pyphenotyper', target='node-id-dst', edge_attr='branch-distance')

    # Initialize a list to store lengths of processed skeletons
    lengths = []
    
    # Loop through unique skeleton IDs (sorted)
    for skeleton_id in sorted(skeleton_branch_data['skeleton-id'].unique()):
        root_tips = skeleton_branch_data[skeleton_branch_data['skeleton-id'] == skeleton_id]

        # Check if there are any root tips for this skeleton
        if not root_tips.empty:
            # Find the minimum and maximum rows based on source and destination coordinates
            min_row = root_tips.loc[root_tips['coord-pyphenotyper-0'].idxmin()]
            max_row = root_tips.loc[root_tips['coord-dst-0'].idxmax()]

            # Extract coordinates for the junction and root tip
            junction = (int(min_row['coord-pyphenotyper-0']), int(min_row['coord-pyphenotyper-1']))
            root_tip = (int(max_row['coord-dst-0']), int(max_row['coord-dst-1']))

            # Draw circles on the image to mark the junction and root tip
            cv2.circle(image, junction[::-1], 15, (0, 255, 0), 4)  # Green circle for junction
            cv2.circle(image, root_tip[::-1], 15, (0, 0, 255), 4)  # Red circle for root tip

            # Add edges from the skeleton data to the graph
            for _, row in skeleton_branch_data.iterrows():
                src_node = row['node-id-pyphenotyper']
                dst_node = row['node-id-dst']
                distance = row['branch-distance']
                G.add_edge(src_node, dst_node, branch_distance=distance)

            junction_node_id = min_row['node-id-pyphenotyper']
            root_tip_node_id = max_row['node-id-dst']

            # Calculate the length of the path from junction to root tip in the graph
            length = nx.dijkstra_path_length(G, junction_node_id, root_tip_node_id, weight='branch_distance')
            
            # Append the length to the list
            lengths.append(length)

    # Return the maximum length among all processed skeletons, or 0 if no skeletons were processed
    return max(lengths) if lengths else 0

# Main Measurement Function
def measure_images_in_folder(folder_path):
    """
    Processes images in a given folder, identifying features' lengths and updating a DataFrame.

    The function iterates over TIFF images in the specified folder, performs image processing to skeletonize the images,
    and calculates the length of the skeletonized features. It updates a DataFrame with these lengths and saves the
    DataFrame to a new CSV file. Additionally, it plots the original and skeletonized (or dilated skeletonized) images.

    Parameters:
    - folder_path (str): The path to the folder containing the images to be processed.

    Returns:
    - DataFrame: The updated DataFrame containing the lengths of features for each processed image.
    """

    # Initialize an empty list to store results (unused in the provided snippet)
    results = []

    # Read a CSV file into a DataFrame and initialize a new column for storing lengths
    plant_id = []
    lengths = []
    counter = 1
    # Iterate over all files in the given folder path
    for filename in os.listdir(folder_path):
        original_image = folder_path.replace('plants/','')
        
        # Process only TIFF files
        if 'roi' in filename:
            
            # Construct the full path to the image and load it
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if np.count_nonzero(image) != 0:
                # Perform skeletonization on the image
                skeleton = skeletonize(image)
                # Summarize the skeleton data, likely extracting features such as branch lengths
                skeleton_branch_data = summarize(Skeleton(skeleton))

                # Process the skeleton based on the number of unique skeleton IDs found
                if len(skeleton_branch_data['skeleton-id'].unique()) == 1:
                    # Process a single skeleton
                    length = process_single_skeleton(skeleton_branch_data, image)
                    plant_id.append(original_image + '_plant' + str(counter))
                    lengths.append(length)
                    counter += 1
                else:
                    # Dilate the image and re-skeletonize for multiple skeletons
                    kernel = np.ones((5, 5), dtype="uint8")
                    im_blobs_dilation = cv2.dilate(image.astype(np.uint8), kernel, iterations=4)
                    skeleton_dilated = skeletonize(im_blobs_dilation)
                    skeleton_branch_data_dilated = summarize(Skeleton(skeleton_dilated))
                    length = process_multiple_skeletons(skeleton_branch_data_dilated, image)
                    plant_id.append(original_image + '_plant_' + str(counter))
                    lengths.append(length)
                    counter += 1
                
            else:
                length = 0
                plant_id.append(original_image + '_plant_' + str(counter))
                lengths.append(length)
                counter += 1
            
    # Save the updated DataFrame to a new CSV file
    df = pd.DataFrame({'Plant ID': plant_id, 'Length (px)': lengths})
    df.to_csv(folder_path + '/measurements.csv', index=False)

    # Return the updated DataFrame
    return df
