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
import skan
from skimage.morphology import skeletonize
import networkx as nx

from data.data_processing import *
from utils.helpers import load_image, load_images_from_folder, create_folder, check_input_type
import os
import cv2
import numpy as np
import pandas as pd
import networkx as nx
from skimage.morphology import skeletonize

def predict_image(image_path: str, patch_size: int, segmentation_model, occlusion_inpainter, shoot_model, refinement_steps: int = 10)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict root and shoot masks for the given image using segmentation models.

    Authors: Vlad Matache 224108@buas.nl
             Francisco Ribeiro Mansilha 220387@buas.nl

    
    :param image_path: Image path.
    :param patch_size: Size of the patch.
    :param segmentation_model: Pre-loaded segmentation model.
    :param patching_model: Model or function for handling image patching.
    :param shoot_model: Pre-loaded shoot model.
    :return: Root and shoot masks.
    """
    # Load the image
    image = load_image(image_path)
    padded_image = padder(image, patch_size)

    # Extracting coordinates for ROI
    min_x, max_x, min_y, max_y = roi_extraction_coords_direct(padded_image)
    # Preparing patches for prediction
    patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    patches_reshaped = patches.reshape(-1, patch_size, patch_size, 1)
    
    # Predict with the root model (segmentation_model)
    root_predictions = segmentation_model.predict(patches_reshaped / 255.0, verbose=0)
    root_predictions_reshaped = root_predictions.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    root_mask = unpatchify(root_predictions_reshaped, padded_image.shape)
    root_mask = (root_mask > 0.5).astype(np.uint8)
    root_mask = set_outside_pixels_to_zero(root_mask, min_x, max_x, min_y, max_y)

    # Apply occlusion inpainting for refinement
    refined_mask = root_mask.copy()
    occ_patch_size = 256
    for _ in range(refinement_steps + 1):
        # Preparing patches for prediction
        refined_patches = patchify(refined_mask * 255, (occ_patch_size, occ_patch_size), step=occ_patch_size)
        refined_patches_reshaped = refined_patches.reshape(-1, occ_patch_size, occ_patch_size, 1)
        refined_predictions = occlusion_inpainter.predict(refined_patches_reshaped / 255.0, verbose=0)
        refined_predictions_reshaped = refined_predictions.reshape(patches.shape[0], patches.shape[1], occ_patch_size, occ_patch_size)
        refined_mask = unpatchify(refined_predictions_reshaped, padded_image.shape)
        refined_mask = (refined_mask > 0.5).astype(np.uint8)

    # Calculate the occlusion mask
    occlusion_mask = np.not_equal(root_mask, refined_mask).astype(np.uint8)
    occlusion_mask = set_outside_pixels_to_zero(occlusion_mask, min_x, max_x, min_y, max_y)

    # Predict with the shoot model
    shoot_predictions = shoot_model.predict(patches_reshaped / 255.0, verbose=0)
    shoot_predictions_reshaped = shoot_predictions.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    shoot_mask = unpatchify(shoot_predictions_reshaped, padded_image.shape)
    shoot_mask = (shoot_mask > 0.5).astype(np.uint8)
    shoot_mask = set_outside_pixels_to_zero(shoot_mask, min_x, max_x, min_y, max_y)

    # Manually removing some noise in the top and bottom of the shoot mask
    shoot_mask[:100, :] = 0 
    shoot_mask[-2000:, :] = 0

    return root_mask, shoot_mask, occlusion_mask

def save_predictions(image_path: str, filename: str, root_segmentation_model, occlusion_inpainter, shoot_segmentation_model, padder, output_folder: str, refinement_steps: int = 10, verbose: bool = False)-> None:
    """
    Saves predictions from root and shoot segmentation models for each image in the specified input folder.

    Author: Francisco Ribeiro Mansilha 220387@buas.nl

    
    :param image_path: Path to the image.
    :param filename: Name of the image file(used for output).
    :param root_segmentation_model: Pre-loaded root segmentation model.
    :param patching_model: Model or function for handling image patching.
    :param shoot_segmentation_model: Pre-loaded shoot segmentation model.
    :param padder: Function to pad images to the required size.
    """
    # Load image
    image = load_image(image_path)
    if verbose:
        print(f"Loaded image: {filename} with shape: {image.shape}")
    # Predict the root, shoot, and occlusion masks
    root_mask, shoot_mask, occlusion_mask = predict_image(
        image_path, 256, root_segmentation_model, occlusion_inpainter, shoot_model = shoot_segmentation_model, refinement_steps=refinement_steps
    )

    full_root_mask = root_mask + occlusion_mask
    

    # Create a new folder for each image
    image_folder_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, image_folder_name)
    create_folder(output_path)

    # Save the padded original image
    padded_image = padder(image, 256)
    cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_original_padded.png"), padded_image)
    if verbose:
        print(f"Original image saved for: {filename}")
    # Save the root mask
    cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_root_mask.png"), root_mask * 255)
    if verbose:
        print(f"Root mask saved for: {filename}")
    # Save the shoot mask
    cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_shoot_mask.png"), shoot_mask * 255)
    if verbose:
        print(f"Shoot mask saved for: {filename}")
    # Save the occlusion mask
    cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_occlusion_mask.png"), occlusion_mask * 255)
    if verbose:
        print(f"Occlusion mask saved for: {filename}")
    # Save the full root mask
    cv2.imwrite(os.path.join(output_path, f"{image_folder_name}_root_mask_inpainted.png"), full_root_mask * 255)
    if verbose:
        print(f"Full root mask saved for: {filename}")


def overlay_root_shoot_masks(input_folder: str)-> None:
    def overlay_masks_on_images(input_folder: str):
        """
        Overlay root and shoot masks on original images in the specified input folder.

        Author: Francisco Ribeiro Mansilha 220387@buas.nl

        
        :param input_folder (str): Path to the input folder containing original images and masks.
        """
        # Iterate through each subfolder in the input folder
        for root, dirs, files in os.walk(input_folder):
            for subdir in dirs:
                folder_path = os.path.join(root, subdir)
                original_image_path = None
                root_mask_path = None
                shoot_mask_path = None
                occlusion_mask_path = None

                # Identify the required files in each subfolder
                for file in os.listdir(folder_path):
                    if file.endswith("_original_padded.png"):
                        original_image_path = os.path.join(folder_path, file)
                    elif file.endswith("_root_mask.png"):
                        root_mask_path = os.path.join(folder_path, file)
                    elif file.endswith("_shoot_mask.png"):
                        shoot_mask_path = os.path.join(folder_path, file)
                    elif file.endswith("_occlusion_mask.png"):
                        occlusion_mask_path = os.path.join(folder_path, file)

                if original_image_path and root_mask_path and shoot_mask_path and occlusion_mask_path:
                    # Load the original image
                    original_image = load_image(original_image_path)
                    original_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay

                    # Load masks
                    root_mask = load_image(root_mask_path)
                    shoot_mask = load_image(shoot_mask_path)
                    occlusion_mask = load_image(occlusion_mask_path)

                    # Create colored overlays
                    red_overlay = np.zeros_like(original_colored)
                    green_overlay = np.zeros_like(original_colored)
                    blue_overlay = np.zeros_like(original_colored)

                    # Assign colors to the masks (Red for root, Green for shoot, Blue for occlusion)
                    red_overlay[root_mask == 255] = [0, 0, 255]
                    green_overlay[shoot_mask == 255] = [0, 255, 0]
                    blue_overlay[occlusion_mask == 255] = [255, 255, 0]

                    # Combine overlays with the original image
                    combined_overlay = cv2.addWeighted(original_colored, 1, red_overlay, 0.5, 0)
                    combined_overlay = cv2.addWeighted(combined_overlay, 1, green_overlay, 0.5, 0)
                    combined_overlay = cv2.addWeighted(combined_overlay, 1, blue_overlay, 0.5, 0)

                    # Save the overlayed image
                    overlayed_image_path = os.path.join(folder_path, f"{subdir}_overlayed.png")
                    cv2.imwrite(overlayed_image_path, combined_overlay)

    #Measurement functions#
    def process_single_skeleton(skeleton_branch_data, image: np.ndarray) -> float:
        """
        Process a single skeleton branch data and extract features.

        Author: Francisco Ribeiro Mansilha 220387@buas.nl

        
        :param skeleton_branch_data: Dataframe containing skeleton branch data.
        :param image (numpy.ndarray): Input image.
        :return: Length of the calculated path.
        """
        G = nx.from_pandas_edgelist(skeleton_branch_data, source='node-id-pyphenotyper', target='node-id-dst', edge_attr='branch-distance')
        
        for skeleton_id in skeleton_branch_data['skeleton-id'].unique():
            root_tips = skeleton_branch_data[skeleton_branch_data['skeleton-id'] == skeleton_id]

            if not root_tips.empty:
                min_row = root_tips.loc[root_tips['coord-pyphenotyper-0'].idxmin()]
                max_row = root_tips.loc[root_tips['coord-dst-0'].idxmax()]

                junction = (int(min_row['coord-pyphenotyper-0']), int(min_row['coord-pyphenotyper-1']))
                root_tip = (int(max_row['coord-dst-0']), int(max_row['coord-dst-1']))

                cv2.circle(image, junction[::-1], 15, (0, 255, 0), 4)
                cv2.circle(image, root_tip[::-1], 15, (0, 0, 255), 4)

                for _, row in skeleton_branch_data.iterrows():
                    src_node = row['node-id-pyphenotyper']
                    dst_node = row['node-id-dst']
                    distance = row['branch-distance']
                    G.add_edge(src_node, dst_node, branch_distance=distance)

                junction_node_id = min_row['node-id-pyphenotyper']
                root_tip_node_id = max_row['node-id-dst']

                length = nx.dijkstra_path_length(G, junction_node_id, root_tip_node_id, weight='branch_distance')
                
                return length

            
    def process_multiple_skeletons(skeleton_branch_data, image: np.ndarray)-> float:
        """
        Process multiple skeleton branch data and extract features.

        Author: Francisco Ribeiro Mansilha 220387@buas.nl

        
        :param skeleton_branch_data (pandas.DataFrame): Dataframe containing skeleton branch data.
        :param image (numpy.ndarray): Input image.
        :return: Length of the longest calculated path.
        """
        G = nx.from_pandas_edgelist(skeleton_branch_data, source='node-id-pyphenotyper', target='node-id-dst', edge_attr='branch-distance')
        lengths = []

        for skeleton_id in sorted(skeleton_branch_data['skeleton-id'].unique()):
            root_tips = skeleton_branch_data[skeleton_branch_data['skeleton-id'] == skeleton_id]

            if not root_tips.empty:
                min_row = root_tips.loc[root_tips['coord-pyphenotyper-0'].idxmin()]
                max_row = root_tips.loc[root_tips['coord-dst-0'].idxmax()]

                junction = (int(min_row['coord-pyphenotyper-0']), int(min_row['coord-pyphenotyper-1']))
                root_tip = (int(max_row['coord-dst-0']), int(max_row['coord-dst-1']))

                cv2.circle(image, junction[::-1], 15, (0, 255, 0), 4)
                cv2.circle(image, root_tip[::-1], 15, (0, 0, 255), 4)

                for _, row in skeleton_branch_data.iterrows():
                    src_node = row['node-id-pyphenotyper']
                    dst_node = row['node-id-dst']
                    distance = row['branch-distance']
                    G.add_edge(src_node, dst_node, branch_distance=distance)

                junction_node_id = min_row['node-id-pyphenotyper']
                root_tip_node_id = max_row['node-id-dst']

                length = nx.dijkstra_path_length(G, junction_node_id, root_tip_node_id, weight='branch_distance')
                lengths.append(length)

        return max(lengths) if lengths else 0

    def measure_images_in_folder(folder_path: str)-> pd.DataFrame:
        """
        Processes images in a given folder, identifying features' lengths and updating a DataFrame.

        Author: Francisco Ribeiro Mansilha 220387@buas.nl

        The function iterates over TIFF images in the specified folder, performs image processing to skeletonize the images,
        and calculates the length of the skeletonized features. It updates a DataFrame with these lengths and saves the
        DataFrame to a new CSV file. Additionally, it plots the original and skeletonized (or dilated skeletonized) images.

        
        :param folder_path (str): The path to the folder containing the images to be processed.
        :return: The updated DataFrame containing the lengths of features for each processed image.
        """

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
                image = load_image(image_path)
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
