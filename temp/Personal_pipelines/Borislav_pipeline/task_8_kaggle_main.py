import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model

from helpers import task_2_helper as t2_helper
from helpers import task_4_helper as t4_helper
from helpers import task_5_helper as t5_helper
from helpers import task_6_helper as t6_helper

from helpers.config import Config as Config

from skimage.morphology import skeletonize, remove_small_objects

# Defining the local constants used in the pipeline.
test_dir = "./test/"
data = {"Plant ID": [], "Length (px)": [], "Coordinates": []}

models = {}
# Load the DL segmentation models
for el in Config.classes:
    model_name = f"./models/{el}_model.h5"
    models[el] = load_model(model_name, custom_objects={"f1": t4_helper.f1, "iou": t4_helper.iou_rgb})


def detect(cropped, idx_img):
    """
    This function runs the passed image through the initialized global models and extracts the corresponding coordinates
    of each detection. The verbose option is set to off to optimize the processing speed. Debug info of the extracted
    detections are saved depending on the input parameters.

    :param cropped: (array) representing the cropped image in pixels.
    :param idx_img: (idx) the index number of the passed image.
    :return:
        (tuple)  in the format:
        (array, array)
        ([the extracted plants in pixels],
        [the coordinates of the padding that should be added to the cropped image to reconstruct the coordinates]
        )
    """
    all_models = models.values()
    print(cropped.shape)
    cv2.imwrite(f"temp/temp.png", cropped)
    avg_iou, preds, class_iou = t4_helper.predict(all_models, [cropped], Config.class_colors, Config.classes,
                                                  Config.raw_data_dir,
                                                  Config.raw_mask_dir, Config.patch_size, should_pad=True,
                                                  verbose=False)
    print("IoU:", avg_iou)

    img_preds = preds[f"{0}"]

    plants, padding_coordinates = t5_helper.get_plants_ordered(img_preds, cropped, all_models, verbose=False)
    print(f"Plants number: {len(plants)}")
    for idx, el in enumerate(plants):
        pass
        # cv2.imwrite(os.path.join(f"./test_image_{idx_img + 1}_plant_{idx + 1}.png"), el)

    return plants, padding_coordinates


def preprocess_image(location, save_location, idx_img, img=None):
    """
    This function encapsulates the logic behind the preprocessing, the detection and measurement of a single image
    through the pipeline. It calls the required tasks step by step and executes them in order. The results from each
    detection its measurement and corresponding metadata are added to the global (dict) variable data.

    :param location: (str) the path location of the image that should be passed through the pipeline.
    :param save_location: (str) the location where the debug measurement results will be saved. If the value is None
    then the results won't be saved.
    :param idx_img: (int) The current image index. It's used to ensure that the saved files will follow the defined
    naming conventions.
    :param img: (array) representing the original image in pixels. If the argument is None, then the image using the
    argument location will be used.
    :return:
    (dict) in the following format:
    {
    "Plant ID": ['image_1_plant_1'],
    "Length (px)": [(length)],
    "Coordinates": [(x,y)]
    }
    """
    if img is None:
        img = cv2.imread(location)

    print(location)
    # Task 2
    cropped, x_min, x_max, y_min, y_max = t2_helper.crop_image(img, img.shape[0], img.shape[1])
    # Task 4 - 5
    detection_results, padding_coordinates = detect(cropped, idx_img)
    # Task 6 - 7

    for idx_p, im in enumerate(detection_results):
        im = np.array(im, dtype=np.uint8)
        if im.shape == (20, 20, 3):
            data["Plant ID"].append(location.split("/")[-1])
            data["Length (px)"].append(0)
            return None

        temp = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) * 255

        kernel = np.ones((5, 5), dtype="uint8")
        im_dilation = cv2.dilate(temp, kernel, iterations=1)

        print("Skeletonizing image")
        skeleton = skeletonize(im_dilation)

        print("Binarizing image")
        binarized = np.where(cv2.cvtColor(skeleton, cv2.COLOR_RGBA2GRAY) > 0.1, 1, 0) * 255
        # processed = remove_small_objects(binarized.astype(bool), min_size=500, connectivity=2).astype(int)
        # cv2.imwrite(test_dir + "demo3.png", processed * 255)

        # processed = cv2.imread("image_test2.png", cv2.THRESH_BINARY)
        ay, ax = np.where(binarized != [0])
        ay = ay * -1
        nodes_coord = list(zip(ax, ay))
        print("Creating Graph")
        G_init = t6_helper.create_graph(nodes_coord)
        print("Analyzing")
        df = t6_helper.measure(G_init, is_naive=True)
        if type(df) == int and df == -11:
            print("Binarizing image")
            binarized = np.where(cv2.cvtColor(skeleton, cv2.COLOR_RGBA2GRAY) > 0.1, 1, 0)
            processed = remove_small_objects(binarized.astype(bool), min_size=1000, connectivity=2).astype(int)

            ay, ax = np.where(processed != [0])
            ay = ay * -1

            nodes_coord = list(zip(ax, ay))

            print("Creating Graph")
            G_init = t6_helper.create_graph(nodes_coord)
            print("Analyzing")
            df = t6_helper.measure(G_init, is_naive=True)

        finalename = location.split("/")[-1]

        if type(df) is not pd.DataFrame and df == -1:
            data["Plant ID"].append(finalename + finalename.split(".")[0] + f"_plant_{idx_p + 1}.tif")
            data["Length (px)"].append(0)
            data["Coordinates"].append([padding_coordinates[idx_p][0], padding_coordinates[idx_p][1]])
            continue
        print("Creating Result")
        G_morphometric = t6_helper.create_graph(nodes_coordinates=df['nodes'].values)

        print("Drawing Result")
        img_arr = t6_helper.draw_graph_t0_img(G_morphometric, "Morphometric analysis", color=df['color'], node_size=10,
                                              fig_size=binarized.shape)

        print("Saving Result")
        # cv2.imwrite(location.replace("cropped", "result"), img_arr)
        if save_location is not None:
            cv2.imwrite(location.replace("raw/", f"result2/test_image{idx_img}_plant_{idx_p + 1}"), img_arr)
        print(location.split("/")[-1])
        data["Plant ID"].append(finalename + finalename.split(".")[0] + f"_plant_{idx_p + 1}.tif")
        data["Length (px)"].append(len(df.loc[df['color'] == "green"]))
        coordinates = []
        for x, y in df['nodes'].values:
            coordinates.append([
                abs(x) + padding_coordinates[idx_p][0],
                abs(y) + padding_coordinates[idx_p][1]
            ])
        data["Coordinates"].append(coordinates)

    return data


def get_coordinates(img_array):
    """
    Augments the input from the arguments passed from the terminal. It's used to integrate the pipeline to the Robotics.

    :param img_array: (array) representing the original image in pixels.
    :return:
    (pd.DataFrame) in the following format:
    {
    "Plant ID": ['image_1_plant_1'],
    "Length (px)": [(length)],
    "Coordinates": [(x,y)]
    }
    """
    data = preprocess_image(location='/plant', save_location=None, idx_img=0, img=img_array)
    return pd.DataFrame(data)


if __name__ == "__main__":
    """
    The main function of the pipeline. It reads the arguments passed trough the terminal and reads the image depending 
    on the passed arguments. It also saves the results returned from the pipeline to a file for the Kaggle competition. 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", help="Specify the directory with images")
    parser.add_argument("--output_dir", help="Specify the output directory")
    parser.add_argument("--image", help="Specify the image.")

    args = parser.parse_args()

    # if args['dir'] is None or args['output_dir'] is None:
    #     print("invalid input arguments --dir or --output_dir")
    #     exit(0)
    # directory_path = args['dir']
    # TODO Configure the PATH to input images
    directory_path = '..../'
    # save_directory_path = args['output_dir']
    # TODO Configure the PATH to the location where the output images will be saved
    save_directory_path = '..../'
    is_kaggle = True
    image_path = None

    if directory_path and image_path:

        absolute_path = os.path.join(directory_path, image_path)
        data = preprocess_image(absolute_path, save_directory_path, 0)
    elif image_path:

        data = preprocess_image(image_path, save_directory_path, 0)
    elif directory_path:

        for root, dirs, files in os.walk(directory_path, topdown=True):
            for idx_img, name in enumerate(files):
                current_path = os.path.join(root, name)
                data = preprocess_image(current_path, save_directory_path, idx_img)
    else:
        print("Invalid arguments. The pipeline cannot proceed.")
        exit(0)
    df = pd.DataFrame(data)
    if is_kaggle:
        df[['Plant ID', 'Length (px)']].to_csv(directory_path + "measurements.csv")
    else:
        print(df['Coordinates'])
