import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology

from helpers import task_3_helper as t3_helper
from helpers import task_4_helper as t4_helper

from helpers.config import Config as Config


def extract_plants(root_stats, shoot_stats):
    """
    This function constructs the coordinates of each plant by using a custom iterative algorithm imitating a
    recursive search to compare the intersection of the root and the seed since there will be always an
    intersection between the bounding boxes of the shoot and the root.

    :param root_stats: (array) the bounding box coordinates of the root mask in the format [[xmin, ymin, xmax, ymax]].
    :param shoot_stats: (array) the bounding box coordinates of the shoot mask in the format [[xmin, ymin, xmax, ymax]].
    :return:
        (array) the coordinates of the whole plant in the format [[xmin, ymin, xmax, ymax]].
    """

    def bbox_intersect(box1_min, box1_max, box2_min, box2_max):
        """
         This function encapsulates the logic if two boxes intersect with each other at a given point based on the min
         and max coordinates of each point for each box.

        :param box1_min: (array) with a shape of (,2) the xmin and ymin coordinate of the first box.
        :param box1_max: (array) with a shape of (,2) the xmax and ymax coordinate of the second box.
        :param box2_min: (array) with a shape of (,2) the xmin and ymin coordinate of the second box.
        :param box2_max: (array) with a shape of (,2) the xmax and ymax coordinate of the first box.
        :return:
            (bool) if the two boxes intersect with each other
        """

        return (
                box1_min[0] <= box2_max[0] and
                box1_max[0] >= box2_min[0] and
                box1_min[1] <= box2_max[1] and
                box1_max[1] >= box2_min[1]
        )

    ress = []

    for idx, (xs, ys, ws, hs, areaS) in enumerate(root_stats[1:]):
        points_min = (xs, ys)
        points_max = ((xs + ws), (ys + hs))

        for idx2, (xr, yr, wr, hr, areaR) in enumerate(shoot_stats[1:]):
            pointr_min = (xr, yr)
            pointr_max = ((xr + wr), (yr + hr))

            if bbox_intersect(points_min, points_max, pointr_min, pointr_max):
                ress.append([xr, yr, (xr + wr), (yr + hr), areaR])
                ress.append([xs, ys, (xs + ws), (ys + hs), areaR])
                break

    bboxes = ress.copy()
    if len(bboxes) == 0:
        return []
    count = 0
    while True:

        xs, ys, ws, hs, area = bboxes[0]
        points_min = (xs, ys)
        points_max = (ws, hs)
        # print(orig)

        should_break = True

        for idx2, (xr, yr, wr, hr, areaT) in enumerate(bboxes[1:]):

            pointr_min = (xr, yr)
            pointr_max = (wr, hr)

            if bbox_intersect(points_min, points_max, pointr_min, pointr_max):
                # print("Yes")
                # print(oth)
                should_break = False
                count = 0
                min_x = min(xs, xr)
                min_y = min(ys, yr)

                max_w = max(ws, wr)
                max_h = max(hs, hr)

                orig = [min_x, min_y, max_w, max_h, area]

                del bboxes[idx2 + 1]
                del bboxes[0]
                if min_y < 2000:
                    bboxes.append(orig)

                break

        if should_break:
            bboxes.append(bboxes[0])
            del bboxes[0]
            count += 1

        if count == len(bboxes):
            break

    return bboxes


def get_plants_ordered(img_preds, image_metadata, all_models, verbose=True):
    """
    Performs the two main checks related to the recognition and extraction of the 5 plants in each image.
    The first check calculates the intersection between the root and the shoot of each plant and based on their position
    it extracts the plants from the original prediction.
    The second check finds the region of the small plants using a predefined box calculated by using a constant of 550
    by 550px, then the extracted region is passed to the model to recognize the small plants. If the recognition is
    bigger than the predefined then the coordinates of the first check will be taken, otherwise the coordinates of the
    second check (recognition) will be taken. If nothing is recognized then a default 20x20 black box is returned
    indicating that there's a seed in the corresponding position.
    Meanwhile, indicating the position of the plants, they are being ordered from left-to-right to make the movement of
    the robot arm more efficient and their absolute coordinates are being calculated.


    :param img_preds: (array) containing the 4 predictions of each image.
    :param image_metadata: (object) if the type of this parameter is str then the image is read from the
    local filesystem while if the type is an array then the image is copied to eliminate reference problems with the
    image outside the scope of the function.
    :param all_models: (array) which contains the models that will be used to predict the masks.
    :param verbose: (bool) specify if the logs should be displayed.
    :return:
        (tuple) in the format (array, array)
        The first array contains the cropped plant. The second array contains the absolute coordinates of the crops.
    """
    normal_object_size = 800
    small_object_size = 200

    retval, labels, root_stats, centroids = t3_helper.get_stats(img_preds[0], normal_object_size)
    retval2, labels2, shoot_stats, centroids2 = t3_helper.get_stats(img_preds[2], normal_object_size)

    if type(image_metadata) is str or type(image_metadata) is np.str_:
        image = cv2.imread(image_metadata)
    else:
        image = image_metadata.copy()

    bboxes = np.array(extract_plants(root_stats, shoot_stats))

    patches_to_write = []
    coordinates_padding = []

    if len(bboxes) != 5:
        ymin = 200
        ymax = 712
        margin = 200
        WIDTH_CONST = 550

        def find_seeds(image, WIDTH_CONST):
            """
            Extracts the region of the seeds using the predefined constant of 550. Since the images are roughly 
            2800-2900 pixels when cropped, the optimal solution for 5 plants is WIDTH_CONST = 550. This function 
            requires the global variables (ymin, ymax, margin) which are used to locate the seeds.
            
            :param image: (array) representing the original image in pixels. 
            :param WIDTH_CONST: (int) defines the size of the extracted region.
            :return: 
            (array) containing the extracted regions in a pixel format (r,g,b). 
            """
            seed_crops = []
            for i in range(5):
                width = i * WIDTH_CONST
                if i == 0:
                    width += margin
                if i == 5:
                    WIDTH_CONST -= margin
                new_width = width + WIDTH_CONST

                global seed_crop
                seed_crop = image[ymin:ymax, width:new_width]

                seed_crops.append(seed_crop)

            return seed_crops

        seed_crops = find_seeds(image, WIDTH_CONST)
        avg_iou, preds_crops, class_iou = t4_helper.predict(all_models, seed_crops, Config.class_colors, Config.classes,
                                                            Config.raw_data_dir, Config.raw_mask_dir, Config.patch_size,
                                                            should_pad=True,
                                                            verbose=False)
        if verbose is True:
            fig, ax = plt.subplots(1, 5, figsize=(12, 12))

        count = 0
        for idx, small_plant in enumerate(preds_crops):
            small_img_preds = preds_crops[f"{small_plant}"]
            small_retval, small_labels, small_root_stats, small_centroids = t3_helper.get_stats(small_img_preds[0],
                                                                                                small_object_size)

            small_retval2, small_labels2, small_shoot_stats, small_centroids2 = t3_helper.get_stats(
                small_img_preds[2],
                small_object_size)

            small_bboxes = np.array(extract_plants(small_root_stats, small_shoot_stats))

            tempp = t4_helper.padder(seed_crops[idx], Config.patch_size)
            temp2 = image.copy()
            is_temp2 = False

            if len(small_bboxes) != 0 and len(bboxes) > idx - count and len(
                    bboxes[bboxes[:, 0].argsort()]) > idx - count:
                x_min, y_min, x_max, y_max, area = small_bboxes[small_bboxes[:, 0].argsort()][0]
                if y_max < 600:
                    tempp = cv2.rectangle(tempp, (x_min, y_min), (x_max, y_max), 2, 5)
                    count += 1
                    crop = small_labels[y_min:y_max, x_min:x_max]
                    coordinates_padding.append([((idx + 1) * WIDTH_CONST) + x_min, 200 + y_min])

                else:
                    temp_box = bboxes[bboxes[:, 0].argsort()][idx - count]

                    crop = labels[temp_box[1]:temp_box[3], temp_box[0]:temp_box[2]]
                    temp2 = cv2.rectangle(temp2, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), 2, 5)
                    is_temp2 = True
                    coordinates_padding.append([((idx + 1) * WIDTH_CONST) + temp_box[0], 200 + temp_box[1]])

                patches_to_write.append(crop)
                # cv2.imwrite(f"./data/measuring/block/test_image_{idx_img + 1}_plant_{idx + 1}.png", crop)
            else:
                patches_to_write.append(np.zeros((20, 20), np.uint8))
                coordinates_padding.append([(idx + 1) * WIDTH_CONST, ymin + (ymin / 2)])
                # cv2.imwrite(f"./data/measuring/block/test_image_{idx_img + 1}_plant_{idx + 1}.png",
                #             np.zeros((20, 20), np.uint8))
                count += 1
            if is_temp2 and verbose is True:
                ax[idx].imshow(temp2)
            elif verbose is True:
                ax[idx].imshow(tempp)
        if verbose is True:
            plt.show()

    elif len(bboxes) != 0:
        tempp = image.copy()

        for idx_bboxes, (x_min, y_min, x_max, y_max, area) in enumerate(bboxes[bboxes[:, 0].argsort()]):
            tempp = cv2.rectangle(tempp, (x_min, y_min), (x_max, y_max), 2, 5)
            crop = labels[y_min:y_max, x_min:x_max]
            patches_to_write.append(crop)
            coordinates_padding.append([x_min, y_min])
            # cv2.imwrite(f"./data/measuring/block/test_image_{idx_img + 1}_plant_{idx_bboxes + 1}.png", crop)

        if verbose is True:
            plt.imshow(tempp)
            plt.show()

    return patches_to_write, coordinates_padding
