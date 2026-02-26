#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Union
from typing import Tuple

# Libs
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
# Own modules
from colmap_wrapper.visualization import *


def ray_cast_aruco_corners(extrinsics: np.ndarray, intrinsics: np.ndarray, corners: tuple) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    n = x @ K^-1 @ R.T

    :param extrinsics:
    :param intrinsics:
    :param corners:
    :return:
    '''
    R, camera_origin = extrinsics[:3, :3], extrinsics[:3, 3]
    aruco_corners = np.concatenate((corners[0][0], np.ones((4, 1))), axis=1)
    rays = aruco_corners @ np.linalg.inv(intrinsics).T @ R.T
    rays_norm = rays / np.linalg.norm(rays, ord=2, axis=1, keepdims=True)

    return camera_origin, rays_norm


def load_image(image_path: str) -> np.ndarray:
    """
    Load Image. This takes almost 50% of the time. Would be nice if it is possible to speed up this process. Any
    ideas?

    :param image_path:
    :return:
    """
    return np.asarray(Image.open(image_path))


class ArucoDetection:
    def __init__(self, dict_type: int = aruco.DICT_4X4_1000):
        """
        More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

        @param dict_type:
        """
        self.dict_type = dict_type
        self.aruco_dict = aruco.Dictionary_get(dict_type)
        self.aruco_parameters = aruco.DetectorParameters_create()

        # aruco_parameters = aruco.DetectorParameters_create()
        # aruco_parameters.adaptiveThreshConstant = 9.0
        # aruco_parameters.adaptiveThreshWinSizeMax = 369
        # aruco_parameters.adaptiveThreshWinSizeMin = 7
        # aruco_parameters.adaptiveThreshWinSizeStep = 49
        # aruco_parameters.cornerRefinementWinSize = 9
        # aruco_parameters.minDistanceToBorder = 7
        # aruco_parameters.cornerRefinementMaxIterations = 149
        # aruco_parameters.minOtsuStdDev = 4.0
        #
        # aruco_parameters.minMarkerDistanceRate = 0.05
        # aruco_parameters.minMarkerPerimeterRate = 5
        # aruco_parameters.maxMarkerPerimeterRate = 10
        #
        #
        # aruco_parameters.polygonalApproxAccuracyRate = 0.05
        # aruco_parameters.minCornerDistanceRate = 0.05

    def detect_aruco_marker(self, image: Union[np.ndarray, str]) -> Tuple[tuple, np.ndarray, np.ndarray]:
        return detect_aruco_marker(image=image, dict_type=self.aruco_dict, aruco_parameters=self.aruco_parameters)


def detect_aruco_marker(image: np.ndarray, dict_type: int = aruco.DICT_4X4_1000,
                        aruco_parameters: cv2.aruco.DetectorParameters = aruco.DetectorParameters_create()) -> Tuple[
    tuple, np.ndarray]:
    """
    More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

    @param dict_type:
    @param image:
    @param aruco_parameters:
    """

    # Info: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    aruco_dict = aruco.Dictionary_get(dict_type)
    aruco_parameters = aruco.DetectorParameters_create()

    aruco_parameters.polygonalApproxAccuracyRate = 0.01
    aruco_parameters.minMarkerPerimeterRate = 0.1
    aruco_parameters.maxMarkerPerimeterRate = 4.0
    """
    Aruco Corners

        p1------------p2
        |             |
        |             |
        |             |
        |             |
        p4------------p3

    :param image:
    :return:
    """
    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        image = load_image(image_path=image)
    else:
        raise NotImplementedError

    image_size = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, aruco_id, rejected_img_points = aruco.detectMarkers(gray,
                                                                 aruco_dict,
                                                                 parameters=aruco_parameters)

    if aruco_id is None:
        return None, None, image_size

    if False:
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = aruco_id.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners_plot = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_plot
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 5)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 5)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 5)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 5)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), 5)
                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 10)

                plt.imshow(image, cmap='gray')
                plt.show()

    del gray
    del image

    return corners, aruco_id, image_size
