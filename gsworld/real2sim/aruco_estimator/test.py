#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import argparse

from colmap_wrapper.colmap import COLMAP

# Own modules
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor, DEBUG
from aruco_estimator import download

# Libs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate scale factor for COLMAP projects with aruco markers.')
    parser.add_argument('--colmap_project', type=str, help='Path to COLMAP project')
    parser.add_argument('--dense_model', type=str, help='name to the dense model', default='fused.ply')
    parser.add_argument('--aruco_size', type=float, help='Size of the aruco marker in meter.', default=0.15)
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
    parser.add_argument('--point_size', type=float, help='Point size of the visualized dense point cloud. '
                                                         'Depending on the number of points in the model. '
                                                         'Between 0.001 and 2', default=0.1)
    parser.add_argument('--frustum_size', type=float, help='Size of the visualized camera frustums. '
                                                           'Between 0 (small) and 1 (large)', default=0.5)
    parser.add_argument('--test_data', action='store_true', help='Download and try out test data')
    args = parser.parse_args()

    if args.test_data:
        # Download example dataset. Door dataset is roughly 200 MB
        dataset = download.Dataset()
        dataset.download_door_dataset()

        args.colmap_project = dataset.dataset_path
        args.aruco_size = dataset.scale

    if isinstance(args.colmap_project, type(None)):
        raise ValueError('--colmap_project is empty! Please select a path to our colmap project or test it with our '
                         'dataset by setting the flag --test_data')

    project = COLMAP(project_path=args.colmap_project, image_resize=0.4)

    # Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
    aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project,
                                          aruco_size=args.aruco_size,
                                          dense_path=args.dense_model)
    aruco_distance, aruco_corners_3d = aruco_scale_factor.run()
    print('Size of the unscaled aruco markers: ', aruco_distance)

    # Calculate scaling factor and apply to scene
    dense, scale_factor = aruco_scale_factor.apply()
    print('Point cloud and poses are scaled by: ', scale_factor)
    print('Size of the scaled (true to scale) aruco markers in meters: ', aruco_distance * scale_factor)

    if DEBUG:
        aruco_scale_factor.analyze()

    # Visualization of the scene and rays BEFORE scaling. This might be necessary for debugging
    if args.visualize:
        from aruco_estimator.visualization import ArucoVisualization

        vis = ArucoVisualization(aruco_colmap=aruco_scale_factor)
        vis.visualization(frustum_scale=0.7, point_size=0.1)

    # Todo: Save output, PCD and poses. Visualize!
    aruco_scale_factor.write_data()
