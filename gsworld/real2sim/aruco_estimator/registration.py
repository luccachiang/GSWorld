#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import numpy as np
import os
import open3d as o3d
from copy import deepcopy
from typing import Union

# Own modules
from colmap_wrapper.colmap import COLMAP
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from aruco_estimator.visualization import ArucoVisualization
from aruco_estimator.utils import align_point_set, plot_aligned_pointset, manual_registration


class ArucoMarkerScaledRegistration(object):
    def __init__(self, project_path_a: str, project_path_b: str, dense_pc: str = 'fused.ply'):
        # Name of both subprojects
        self.project_path_a = project_path_a
        self.project_path_b = project_path_b

        # Name of dense point cloud (incase it is cropped)
        self.dense_pc = dense_pc

        self.project_a: COLMAP = None
        self.project_b: COLMAP = None

        # Load COLMAP projects
        self.load_projects()

        # ArucoScaleFactor class for estimating scale factor
        self.aruco_scale_factor_a: ArucoScaleFactor = None
        self.aruco_scale_factor_b: ArucoScaleFactor = None

        # Sorted array with 3d location of all 4 aruco corners
        self.aruco_corners_3d_a: np.ndarray = None
        self.aruco_corners_3d_b: np.ndarray = None

        # Scaled point cloud
        self.pcd_a: o3d.geometry.PointCloud = None
        self.pcd_b: o3d.geometry.PointCloud = None
        self.pcd_combined: o3d.geometry.PointCloud = None

        # Scale factor for individual point clouds
        self.scale_factor_a: float = None
        self.scale_factor_b: float = None

        # Transformation matrix (4x4) to scale, rotate and translate pcd
        self.transformation_b2a: np.ndarray = None

    def load_projects(self):
        self.project_a = COLMAP(self.project_path_a, image_resize=0.3, dense_pc=self.dense_pc)
        self.project_b = COLMAP(self.project_path_b, image_resize=0.3, dense_pc=self.dense_pc)

    def scale(self, debug=False):
        self.aruco_scale_factor_a = ArucoScaleFactor(photogrammetry_software=self.project_a, aruco_size=0.3)
        self.aruco_scale_factor_b = ArucoScaleFactor(photogrammetry_software=self.project_b, aruco_size=0.3)

        aruco_distance_a, self.aruco_corners_3d_a = self.aruco_scale_factor_a.run()
        aruco_distance_b, self.aruco_corners_3d_b = self.aruco_scale_factor_b.run()

        self.pcd_a, self.scale_factor_a = self.aruco_scale_factor_a.apply()
        self.pcd_b, self.scale_factor_b = self.aruco_scale_factor_b.apply()

        if debug:
            # Visualization of the scene and rays
            vis = ArucoVisualization(aruco_colmap=self.aruco_scale_factor_a)
            vis.visualization(frustum_scale=0.3, point_size=0.1)

            # Visualization of the scene and rays
            vis = ArucoVisualization(aruco_colmap=self.aruco_scale_factor_b)
            vis.visualization(frustum_scale=0.3, point_size=0.1)

            o3d.visualization.draw_geometries([self.pcd_a, self.pcd_b])

    def registrate(self, additional_points: Union[type(None), tuple], manual=False, debug=False):

        if additional_points:
            self.aruco_corners_3d_a = np.vstack([self.aruco_corners_3d_a, additional_points[0]])
            self.aruco_corners_3d_b = np.vstack([self.aruco_corners_3d_b, additional_points[1]])

        if manual:
            manual_points_1, manual_points_2 = manual_registration(self.pcd_a, self.pcd_b)
            self.aruco_corners_3d_a = np.vstack([self.aruco_corners_3d_a, manual_points_1])
            self.aruco_corners_3d_b = np.vstack([self.aruco_corners_3d_b, manual_points_2])

        # Scale 3d aruco corners for alignment
        scaled_aruco_corners_a = self.scale_factor_a * self.aruco_corners_3d_a
        scaled_aruco_corners_b = self.scale_factor_b * self.aruco_corners_3d_b

        A, B, transformation_b2a = align_point_set(scaled_aruco_corners_a, scaled_aruco_corners_b)

        if debug:
            plot_aligned_pointset(A, B)

        # Init 4x4 transformation matrix
        self.transformation_b2a = np.eye(4)
        self.transformation_b2a[:3, :4] = np.hstack(
            [(transformation_b2a[1]) * transformation_b2a[0], np.expand_dims(transformation_b2a[2], axis=0).T])

        pcd_b_transformed = self.pcd_b.transform(self.transformation_b2a)
        self.pcd_combined = deepcopy(self.pcd_a)
        self.pcd_combined += pcd_b_transformed

        if debug:
            viewer = o3d.visualization.Visualizer()
            viewer.create_window(window_name='Combined PCD')

            viewer.add_geometry(self.pcd_combined)
            opt = viewer.get_render_option()
            # opt.show_coordinate_frame = True
            opt.point_size = 0.01
            opt.line_width = 0.01
            opt.background_color = np.asarray([1, 1, 1])
            viewer.run()
            viewer.destroy_window()

    def write(self):
        common_path = os.path.commonpath(([self.project_path_a, self.project_path_b]))
        # Save combined pcd and transformation from a to b
        o3d.io.write_point_cloud(os.path.join(common_path, './combined.ply'), self.pcd_combined)
        np.savetxt(os.path.join(common_path, 'transformation_b2a.txt'), self.transformation_b2a)

        self.aruco_scale_factor_a.write_data()
        self.aruco_scale_factor_b.write_data()


if __name__ == '__main__':
    scaled_registration = ArucoMarkerScaledRegistration(project_path_a="/home/luigi/Documents/reco/Baum 8/side_1",
                                                        project_path_b="/home/luigi/Documents/reco/Baum 8/side_2",
                                                        dense_pc='cropped.ply')
    scaled_registration.scale(debug=False)
    point_set = (np.asarray([-3.074686, -3.703092, 4.512500]), np.asarray([-4.271004, -4.733126, 3.378184])) # Baum 08
    # (np.asarray([-4.037381, -1.749546, 6.646245]), np.asarray([2.538995, -4.001166, 4.676914])) # Baum 07
    scaled_registration.registrate(additional_points=point_set, manual=False, debug=True)
    scaled_registration.write()
