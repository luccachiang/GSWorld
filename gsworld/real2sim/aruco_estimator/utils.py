#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import numpy as np
import open3d as o3d
from copy import copy


def kabsch_umeyama(pointset_A, pointset_B):
    """
    Kabsch–Umeyama algorithm is a method for aligning and comparing the similarity between two sets of points.
    It finds the optimal translation, rotation and scaling by minimizing the root-mean-square deviation (RMSD)
    of the point pairs.

    Source and Explenation: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    @param pointset_A: array of a set of points in n-dim
    @param pointset_B: array of a set of points in n-dim
    @return: Rotation Matrix (3x3), scaling (scalar) translation vector (3x1)
    """
    assert pointset_A.shape == pointset_B.shape
    n, m = pointset_A.shape

    # Find centroids of both point sets
    EA = np.mean(pointset_A, axis=0)
    EB = np.mean(pointset_B, axis=0)

    VarA = np.mean(np.linalg.norm(pointset_A - EA, axis=1) ** 2)

    # Covariance matrix
    H = ((pointset_A - EA).T @ (pointset_B - EB)) / n

    # SVD H = UDV^T
    U, D, VT = np.linalg.svd(H)

    # Detect and prevent reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # rotation, scaling and translation
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def align_point_set(point_set_A, point_set_B):
    R, c, t = kabsch_umeyama(np.asarray(point_set_A), np.asarray(point_set_B))

    point_set_B = np.array([t + c * R @ b for b in point_set_B])

    return point_set_A, point_set_B, [R, c, t]


def plot_aligned_pointset(A, B):
    """
    Visualize transformed point set
    @param A: array of a set of points in n-dim
    @param B: array of a set of points in n-dim
    @return: both point clouds
    """


    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(A)

    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(B)


    o3d.visualization.draw_geometries([pcdA, pcdB])

    return pcdA, pcdB


def get_icp_transformation(source, target, trafo, max_iteration=2000):
    threshold = 0.02
    trans_init = np.eye(4)
    trans_init[:3, :4] = np.hstack([trafo[1] * trafo[0], np.expand_dims(trafo[2], axis=0).T])

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p


def manual_registration(pcd_1, pcd_2):
    """
    Source: http://www.open3d.org/docs/latest/tutorial/Advanced/interactive_visualization.html

    @param pcd_1:
    @param pcd_2:
    @return:
    """

    def pick_points(pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        viewer = o3d.visualization.VisualizerWithEditing()
        viewer.create_window(window_name='Picker')
        opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        opt.point_size = 2.5
        viewer.add_geometry(pcd)
        viewer.run()  # user picks points
        viewer.destroy_window()
        print("")
        return viewer.get_picked_points()

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(pcd_1)
    picked_id_target = pick_points(pcd_2)

    picked_points_1 = pcd_1.select_by_index(picked_id_source)
    picked_points_2 = pcd_1.select_by_index(picked_id_target)

    return np.asarray(picked_points_1.points), np.asarray(picked_points_2.points)
