#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Tuple

# Libs
import numpy as np


# Own modules
# ...


def intersect(P0: np.ndarray, N: np.ndarray, solve='pseudo') -> np.ndarray:
    '''
    Least Squares Intersection of Lines: https://silo.tips/download/least-squares-intersection-of-lines#modals

    :param P0:
    :param n:
    :param solve:
    :return:
    '''
    # generate the array of all projectors
    projs = np.eye(N.shape[1]) - N[:, :, np.newaxis] * N[:, np.newaxis]  # I - n*n.T

    # Sum over all K lines to get R
    R = projs.sum(axis=0)

    # Sum over all K lines to get q
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)

    # Solve LS for Rp = q to find feasible p
    if solve == 'ls':
        p = np.linalg.lstsq(R, q, rcond=None)[0]
    # P_hat = R_pseudo @ q
    elif solve == 'pseudo':
        p = np.linalg.pinv(R) @ q
    else:
        return NotImplementedError

    return p


def ls_intersection_of_lines(P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    p1 = intersect(P0, N[:, 0])
    p2 = intersect(P0, N[:, 1])
    p3 = intersect(P0, N[:, 2])
    p4 = intersect(P0, N[:, 3])

    intersctions_3d = np.concatenate([p1, p2, p3, p4], axis=1).T

    return intersctions_3d


def intersect_parallelized(P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Here the intersection of the rays are calculated. More info can be found in the method directly.
    The original resource for the underlying LS can be found here:
    https://silo.tips/download/least-squares-intersection-of-lines (Sorry this website is quite shady)

    :param P0:
    :param n:
    :return:
    """
    # generate the array of all projectors. Shape (#Corners, #Lines, Dim, Dim)
    I = np.zeros((N.shape[1], N.shape[0], N.shape[2], N.shape[2])) + np.eye(N.shape[2])
    # I - n*n.T
    projs = I - N.transpose(1, 0, 2)[..., np.newaxis] * N.transpose(1, 0, 2)[:, :, np.newaxis]

    # Sum over all K lines to get R
    R = projs.sum(axis=1)

    # Sum over all K lines to get q
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=1)

    # P_hat = R_pseudo @ q
    p = np.linalg.pinv(R) @ q

    return p[..., 0]


def ls_intersection_of_lines_parallelized(P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    intersctions_3d = intersect_parallelized(P0, N)

    return intersctions_3d[..., 0]


if __name__ == '__main__':
    P0 = np.array([[-0.66467896, -0.12612974, 0.18842651],
                   [2.0592406, -0.07465627, -1.96427852],
                   [-3.16910332, -0.15764522, 0.7080943],
                   [2.09202868, 0.84604156, -1.07464186],
                   [-0.81283296, -0.15276553, 0.78820854],
                   [-0.53366444, 0.39663029, 3.86951385],
                   [-0.5151707, -1.21208185, 4.2276595],
                   [-3.15752241, 0.03126854, 2.83419009],
                   [0.12736155, 0.12046079, -4.30550161],
                   [2.92592488, 0.20218443, -5.10770593],
                   [-4.21380129, -2.06626309, 0.59953318],
                   [-0.42358945, 0.68561167, -3.82537768],
                   [1.51545511, 0.18264608, 2.97991268],
                   [2.76166751, 0.4460384, -5.1150681],
                   [0.18052909, 0.05932384, 2.06230317]])

    N = np.array([[[0.54419107, -0.02476019, 0.83859586],
                   [0.63746768, -0.0236095, 0.77011529],
                   [0.63337776, 0.09169506, 0.76839093],
                   [0.54055894, 0.093205, 0.8361273]],
                  [[0.00394709, -0.02735305, 0.99961804],
                   [0.09212035, -0.02801828, 0.99535361],
                   [0.09327727, 0.06896309, 0.99324893],
                   [0.00646593, 0.06578877, 0.99781262]],
                  [[0.81624627, -0.01466789, 0.57751786],
                   [0.85860798, -0.01343522, 0.51245666],
                   [0.85466652, 0.07482142, 0.51375762],
                   [0.81177823, 0.07791933, 0.57874406]],
                  [[-0.00153974, -0.19557576, 0.98068739],
                   [0.09972525, -0.20317522, 0.97405067],
                   [0.10299916, -0.09251289, 0.9903699],
                   [0.00132274, -0.08931349, 0.99600269]],
                  [[0.62279618, -0.02139635, 0.7820915],
                   [0.71332764, -0.01977434, 0.70055167],
                   [0.70754649, 0.10374689, 0.69900969],
                   [0.61739984, 0.10668682, 0.77938139]],
                  [[0.95071483, -0.23560843, 0.2015688],
                   [0.97571718, -0.19858252, 0.09241739],
                   [0.9945175, -0.01371325, 0.10366722],
                   [0.97528938, -0.01766627, 0.22022383]],
                  [[0.93573828, 0.34548816, 0.07093517],
                   [0.95600904, 0.29279806, -0.01777671],
                   [0.89612633, 0.44373804, -0.00735896],
                   [0.85612904, 0.51089219, 0.07766744]],
                  [[0.9556413, -0.05168605, 0.2899625],
                   [0.9733264, -0.0471822, 0.22452072],
                   [0.97192197, 0.05355593, 0.22912756],
                   [0.95356379, 0.05715282, 0.29571887]],
                  [[0.21848919, -0.04159272, 0.97495257],
                   [0.28282464, -0.0415682, 0.95827048],
                   [0.28310061, 0.02610105, 0.95873499],
                   [0.21924028, 0.02526849, 0.97534363]],
                  [[-0.08802003, -0.047209, 0.99499939],
                   [-0.03214429, -0.04852336, 0.99830468],
                   [-0.03083776, 0.01595178, 0.9993971],
                   [-0.08604142, 0.01524213, 0.99617496]],
                  [[0.8298362, 0.23926937, 0.50410519],
                   [0.86299886, 0.22923912, 0.45020261],
                   [0.84495217, 0.29812405, 0.44404716],
                   [0.81091626, 0.31010282, 0.4962369]],
                  [[0.28928952, -0.10819229, 0.95110778],
                   [0.35441585, -0.10878594, 0.9287384],
                   [0.35651083, -0.03910737, 0.93347236],
                   [0.29110732, -0.0390253, 0.95589411]],
                  [[0.35247267, -0.26997514, 0.89603373],
                   [0.66112726, -0.25578734, 0.70532516],
                   [0.67464766, 0.1026629, 0.7309657],
                   [0.36337418, 0.10496366, 0.92571153]],
                  [[-0.07048161, -0.07272375, 0.99485858],
                   [-0.01450195, -0.07474184, 0.99709746],
                   [-0.01306351, -0.01022331, 0.9998624],
                   [-0.06890242, -0.01026508, 0.99757059]],
                  [[0.62422058, -0.10213532, 0.77454312],
                   [0.75479275, -0.09486844, 0.64906694],
                   [0.75196432, 0.08965499, 0.65307859],
                   [0.62116283, 0.09398982, 0.77802484]]])

    true_value = np.array([[2.08410173, -0.25158843, 4.42470409],
                           [2.6280005, -0.24844132, 4.16826468],
                           [2.64014018, 0.35195336, 4.20018012],
                           [2.09960334, 0.34874095, 4.46436423]])

    ret = ls_intersection_of_lines(P0, N)
    np.testing.assert_array_almost_equal(ret, true_value)

    ret = ls_intersection_of_lines_parallelized(P0, N)
    np.testing.assert_array_almost_equal(ret, true_value)
