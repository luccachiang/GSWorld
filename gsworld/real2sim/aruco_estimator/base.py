#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""


# Built-in/Generic Imports
# ...

# Libs
# ...

# Own modules
from colmap_wrapper.colmap import COLMAPProject

class ScaleFactorBase(object):
    def __init__(self, photogrammetry_software: COLMAPProject):
        """
        Base class for scale factor estimation.

            ---------------
            |    Detect   |
            ---------------
                    |
                    v
            ---------------
            |     Run     |
            ---------------
                    |
                    v
            ---------------
            |   Evaluate  |
            ---------------
                    |
                    v
            ---------------
            |     Apply   |
            ---------------
        """
        self.photogrammetry_software = photogrammetry_software.projects

    def __detect(self):
        return NotImplemented

    def __evaluate(self):
        return NotImplemented

    def get_dense_scaled(self):
        return NotImplemented

    def get_sparse_scaled(self):
        return NotImplemented

    def run(self):
        return NotImplemented

    def apply(self, *args, **kwargs):
        return NotImplemented

    def write_data(self):
        return NotImplemented