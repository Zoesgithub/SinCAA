# -*- coding: utf-8 -*-
"""
@File   :  geomesh.py
@Time   :  2023/12/11 19:10
@Author :  Yufan Liu
@Desc   :  Geometric mesh 
"""

import numpy as np
import pymeshlab
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler


class GeoMesh(pymeshlab.MeshSet):
    def __init__(self, load=False, **kwargs):
        super().__init__()

        if not load:
            # define sth. into meta data
            if not "vertex_matrix" in kwargs or not "face_matrix" in kwargs:
                raise KeyError("not find essential keys vertex/face")

            self.original_mesh = pymeshlab.Mesh(
                vertex_matrix=kwargs["vertex_matrix"], face_matrix=kwargs["face_matrix"]
            )
            self.add_mesh(self.original_mesh)

            self.metadata = {}
            for key in kwargs:  # add features
                if key != "vertex_matrix" and key != "face_matrix":
                    self.metadata[key] = kwargs[key]
            self.feat_norm_flag = False

    @property
    def vertices(self):
        return self.current_mesh().vertex_matrix()

    @property
    def faces(self):
        return self.current_mesh().face_matrix()

    @property
    def vertex_normals(self):
        return self.current_mesh().vertex_normal_matrix()

    def set_attribute(self, key, value):
        # to metadata
        self.metadata[key] = value

    def get_attribute(self, key):
        return self.metadata[key]

    def normalize_features(self):
        # feature order: shapeindex, ddc, charge, logp, apbs
        # only normalize logp and apbs, others are in -1 and 1
        feature_dict = self.metadata
        assert "input_feature" in feature_dict, "Not all feature are processed!"
        # for logp
        # todo all in min-max
        logp = feature_dict["input_feature"][:, :, 3]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        logp = scaler.fit_transform(logp)

        # for apbs
        apbs = feature_dict["input_feature"][:, :, 4]
        apbs[apbs > 30] = 30
        apbs[apbs < -30] = -30
        apbs = scaler.fit_transform(apbs)

        feature_dict["input_feature"][:, :, 3] = logp
        feature_dict["input_feature"][:, :, 4] = apbs
        self.feat_norm_flag = True

    def save_feature(self, file):
        np.save(file, self.metadata["input_feature"])

    def load_feature(self, file):
        feat = np.load(file, allow_pickle=True).item()
        assert type(feat) == dict
        self.metadata = feat
