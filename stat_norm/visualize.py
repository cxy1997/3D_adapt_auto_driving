import os
import numpy as np

import sys
sys.path.insert(0, "..")
from config_path import dataset_path, datasets, dataset_paths
from utils.plotly_utils import showvelo2
from utils.object_3d import Object3d, read_label
from utils.kitti_util import load_velo_scan, Calibration
import pandas as pd


def get_object_mask(velo, calib, labels, rescaled_classes=("Car", "Van")):
    mask = np.zeros(velo.shape[0]).astype(bool)
    velo = calib.project_velo_to_rect(velo[:, :3])

    for obj in labels:
        if obj.cls_type in rescaled_classes:
            R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                          [0, 1, 0],
                          [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
            _ptc = np.dot(velo - obj.t, R)
            _mask = (_ptc[:, 0] > -obj.l / 2.0) & (_ptc[:, 0] < obj.l / 2.0) & \
                    (_ptc[:, 1] > -obj.h) & (_ptc[:, 1] < 0) & \
                    (_ptc[:, 2] > -obj.w / 2.0) & (_ptc[:, 2] < obj.w / 2.0)
            if np.sum(_mask) > 0:
                mask[_mask] = True
    return mask


def compare_stat_norm(src, dst, dataid=1, rescaled_classes=("Car", "Van"), path=None):
    assert src in datasets and dst in datasets
    path = path or os.path.join(dataset_path, "rescaled_datasets")
    velo_src = load_velo_scan(os.path.join(dataset_paths[src], "training", "velodyne", f"{dataid:06d}.bin"))
    velo_dst = load_velo_scan(os.path.join(path, f"{src}_scaledto_{dst}", "training", "velodyne", f"{dataid:06d}.bin"))
    calib = Calibration(os.path.join(dataset_paths[src], "training", "calib", f"{dataid:06d}.txt"))
    label_src = read_label(os.path.join(dataset_paths[src], "training", "label_2", f"{dataid:06d}.txt"))
    label_dst = read_label(os.path.join(path, f"{src}_scaledto_{dst}", "training", "label_2", f"{dataid:06d}.txt"))

    obj_mask_src = get_object_mask(velo_src, calib, label_src, rescaled_classes=rescaled_classes)
    obj_mask_dst = get_object_mask(velo_dst, calib, label_dst, rescaled_classes=rescaled_classes)
    env_mask_src = np.logical_not(obj_mask_src)

    showvelo2(lidar_common=velo_src[env_mask_src],
              lidar_before=velo_src[obj_mask_src],
              lidar_after=velo_dst[obj_mask_dst],
              calib=calib,
              labels_before=label_src,
              labels_after=label_dst,
              classes=rescaled_classes
             )
