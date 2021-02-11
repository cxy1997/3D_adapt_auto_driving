import os
import argparse
import shutil
import numpy as np
import copy
import json
from itertools import islice, zip_longest
from PIL import Image
from itertools import chain

import multiprocessing as _mp
mp = _mp.get_context('spawn')

import sys
sys.path.insert(0, "..")
from config_path import dataset_path, datasets, dataset_paths
from utils.object_3d import Object3d, read_label
from utils.kitti_util import load_velo_scan, Calibration


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


car_sales_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "car_sales")

us_car_stats = load_json(os.path.join(car_sales_path, "us.json"))
germany_car_stats = load_json(os.path.join(car_sales_path, "germany.json"))
car_stats_external = {"kitti": germany_car_stats,
                      "argo_new": us_car_stats,
                      "nusc": us_car_stats,
                      "lyft": us_car_stats,
                      "waymo": us_car_stats,
}


def format_lidar_data(x, dst):
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float32)
    x = x.reshape(-1)
    x.tofile(dst)


def save_labels(labels, dst):
    labels = list(map(lambda x: x.to_kitti_format(), labels))
    with open(dst, "w") as f:
        f.write("\n".join(labels))


def single_scale(x, src, dst, ratio=1):
    return x + (dst["mean"] - src["mean"]) * ratio
    # return (x - src["mean"]) / src["std"] * dst["std"] + dst["mean"]


def get_scale_map(src, dst):
    return lambda x, ratio: (np.array([
        single_scale(x.l, src["length"], dst["length"], ratio),
        single_scale(x.h, src["height"], dst["height"], ratio),
        single_scale(x.w, src["width"], dst["width"], ratio),
    ]) / np.array([x.l, x.h, x.w])).reshape(1, 3)


def get_image_size(path):
    with open(os.path.join(path, "train.txt")) as f:
        sample_img_name = f.readlines()[0]
    sample_img = Image.open(os.path.join(path, "training", "image_2", f"{sample_img_name.rstrip()}.png"))
    return sample_img.size


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def gen_obj_box_ptc(obj):
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    x = obj.t[0]
    y = obj.t[1]
    z = obj.t[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])

    corners_3d = np.dot(R, corners)

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    return np.transpose(corners_3d)


def refine(obj, calib, w, h):
    box3d_pts_3d = gen_obj_box_ptc(obj)
    uv = calib.project_rect_to_image2(box3d_pts_3d)
    bbox = list(chain(np.min(uv, axis=0).tolist()[0:2], np.max(uv, axis=0).tolist()[0:2]))

    _bbox = [0] * 4
    _bbox[0] = max(0, bbox[0])
    _bbox[1] = max(0, bbox[1])
    _bbox[2] = min(w, bbox[2])
    _bbox[3] = min(h, bbox[3])

    obj.box2d = np.array(_bbox)
    return obj


def postprocessing(objs, w, h):
    _map = np.ones((h, w), dtype=np.uint8) * -1
    objs = sorted(objs, key=lambda x: x.t[2], reverse=True)
    for i, obj in enumerate(objs):
        _map[int(round(obj.box2d[1])):int(round(obj.box2d[3])), int(round(obj.box2d[0])):int(round(obj.box2d[2]))] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj.box2d[3] - obj.box2d[1]) / (obj.box2d[2] - obj.box2d[0])
        obj.trucation = int(np.clip(occlusion * 4, 0, 3))
    return objs


def regenerate_labels(objs, calib, w, h):
    for i in range(len(objs)):
        objs[i] = refine(objs[i], calib, w, h)
    return postprocessing(objs, w, h)


def scale_labels(objs, mapping, ratios, calib, w0, h0, align_front=False, rescaled_classes=("Car", "Van")):
    new_obj = []
    cnt = 0
    for obj in objs:
        _obj = copy.deepcopy(obj)
        if obj.cls_type in rescaled_classes:
            l, h, w = (np.array([obj.l, obj.h, obj.w]) * mapping(obj, ratios[cnt]).reshape(-1)).tolist()
            if align_front:
                dist = np.linalg.norm(obj.t)
                alpha = np.arctan2(np.sin(obj.alpha), np.cos(obj.alpha))
                if np.abs(np.sin(alpha)) * dist > obj.l / 2.0:
                    shift = (obj.l - l) / 2.0
                    if 0 < alpha:
                        angle = -obj.ry
                    else:
                        angle = -obj.ry + np.pi
                    _obj.t[0] += shift * np.cos(angle)
                    _obj.t[2] += shift * np.sin(angle)
                if np.abs(np.cos(alpha)) * dist > obj.w / 2.0:
                    shift = (obj.w - w) / 2.0
                    if -np.pi / 2.0 < alpha < np.pi / 2.0:
                        angle = -obj.ry - np.pi / 2.0
                    else:
                        angle = -obj.ry + np.pi / 2.0
                    _obj.t[0] += shift * np.cos(angle)
                    _obj.t[2] += shift * np.sin(angle)
            _obj.l, _obj.h, _obj.w = l, h, w
            cnt += 1
        new_obj.append(_obj)
    return regenerate_labels(new_obj, calib, w0, h0)


def rescale_ptc(mapping, velo, labels, calib, avoid_conflict=False, align_front=False, rescaled_classes=("Car", "Van")):
    ptc = calib.project_velo_to_rect(velo[:, :3])
    new_ptc = []
    mask = np.ones(ptc.shape[0]).astype(bool)
    ratios = []

    for obj in labels:
        if obj.cls_type in rescaled_classes:
            R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                          [0, 1, 0],
                          [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
            _ptc = np.dot(ptc - obj.t, R)
            _mask = (_ptc[:, 0] > -obj.l / 2.0) & (_ptc[:, 0] < obj.l / 2.0) & \
                    (_ptc[:, 1] > -obj.h) & (_ptc[:, 1] < 0) & \
                    (_ptc[:, 2] > -obj.w / 2.0) & (_ptc[:, 2] < obj.w / 2.0)
            ratio = 0
            _env_mask0 = (_ptc[:, 0] > -obj.l / 2.0) & (_ptc[:, 0] < obj.l / 2.0) & \
                         (_ptc[:, 1] > -obj.h) & (_ptc[:, 1] < -0.5) & \
                         (_ptc[:, 2] > -obj.w / 2.0) & (_ptc[:, 2] < obj.w / 2.0)
            if np.sum(_mask) > 0:
                mask[_mask] = False
                if avoid_conflict:
                    for ratio in np.arange(1, -0.1, -0.1):
                        tmp_ptc = _ptc[_mask] * mapping(obj, ratio)
                        _env_mask = (_ptc[:, 0] > np.min(tmp_ptc[:, 0])) & (_ptc[:, 0] < np.max(tmp_ptc[:, 0])) & \
                                    (_ptc[:, 1] > np.min(tmp_ptc[:, 1])) & (_ptc[:, 1] < -0.5) & \
                                    (_ptc[:, 2] > np.min(tmp_ptc[:, 2])) & (_ptc[:, 2] < np.max(tmp_ptc[:, 2]))
                        if np.sum(_env_mask) - np.sum(_env_mask0) < 10:
                            break
                else:
                    ratio = 1
                    tmp_ptc = _ptc[_mask] * mapping(obj, ratio)
                ptc_patch = np.dot(tmp_ptc, R.T) + obj.t

                if align_front:
                    l, h, w = (np.array([obj.l, obj.h, obj.w]) * mapping(obj, ratio).reshape(-1)).tolist()

                    dist = np.linalg.norm(obj.t)
                    alpha = np.arctan2(np.sin(obj.alpha), np.cos(obj.alpha))
                    if np.abs(np.sin(alpha)) * dist > obj.l / 2.0:
                        shift = (obj.l - l) / 2.0
                        if 0 < alpha:
                            angle = -obj.ry
                        else:
                            angle = -obj.ry + np.pi
                        ptc_patch[:, 0] += shift * np.cos(angle)
                        ptc_patch[:, 2] += shift * np.sin(angle)
                    if np.abs(np.cos(alpha)) * dist > obj.w / 2.0:
                        shift = (obj.w - w) / 2.0
                        if -np.pi / 2.0 < alpha < np.pi / 2.0:
                            angle = -obj.ry - np.pi / 2.0
                        else:
                            angle = -obj.ry + np.pi / 2.0
                        ptc_patch[:, 0] += shift * np.cos(angle)
                        ptc_patch[:, 2] += shift * np.sin(angle)

                new_ptc.append(ptc_patch)
            ratios.append(ratio)
    return calib.project_rect_to_velo(np.concatenate(new_ptc + [ptc[mask]], axis=0)), ratios


def convert(src, dst,
            spath=None,
            dpath=None,
            image_folder="image_2",
            calib_folder="calib",
            label_folder="label_2",
            use_car_sales_stats=False,
            avoid_conflict=False,
            align_front=False,
            rescaled_classes=("Car", "Van")):
    assert src in datasets and dst in datasets
    spath = spath or dataset_paths[src]

    if use_car_sales_stats:
        mapping = get_scale_map(car_stats_external[src], car_stats_external[dst])
    else:
        with open(os.path.join(dataset_paths[src], "label_stats_train.json")) as f:
            src_label_stats = json.load(f)
        with open(os.path.join(dataset_paths[dst], "label_stats_train.json")) as f:
            dst_label_stats = json.load(f)
        mapping = get_scale_map(src_label_stats, dst_label_stats)

    w, h = get_image_size(spath)

    dpath = dpath or os.path.join(dataset_path, "rescaled_datasets")
    root = os.path.join(dpath, f"{src}_scaledto_{dst}")
    os.makedirs(root, exist_ok=True)

    for split in ["train", "val", "trainval"]:
        shutil.copyfile(os.path.join(spath, f"{split}.txt"), os.path.join(root, f"{split}.txt"))

    root = os.path.join(root, "training")
    os.makedirs(root, exist_ok=True)
    # Link common files
    if os.path.exists(os.path.join(root, "image_2")):
        os.remove(os.path.join(root, "image_2"))
    os.symlink(os.path.join(spath, "training", image_folder), os.path.join(root, "image_2"))
    if os.path.exists(os.path.join(root, "calib")):
        os.remove(os.path.join(root, "calib"))
    if not os.path.islink(os.path.join(root, "calib")):
        os.symlink(os.path.join(spath, "training", calib_folder), os.path.join(root, "calib"))
    # Generate new pointclouds and labels
    os.makedirs(os.path.join(root, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(root, label_folder), exist_ok=True)

    with open(os.path.join(spath, "trainval.txt")) as f:
        names = list(map(lambda x: x.strip(), f.readlines()))
    n = len(names)

    for i, name in enumerate(names):
        ptc = load_velo_scan(os.path.join(spath, "training", "velodyne", f"{name}.bin"))
        calib = Calibration(os.path.join(spath, "training", calib_folder, f"{name}.txt"))
        labels = read_label(os.path.join(spath, "training", label_folder, f"{name}.txt"))
        labels = list(filter(lambda x: x.cls_type != "DontCare", labels))

        new_ptc, ratios = rescale_ptc(mapping, ptc, labels, calib, avoid_conflict=avoid_conflict,
                                      align_front=align_front, rescaled_classes=rescaled_classes)
        format_lidar_data(new_ptc, os.path.join(root, "velodyne", f"{name}.bin"))
        labels = scale_labels(labels, mapping, ratios, calib, w, h,
                              align_front=align_front, rescaled_classes=rescaled_classes)
        save_labels(labels, os.path.join(root, label_folder, f"{name}.txt"))


def launch_rescale(*args, **kwargs):
    processes = []
    for src in datasets:
        for dst in datasets:
            if src != dst:
                p = mp.Process(target=convert, args=(src, dst), kwargs=kwargs)
                p.start()
                processes.append(p)
    for p in processes:
        p.join()
        del p
    dpath = kwargs["dpath"] if "dpath" in kwargs.keys() else os.path.join(dataset_path, "rescaled_datasets")
    print(f"Rescaled datasets have been generated to {dpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", type=str, help="path to store converted datasets", default=os.path.join(dataset_path, "rescaled_datasets"))
    args = parser.parse_args()

    launch_rescale(dpath=args.path)
