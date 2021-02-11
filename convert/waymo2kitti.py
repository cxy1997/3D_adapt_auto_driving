import os
import shutil
import argparse
from PIL import Image
import cv2
import tensorflow as tf
import math
import numpy as np
import glob
from itertools import chain
import multiprocessing as _mp
mp = _mp.get_context('spawn')
import time

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def print_line_separator():
    print("=" * int(os.popen("stty size", "r").read().split()[1]))


def build_kitti_path(kitti_root):
    kitti_path = dict()
    kitti_path["full_names"] = {"train": "training", "test": "testing"}

    for key, value in kitti_path["full_names"].items():
        kitti_path[key] = dict()
        kitti_path[key]["list"] = os.path.join(kitti_root, f"{key}.txt")
        kitti_path[key]["calib"] = os.path.join(kitti_root, value, "calib")
        kitti_path[key]["left"] = os.path.join(kitti_root, value, "image_2")
        # kitti_path[key]["right"] = os.path.join(kitti_root, value, "image_3")
        kitti_path[key]["label"] = os.path.join(kitti_root, value, "label_2")
        kitti_path[key]["lidar"] = os.path.join(kitti_root, value, "velodyne")

    # kitti_path["val"]["list"] = os.path.join(kitti_root, "trainval.txt")

    for key in kitti_path["full_names"].keys():
        for path in kitti_path[key].values():
            if path.find('.') == -1:
                os.makedirs(path, exist_ok=True)

    return kitti_path


def save_image(frame, dst):
    front_images = list(filter(lambda x: x.name == open_dataset.CameraName.Name.FRONT, frame.images))
    assert len(front_images) == 1
    front_image = Image.fromarray(tf.image.decode_jpeg(front_images[0].image).numpy())
    front_image.save(dst)


def save_pc(frame, dst):
    range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)
    points = np.concatenate(points + points_ri2, axis=0)
    # norm = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    # points = np.dot(points, norm)
    # set the reflectance to 1.0 for every point
    points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points = points.reshape(-1).astype(np.float32)
    points.tofile(dst)


def gen_obj_box_ptc(obj):
    R = rotz(-np.pi / 2 - obj.heading)

    # 3d bounding box dimensions
    l = obj.length
    w = obj.width
    h = obj.height

    x = obj.center_x
    y = obj.center_y
    z = obj.center_z

    # 3d bounding box corners
    y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners = np.vstack([x_corners, y_corners, z_corners])

    corners_3d = np.dot(R, corners)

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    return np.transpose(corners_3d)


def compute_extrinsic(calib):
    # Compute real extrinsic matrix
    extrinsic = np.reshape(np.array(calib.extrinsic.transform), [4, 4])
    extrinsic = tf.linalg.inv(extrinsic).numpy()
    norm = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    extrinsic[:3, 3] = extrinsic[:3, 3].reshape(1, 3).dot(norm)
    _norm = np.eye(4)
    _norm[:3, :3] = norm.T
    extrinsic = extrinsic.dot(_norm)
    return extrinsic


def project_from_ego_to_cam(pts_3d, extrinsic):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    uv_cam = extrinsic.dot(pts_3d_hom.transpose()).transpose()[:, 0:3]
    return uv_cam


def project_cam_to_image(intrinsic, points_rect):
    hom = np.hstack((points_rect, np.ones((points_rect.shape[0], 1))))
    pts_2d = np.dot(hom, np.transpose(intrinsic)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:, :2]


CLASS_MAP = {
    0: "UNKNOWN",
    1: "Car",
    2: "Pedestrian",
    3: "SIGN",
    4: "Cyclist"
}


def form_kitty_label(label, extrinsic, intrinsic, height, width):
    translation_ego = np.array([label.box.center_x, label.box.center_y, label.box.center_z]).reshape(1, 3)
    translation_cam = project_from_ego_to_cam(translation_ego, extrinsic)

    if translation_cam[0, 2] <= 0 or label.type in [0, 3] or np.abs(translation_cam[0, 0]) >= np.abs(translation_cam[0, 2]):
        return None

    pts_3d = gen_obj_box_ptc(label.box)
    uv_cam = project_from_ego_to_cam(pts_3d, extrinsic)
    uv = project_cam_to_image(intrinsic, uv_cam)

    bbox = list(chain(np.min(uv, axis=0).tolist()[0:2], np.max(uv, axis=0).tolist()[0:2]))

    inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (0 <= bbox[0] < width and 0 < bbox[2] <= width) and np.min(uv_cam[:, 2], axis=0) > 0
    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (0 <= bbox[0] < width or 0 < bbox[2] <= width) and np.min(uv_cam[:, 2], axis=0) > 0
    if not valid:
        return None

    truncated = valid and not inside
    if truncated:
        _bbox = [0] * 4
        _bbox[0] = max(0, bbox[0])
        _bbox[1] = max(0, bbox[1])
        _bbox[2] = min(width, bbox[2])
        _bbox[3] = min(height, bbox[3])

        truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        bbox = _bbox
    else:
        truncated = 0.0

    rot_y = - np.pi / 2.0 - label.box.heading
    rot_y = np.arctan2(np.sin(rot_y), np.cos(rot_y))

    alpha = -np.arctan2(translation_cam[0, 0], translation_cam[0, 2]) + rot_y

    obj = dict()
    obj["type"] = CLASS_MAP[label.type]
    obj["truncated"] = truncated
    obj["alpha"] = alpha
    obj["bbox"] = bbox
    obj["dimensions"] = [label.box.height, label.box.width, label.box.length]
    obj["location"] = translation_cam.reshape(-1)
    obj["location"][1] += label.box.height / 2.0
    obj["rotation_y"] = rot_y
    obj["depth"] = np.linalg.norm(translation_cam)
    return obj


def get_camera_intrinsic_matrix(intrinsic):
    # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
    intrinsic_matrix = np.zeros((3, 4))
    intrinsic_matrix[0, 0] = intrinsic[0]
    intrinsic_matrix[0, 1] = 0.0
    intrinsic_matrix[0, 2] = intrinsic[2]
    intrinsic_matrix[1, 1] = intrinsic[1]
    intrinsic_matrix[1, 2] = intrinsic[3]
    intrinsic_matrix[2, 2] = 1.0
    return intrinsic_matrix


def convert_calib(calib):
    imu = 'Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03'+\
        ' -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 '+\
        '3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01'
    R = 'R0_rect: '+' '.join([str(x) for x in np.eye(3).reshape(-1).tolist()])
    velo = compute_extrinsic(calib)[:3,:]
    velo = 'Tr_velo_to_cam: '+' '.join([str(x) for x in velo.reshape(-1).tolist()])

    K = get_camera_intrinsic_matrix(calib.intrinsic)
    K = ' '.join([str(x) for x in K.reshape(-1).tolist()])

    info = f"P0: {K}\nP1: {K}\nP2: {K}\nP3: {K}\n{R}\n{velo}\n{imu}\n"
    return info


def read_file(file, path, start_idx, signal, done, target):
    # print(f"Reading {file}, start_index={start_idx}")
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    try:
        signal.value = sum(1 for _ in dataset)
    except:
        signal.value = 0
        with open(os.path.join("/tmp", "waymo_missing.txt"), "a") as f:
            f.write(f"{file}\n")
        return

    target.value += signal.value

    idx = start_idx
    for data in dataset:
        dname = "%06d" % idx
        idx += 1

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Image
        save_image(frame, os.path.join(path["left"], f"{dname}.png"))

        # Point Cloud
        save_pc(frame, os.path.join(path["lidar"], f"{dname}.bin"))

        # Calib
        front_calibs = list(filter(lambda x: x.name == open_dataset.CameraName.Name.FRONT, frame.context.camera_calibrations))
        assert len(front_calibs) == 1
        front_calib = front_calibs[0]
        with open(os.path.join(path["calib"], f"{dname}.txt"), "w") as f:
            f.write(convert_calib(front_calib))

        # Label
        extrinsic = compute_extrinsic(front_calib)
        intrinsic = get_camera_intrinsic_matrix(front_calib.intrinsic)
        objs = map(lambda x: form_kitty_label(x, extrinsic, intrinsic, front_calib.height, front_calib.width), frame.laser_labels)
        objs = list(filter(lambda x: x is not None, objs))
        objs = postprocessing(objs, front_calib.height, front_calib.width)
        save_label_file(objs, os.path.join(path["label"], f"{dname}.txt"))

        done.value += 1
        # print(f"Converted {done.value} / {target.value}")


def postprocessing(objs, height, width):
    _map = np.ones((height, width), dtype=np.uint8) * -1
    objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
    for i, obj in enumerate(objs):
        _map[round(obj["bbox"][1]):round(obj["bbox"][3]), round(obj["bbox"][0]):round(obj["bbox"][2])] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj["bbox"][3] - obj["bbox"][1]) / (obj["bbox"][2] - obj["bbox"][0])
        obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))
    return objs


def save_label_file(objs, path):
    labels = []
    for obj in objs:
        string_to_write = f"{obj['type']} {'%.2f' % obj['truncated']} {obj['occluded']} {'%.2f' % obj['alpha']} "
        string_to_write += " ".join(map(lambda x: "%.2f" % x, obj["bbox"])) + " "
        string_to_write += " ".join(map(lambda x: "%.2f" % x, obj["dimensions"])) + " "
        string_to_write += " ".join(map(lambda x: "%.2f" % x, obj["location"])) + " "
        string_to_write += "%0.2f" % obj["rotation_y"]
        labels.append(string_to_write)

    with open(path, "w") as f:
        f.write("\n".join(labels))


def waymo_to_kitti(waymo_path, kitti_path, seed=19260817):
    np.random.seed(seed)

    # print_line_separator()
    print(f"Source Waymo dataset: {waymo_path}")
    print(f"Destination KITTI dataset: {kitti_path}")
    assert os.path.isdir(waymo_path)
    if os.path.isdir(kitti_path):
        return
        # print("Delete existing folder? (Y/N)")
        # response = input()
        # if response[0].lower() == "y":
        #     shutil.rmtree(kitti_path)
        #     print("Deleted existing destination directory.")
    kitti_path = build_kitti_path(kitti_root=kitti_path)
    # print_line_separator()

    start_idx = 0
    signal = mp.Value('i', 0)
    done = mp.Value('i', 0)
    target = mp.Value('i', 0)
    processes = []

    start_idx = 0
    files = glob.glob(os.path.join(waymo_path, "training", "*.tfrecord"))

    str_to_write = ""
    for file in files:
        signal.value = -1
        p = mp.Process(target=read_file,
                       args=(file, kitti_path["train"], start_idx, signal, done, target))
        p.start()
        processes.append(p)
        while signal.value == -1:
            time.sleep(1)
        str_to_write += "%d " % start_idx
        start_idx += signal.value
        str_to_write += "%d\n" % start_idx
    # with open("waymo_train.txt", "w") as f:
    #     f.write(str_to_write)

    files = glob.glob(os.path.join(waymo_path, "testing", "*.tfrecord"))
    for file in files:
        signal.value = -1
        p = mp.Process(target=read_file,
                       args=(file, kitti_path["test"], start_idx, signal, done, target))
        p.start()
        processes.append(p)
        while signal.value == -1:
            time.sleep(1)
        start_idx += signal.value

    for p in processes:
        p.join()
        del p
