from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST, get_image_dims_for_camera, STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT, RING_IMG_HEIGHT, RING_IMG_WIDTH
import argparse
import os
import shutil
from itertools import chain
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import logging
import copy
from collections import defaultdict
import pickle
logging.disable(logging.ERROR)

import numpy as np
from scipy.spatial.transform import Rotation

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import get_camera_extrinsic_matrix, point_cloud_to_homogeneous, proj_cam_to_uv, determine_valid_cam_coords
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.visualization.visualization_utils import show_image_with_boxes

import multiprocessing as _mp
mp = _mp.get_context('spawn')

from itertools import zip_longest

camL, camR = STEREO_CAMERA_LIST

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def inverse_rigid_trans(Tr):
    """
    Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    """

    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_camera_intrinsic_matrix(camera_config, bx, by):
    """
    Construct camera intrinsic matrix (including baselines) from Argoverse camera config dictionary.
    """

    intrinsic_matrix = np.zeros((3, 4))
    intrinsic_matrix[0, 0] = camera_config["focal_length_x_px_"]
    intrinsic_matrix[0, 1] = camera_config["skew_"]
    intrinsic_matrix[0, 2] = camera_config["focal_center_x_px_"]
    intrinsic_matrix[1, 1] = camera_config["focal_length_y_px_"]
    intrinsic_matrix[1, 2] = camera_config["focal_center_y_px_"]
    intrinsic_matrix[2, 2] = 1.0
    intrinsic_matrix[0, 3] = -intrinsic_matrix[0, 0] * bx
    intrinsic_matrix[1, 3] = -intrinsic_matrix[1, 1] * by
    return intrinsic_matrix


def print_line_separator():
    print("=" * int(os.popen("stty size", "r").read().split()[1]))


def build_kitti_path(kitti_root):
    kitti_path = dict()
    kitti_path["full_names"] = {"train": "training", "val": "training", "test": "testing"}

    for key, value in kitti_path["full_names"].items():
        kitti_path[key] = dict()
        kitti_path[key]["list"] = os.path.join(kitti_root, f"{key}.txt")
        kitti_path[key]["calib"] = os.path.join(kitti_root, value, "calib")
        kitti_path[key]["left"] = os.path.join(kitti_root, value, "image_2")
        kitti_path[key]["right"] = os.path.join(kitti_root, value, "image_3")
        for cam in RING_CAMERA_LIST:
            kitti_path[key][cam] = os.path.join(
                kitti_root, value, f"image_{cam}")
            kitti_path[key][f"calib_{cam}"] = os.path.join(
                kitti_root, value, f"calib_{cam}")
        if key != "test":
            kitti_path[key]["label"] = os.path.join(kitti_root, value, "label_2")
            kitti_path[key]["label_front"] = os.path.join(
                kitti_root, value, "label_front")
        kitti_path[key]["lidar"] = os.path.join(kitti_root, value, "velodyne")
        kitti_path[key]["pose"] = os.path.join(kitti_root, value, "oxts")

    return kitti_path


def extract_datapoints(root_dir, test_set=False):
    argoverse_loader = ArgoverseTrackingLoader(root_dir=root_dir)

    data = []
    for log_id in argoverse_loader.log_list:
        # print(f"Processing log: {os.path.join(root_dir, log_id)}", end="\r")
        argoverse_data = argoverse_loader.get(log_id=log_id)
        # calibL = argoverse_data.get_calibration(camera=camL, log_id=log_id)
        # calibR = argoverse_data.get_calibration(camera=camR, log_id=log_id)
        calibs = {}
        for cam in STEREO_CAMERA_LIST + RING_CAMERA_LIST:
            calibs[cam] = argoverse_data.get_calibration(camera=cam, log_id=log_id)

        count = 0
        for lidar_timestamp in argoverse_data.lidar_timestamp_list:
            data_point = dict()
            data_point["log_id"] = log_id
            data_point["frame_id"] = count
            count += 1
            for cam in STEREO_CAMERA_LIST + RING_CAMERA_LIST:
                cam_timestamp = argoverse_loader.sync.get_closest_cam_channel_timestamp(
                    lidar_timestamp=lidar_timestamp, camera_name=cam, log_id=log_id)
                if cam_timestamp is not None:
                    data_point[cam] = argoverse_loader.get_image_at_timestamp(
                        timestamp=cam_timestamp, camera=cam, log_id=log_id, load=False)
                else:
                    data_point[cam] = None
            data_point["timestamp"] = lidar_timestamp
            data_point["lidar"] = argoverse_loader.timestamp_lidar_dict[lidar_timestamp]
            data_point["calibs"] = calibs
            d = argoverse_data.get_pose(
                argoverse_data.get_idx_from_timestamp(lidar_timestamp)).translation
            r = Rotation.from_dcm(argoverse_data.get_pose(
                    argoverse_data.get_idx_from_timestamp(lidar_timestamp)).rotation)
            data_point["pose"] = (d, r.as_euler('xyz'))
            if not test_set:
                data_point["labels"] = argoverse_loader.get_label_object(
                    idx=argoverse_loader.lidar_timestamp_list.index(lidar_timestamp), log_id=log_id)
            data.append(data_point)
    return data


def read_argoverse(argo_path):
    data = dict()
    data["train"] = list(chain(*map(extract_datapoints, glob.glob(f"{argo_path}/train*/"))))
    data["val"] = extract_datapoints(os.path.join(argo_path, "val/"))
    data["test"] = extract_datapoints(os.path.join(argo_path, "test/"), test_set=True)
    print(f"Train data: {len(data['train'])}")
    print(f"Val data: {len(data['val'])}")
    print(f"Test data: {len(data['test'])}")
    return data


def convert_calib(calibL, calibR):
    imu = 'Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03'+\
        ' -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 '+\
        '3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01'
    R = 'R0_rect: '+' '.join([str(x) for x in np.eye(3).reshape(-1).tolist()])
    velo = calibL.extrinsic[:3,:]
    velo = 'Tr_velo_to_cam: '+' '.join([str(x) for x in velo.reshape(-1).tolist()])

    P2 = calibL.K
    # calibL.K = P2 = get_camera_intrinsic_matrix(calibL.calib_data['value'], 0, 0)
    P2 = ' '.join([str(x) for x in P2.reshape(-1).tolist()])

    P3 = calibR.K
    calibR.bx = 0.2986
    # calibR.K = P3 = get_camera_intrinsic_matrix(calibR.calib_data['value'], calibR.bx, 0)
    P3 = ' '.join([str(x) for x in P3.reshape(-1).tolist()])
    info = f"P0: {P2}\nP1: {P2}\nP2: {P2}\nP3: {P3}\n{R}\n{velo}\n{imu}\n"
    return info

def convert_calib_ring(calib):
    imu = 'Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03'+\
        ' -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 '+\
        '3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01'
    R = 'R0_rect: '+' '.join([str(x) for x in np.eye(3).reshape(-1).tolist()])
    velo = calib.extrinsic[:3, :]
    velo = 'Tr_velo_to_cam: '+' '.join([str(x) for x in velo.reshape(-1).tolist()])

    P2 = calib.K
    # calibL.K = P2 = get_camera_intrinsic_matrix(calibL.calib_data['value'], 0, 0)
    P2 = ' '.join([str(x) for x in P2.reshape(-1).tolist()])

    # P3 = calibR.K
    # calibR.bx = 0.2986
    # # calibR.K = P3 = get_camera_intrinsic_matrix(calibR.calib_data['value'], calibR.bx, 0)
    # P3 = ' '.join([str(x) for x in P3.reshape(-1).tolist()])
    info = f"P0: {P2}\nP1: {P2}\nP2: {P2}\nP3: {P2}\n{R}\n{velo}\n{imu}\n"
    return info


def to_euler(q):
    sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = +2.0 * (q[0] * q[2] - q[3] * q[1])
    if (np.abs(sinp) >= 1):
        pitch = np.sign(sinp) * np.pi * 0.5
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = +1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def format_lidar_data(src, dst):
    x = load_ply(src)
    # set the reflectance to 1.0 for every point
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    x = x.reshape(-1)
    x.tofile(dst)


CLASS_MAP = {
    "VEHICLE": "Car",
    "PEDESTRIAN": "Pedestrian",
    "LARGE_VEHICLE": "Truck",
    "BICYCLIST": "Cyclist",
    "BUS": "Truck",
    "TRAILER": "Truck",
    "MOTORCYCLIST": "Cyclist",
    "EMERGENCY_VEHICLE": "Van",
    "SCHOOL_BUS": "Truck"
}


def form_kitty_label(label, calib, is_stereo=True):
    if label.label_class not in CLASS_MAP.keys():
        return None
    box3d_pts_3d = label.as_3d_bbox()
    uv = calib.project_ego_to_image(box3d_pts_3d)
    uv_cam = calib.project_ego_to_cam(box3d_pts_3d)
    bbox = list(chain(np.min(uv, axis=0).tolist()[0:2], np.max(uv, axis=0).tolist()[0:2]))

    if is_stereo:
        inside = (0 <= bbox[1] < STEREO_IMG_HEIGHT and 0 < bbox[3] <= STEREO_IMG_HEIGHT) and (0 <= bbox[0] < STEREO_IMG_WIDTH and 0 < bbox[2] <= STEREO_IMG_WIDTH) and np.min(uv_cam[:, 2], axis=0) > 0
        valid = (0 <= bbox[1] < STEREO_IMG_HEIGHT or 0 < bbox[3] <= STEREO_IMG_HEIGHT) and (
            0 <= bbox[0] < STEREO_IMG_WIDTH or 0 < bbox[2] <= STEREO_IMG_WIDTH) and np.min(uv_cam[:, 2], axis=0) > 0 and label.translation[0] > 0
    else:
        inside = (0 <= bbox[1] < RING_IMG_HEIGHT and 0 < bbox[3] <= RING_IMG_HEIGHT) and (
            0 <= bbox[0] < RING_IMG_WIDTH and 0 < bbox[2] <= RING_IMG_WIDTH) and np.min(uv_cam[:, 2], axis=0) > 0
        valid = (0 <= bbox[1] < RING_IMG_HEIGHT or 0 < bbox[3] <= RING_IMG_HEIGHT) and (
            0 <= bbox[0] < RING_IMG_WIDTH or 0 < bbox[2] <= RING_IMG_WIDTH) and np.min(uv_cam[:, 2], axis=0) > 0 and label.translation[0] > 0
    if not valid:
        return None

    truncated = valid and not inside
    if truncated:
        _bbox = [0] * 4
        _bbox[0] = max(0, bbox[0])
        _bbox[1] = max(0, bbox[1])
        if is_stereo:
            _bbox[2] = min(STEREO_IMG_WIDTH, bbox[2])
            _bbox[3] = min(STEREO_IMG_HEIGHT, bbox[3])
        else:
            _bbox[2] = min(RING_IMG_WIDTH, bbox[2])
            _bbox[3] = min(RING_IMG_HEIGHT, bbox[3])

        truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        bbox = _bbox
    else:
        truncated = 0.0

    dcm_LiDAR = argoverse.utils.transform.quat2rotmat(label.quaternion)
    dcm_cam = calib.R.dot(dcm_LiDAR.dot(calib.R.T))
    rot_y = -np.pi * 0.5 + Rotation.from_dcm(dcm_cam).as_rotvec()[1]
    rot_y = np.arctan2(np.sin(rot_y), np.cos(rot_y))
    translation_cam = calib.project_ego_to_cam(label.translation.reshape(1, 3))
    alpha = -np.arctan2(translation_cam[0, 0], translation_cam[0, 2]) + rot_y

    obj = dict()
    obj["original_type"] = label.label_class
    obj["type"] = CLASS_MAP[label.label_class]
    obj["truncated"] = truncated
    obj["alpha"] = alpha
    obj["bbox"] = bbox
    obj["dimensions"] = [label.height, label.width, label.length]
    obj["location"] = calib.project_ego_to_cam(label.translation.reshape(1, 3)).reshape(-1)
    obj["location"][1] += label.height / 2.0
    obj["rotation_y"] = rot_y
    obj["depth"] = translation_cam[0, 2]
    obj["track_id"] = label.track_id
    return obj


def postprocessing(objs, is_stereo=True):
    if is_stereo:
        _map = np.ones((STEREO_IMG_HEIGHT, STEREO_IMG_WIDTH), dtype=np.uint8) * -1
    else:
        _map = np.ones((RING_IMG_HEIGHT, RING_IMG_WIDTH), dtype=np.uint8) * -1
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


def rectify_image(left_src, right_src, calibL, calibR, left_dst, right_dst):
    left_img, right_img = cv2.imread(left_src), cv2.imread(right_src)
    calibL, calibR = copy.deepcopy(calibL), copy.deepcopy(calibR)
    extrinsic = np.dot(calibR.extrinsic, np.linalg.inv(calibL.extrinsic))
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    distCoeff = np.zeros(4)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=calibL.K[:3, :3],
        distCoeffs1=distCoeff,
        cameraMatrix2=calibR.K[:3, :3],
        distCoeffs2=distCoeff,
        imageSize=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=calibL.K[:3, :3],
        distCoeffs=distCoeff,
        R=R1,
        newCameraMatrix=P1,
        size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        m1type=cv2.CV_32FC1)

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=calibR.K[:3, :3],
        distCoeffs=distCoeff,
        R=R2,
        newCameraMatrix=P2,
        size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        m1type=cv2.CV_32FC1)

    calibL.K = P1
    calibR.K = P2
    calibL.extrinsic[:3, :] = np.dot(R1, calibL.extrinsic[:3, :])
    calibR.extrinsic = calibL.extrinsic

    left_img_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    right_img_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    cv2.imwrite(left_dst, left_img_rect)
    cv2.imwrite(right_dst, right_img_rect)

    return calibL, calibR


def process(index, lst, path, signal, target):
    for name, dp in zip(index, lst):
        if name is None or dp is None:
            continue
        # print(f"converting {name}")

        for cam in RING_CAMERA_LIST:
            if dp[cam] is None:
                # print(f"skipping {name} {cam}")
                continue
            shutil.copyfile(dp[cam], os.path.join(
                path[cam], f"{name}.png"))

        if dp["stereo_front_left"] is not None and dp["stereo_front_right"] is not None:
            calibL, calibR = rectify_image(dp["stereo_front_left"],
                                        dp["stereo_front_right"],
                                        dp['calibs']["stereo_front_left"],
                                        dp['calibs']["stereo_front_right"],
                                        os.path.join(
                                                path["left"], f"{name}.png"),
                                                os.path.join(path["right"], f"{name}.png"))

            with open(os.path.join(path["calib"], f"{name}.txt"), "w") as calib_file:
                calib_file.write(convert_calib(calibL, calibR))
        # else:
            # print(f"skipping {name} stereo")

        for cam in RING_CAMERA_LIST:
            with open(os.path.join(path[f"calib_{cam}"], f"{name}.txt"), "w") as calib_file:
                calib_file.write(convert_calib_ring(dp["calibs"][cam]))

        format_lidar_data(dp["lidar"], os.path.join(path["lidar"], f"{name}.bin"))

        if "labels" in dp.keys():
            objs = map(lambda x: form_kitty_label(
                x, dp["calibs"]['ring_front_center'], is_stereo=False), copy.deepcopy(dp["labels"]))
            objs = list(filter(lambda x: x is not None, objs))
            objs = postprocessing(objs, is_stereo=False)
            save_label_file(objs, os.path.join(path["label_front"], f"{name}.txt"))

            if dp["stereo_front_left"] is not None and dp["stereo_front_right"] is not None:
                objs = map(lambda x: form_kitty_label(
                    x, calibL, is_stereo=True), copy.deepcopy(dp["labels"]))
                objs = list(filter(lambda x: x is not None, objs))
                objs = postprocessing(objs, is_stereo=True)
                save_label_file(objs, os.path.join(
                    path["label"], f"{name}.txt"))

        with open(os.path.join(path['pose'], f"{name}.txt"), "w") as f:
            f.write(
                " ".join([f"{num:.8f}" for num in np.concatenate((dp["pose"][0], dp["pose"][1]))]))

        signal.value += 1
        # print(f"Converted {signal.value} / {target}")


def format_data(data, path, start_idx, num_workers):
    target = len(data)
    index = list(map(lambda x: "%06d" % x, range(start_idx, start_idx + target, 1)))

    chunk_size = (len(data) - 1) // num_workers + 1
    processes = []
    signal = mp.Value('i', 0)

    for idx, lst in zip(grouper(chunk_size, index), grouper(chunk_size, data)):
        p = mp.Process(target=process,
                       args=(idx, lst, path, signal, target))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        del p
    del signal

    with open(path["list"], "w") as f:
        f.write("\n".join(index))


def argo_to_kitti(argo_path, kitti_path, worker=16, seed=19260817):
    argo_path = os.path.join(argo_path, "argoverse-tracking")
    np.random.seed(seed)

    # print_line_separator()
    print(f"Source Argoverse dataset: {argo_path}")
    print(f"Destination KITTI dataset: {kitti_path}")
    assert os.path.isdir(argo_path)
    if os.path.isdir(kitti_path):
        return
        # response = input("Delete original files? (Y/N): ")
        # if len(response) > 0 and response[0].lower() == "y":
        #     shutil.rmtree(kitti_path)
        #     print("Deleted existing destination directory.")

    data = read_argoverse(argo_path)

    # log_ids = set(chain(*map(lambda y: list(map(lambda x: x["log_id"], y)), data.values())))
    # log_ids_dict = dict(zip(log_ids, range(len(log_ids))))
    for v in data.values():
        np.random.shuffle(v)

    # print(f"Saving tracks_mapping....")
    # tracks_mapping = defaultdict(list)
    # for idx, datapoint in enumerate(data['train']):
    #     tracks_mapping[datapoint['log_id']].append((idx, datapoint['frame_id']))
    # # for key, value in tracks_mapping.items():
    # #     tracks_mapping[key] = [item[0] for item in sorted(tracks_mapping[key], key=lambda item: item[1])]
    # with open("train_mapping.pkl","wb") as f:
    #     pickle.dump(tracks_mapping, f)

    # print("Writing/Linking in KITTI format ...")
    kitti_path = build_kitti_path(kitti_root=kitti_path)
    for key in kitti_path["full_names"].keys():
        for path in kitti_path[key].values():
            if path.find('.') == -1:
                os.makedirs(path, exist_ok=True)

    # cnt = 0
    # print("Writing metadata ...")
    # with open(os.path.join(kitti_path, "meta_data.txt"), "w") as f:
    #     for key, value in data.items():
    #         for dp in value:
    #             f.write(f"{key}, data-id:{'%06d' % cnt}, log-id:{dp['log_id']}, log-simple-id:{log_ids_dict[dp['log_id']]}, frame-id:{dp['frame_id']}\n")
    #             cnt += 1

    print("Building train set ...")
    format_data(data=data["train"], path=kitti_path["train"], start_idx=0, num_workers=worker)
    print("Building val set ...")
    format_data(data=data["val"], path=kitti_path["val"], start_idx=len(data["train"]), num_workers=worker)
    print("Building test set ...")
    format_data(data=data["test"], path=kitti_path["test"], start_idx=0, num_workers=worker)

    os.system(f"cat {kitti_path}/train.txt {kitti_path}/val.txt > {kitti_path}/trainval.txt")
    print(f"Conversion complete. The new KITTI format dataset is in {kitti_path}")

    # with open("final_data.pkl", "wb") as f:
    #     pickle.dump(data, f)

