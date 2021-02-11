from pathlib import Path
from typing import List, Optional

import os
import fire
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, transform_matrix, view_points
from lyft_dataset_sdk.utils.kitti import KittiDB
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

CLASS_MAP = {
    "other_vehicle": "Truck",
    "truck": "Truck",
    "car": "Car",
    "bus": "Truck",
    "emergency_vehicle": "Truck",
    "pedestrian": "Pedestrian",
    "motorcycle": "Misc",
    "bicycle": "Misc"
}


def box_to_string(name, box, bbox_2d, truncation, occlusion, alpha):
    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
    yaw = -np.arctan2(v[2], v[0])

    # Prepare output.
    name += ' '
    trunc = '{:.2f} '.format(truncation)
    occ = '{:d} '.format(occlusion)
    a = '{:.2f} '.format(alpha)
    bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
    hwl = '{:.2} {:.2f} {:.2f} '.format(box.wlh[2], box.wlh[0], box.wlh[1])  # height, width, length.
    xyz = '{:.2f} {:.2f} {:.2f} '.format(box.center[0], box.center[1], box.center[2])  # x, y, z.
    y = '{:.2f}'.format(yaw)  # Yaw angle.

    output = name + trunc + occ + a + bb + hwl + xyz + y

    return output


def postprocessing(objs, height, width):
    _map = np.ones((height, width), dtype=np.int8) * -1
    objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
    for i, obj in enumerate(objs):
        _map[int(round(obj["bbox_2d"][1])):int(round(obj["bbox_2d"][3])), int(round(obj["bbox_2d"][0])):int(round(obj["bbox_2d"][2]))] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj["bbox_2d"][3] - obj["bbox_2d"][1]) / (obj["bbox_2d"][2] - obj["bbox_2d"][0])
        obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))
    return objs


def project_to_2d(box, p_left, height, width):
    box = box.copy()

    # KITTI defines the box center as the bottom center of the object.
    # We use the true center, so we need to adjust half height in negative y direction.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Check that some corners are inside the image.
    corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))


    inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (0 <= bbox[0] < width and 0 < bbox[2] <= width)
    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (0 <= bbox[0] < width or 0 < bbox[2] <= width)
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
    return {"bbox": bbox, "truncated": truncated}


class KittiConverter:
    def __init__(self, store_dir: str = "/home/yw763/driving/lyft_v1.02-in-kitti-format-new"):
        """

        Args:
            store_dir: Where to write the KITTI-style annotations.
        """
        self.store_dir = Path(store_dir).expanduser()
        self.split_file = os.path.join(store_dir, "train.txt")
        if os.path.isfile(self.split_file):
            os.remove(self.split_file)
        self.store_dir = self.store_dir.joinpath("training")

        # Create store_dir.
        if not self.store_dir.is_dir():
            self.store_dir.mkdir(parents=True)

    def nuscenes_gt_to_kitti(
        self,
        lyft_dataroot: str = "/home/yw763/driving/lyft/v1.02-train",
        table_folder: str = "/home/yw763/driving/lyft/v1.02-train/v1.02-train",
        lidar_name: str = "LIDAR_TOP",
        get_all_detections: bool = True,
        parallel_n_jobs: int = 16,
        samples_count: Optional[int] = None,
    ) -> None:
        """Converts nuScenes GT formatted annotations to KITTI format.

        Args:
            lyft_dataroot: folder with tables (json files).
            table_folder: folder with tables (json files).
            lidar_name: Name of the lidar sensor.
                Only one lidar allowed at this moment.
            get_all_detections: If True, will write all
                bboxes in PointCloud and use only FrontCamera.
            parallel_n_jobs: Number of threads to parralel processing.
            samples_count: Number of samples to convert.

        """
        self.lyft_dataroot = lyft_dataroot
        self.table_folder = table_folder
        self.lidar_name = lidar_name
        self.get_all_detections = get_all_detections
        self.samples_count = samples_count
        self.parallel_n_jobs = parallel_n_jobs

        # Select subset of the data to look at.
        self.lyft_ds = LyftDataset(self.lyft_dataroot, self.table_folder)

        self.kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
        self.kitti_to_nu_lidar_inv = self.kitti_to_nu_lidar.inverse

        # Get assignment of scenes to splits.
        split_logs = [self.lyft_ds.get("log", scene["log_token"])["logfile"] for scene in self.lyft_ds.scene]
        if self.get_all_detections:
            self.cams_to_see = ["CAM_FRONT"]
        else:
            self.cams_to_see = [
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]

        # Create output folders.
        self.label_folder = self.store_dir.joinpath("label_2")
        self.calib_folder = self.store_dir.joinpath("calib")
        self.image_folder = self.store_dir.joinpath("image_2")
        self.lidar_folder = self.store_dir.joinpath("velodyne")
        for folder in [self.label_folder, self.calib_folder, self.image_folder, self.lidar_folder]:
            if not folder.is_dir():
                folder.mkdir(parents=True)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        if self.samples_count is not None:
            sample_tokens = sample_tokens[: self.samples_count]

        # print(len(sample_tokens))
        sample_tokens = sample_tokens

        self.tokens = sample_tokens

        with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
            Parallel()(delayed(self.process_token_to_kitti)(sample_token) for sample_token in tqdm(sample_tokens))

    def process_token_to_kitti(self, sample_token: str) -> None:
        rotate = False
        # Get sample data.
        sample = self.lyft_ds.get("sample", sample_token)
        sample_annotation_tokens = sample["anns"]

        norm = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]])
        norm_4 = np.eye(4)
        norm_4[:3, :3] = norm

        lidar_token = sample["data"][self.lidar_name]
        sd_record_lid = self.lyft_ds.get("sample_data", lidar_token)
        cs_record_lid = self.lyft_ds.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ego_record_lid = self.lyft_ds.get("ego_pose", sd_record_lid["ego_pose_token"])

        for idx, cam_name in enumerate(self.cams_to_see):
            cam_front_token = sample["data"][cam_name]
            if self.get_all_detections:
                token_to_write = sample_token
            else:
                token_to_write = cam_front_token
            sample_name = "%06d" % self.tokens.index(token_to_write)

            # Retrieve sensor records.
            sd_record_cam = self.lyft_ds.get("sample_data", cam_front_token)
            cs_record_cam = self.lyft_ds.get("calibrated_sensor", sd_record_cam["calibrated_sensor_token"])
            ego_record_cam = self.lyft_ds.get("ego_pose", sd_record_cam["ego_pose_token"])
            cam_height = sd_record_cam["height"]
            cam_width = sd_record_cam["width"]
            imsize = (cam_width, cam_height)

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(
                cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False
            )
            lid_ego_to_world = transform_matrix(
                ego_record_lid["translation"], Quaternion(ego_record_lid["rotation"]), inverse=False
            )
            world_to_cam_ego = transform_matrix(
                ego_record_cam["translation"], Quaternion(ego_record_cam["rotation"]), inverse=True
            )
            ego_to_cam = transform_matrix(
                cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]), inverse=True
            )
            velo_to_cam = np.dot(ego_to_cam, np.dot(world_to_cam_ego, np.dot(lid_ego_to_world, lid_to_ego)))

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, self.kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            # Cameras are always rectified.
            p_left_kitti[:3, :3] = cs_record_cam["camera_intrinsic"]

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            if self.lyft_ds.get("sensor", cs_record_cam["sensor_token"])["channel"] == "CAM_FRONT":
                expected_kitti_velo_to_cam_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                assert (velo_to_cam_rot.round(0) == expected_kitti_velo_to_cam_rot).all(), velo_to_cam_rot.round(0)

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will
            # include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam["filename"]
            filename_lid_full = sd_record_lid["filename"]

            # Convert image (jpg to png).
            src_im_path = self.lyft_ds.data_path.joinpath(filename_cam_full)
            dst_im_path = self.image_folder.joinpath(f"{sample_name}.png")
            if not dst_im_path.exists():
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = self.lyft_ds.data_path.joinpath(filename_lid_full)
            dst_lid_path = self.lidar_folder.joinpath(f"{sample_name}.bin")

            pcl = LidarPointCloud.from_file(Path(src_lid_path))
            # In KITTI lidar frame.
            pcl.rotate(self.kitti_to_nu_lidar_inv.rotation_matrix)
            if rotate:
                pcl = np.dot(norm_4, pcl.points).T.astype(np.float32).reshape(-1)
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            # tokens.append(token_to_write)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms["P0"] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms["P1"] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms["P2"] = p_left_kitti  # Left camera transform.
            kitti_transforms["P3"] = np.zeros((3, 4))  # Dummy values.
            # Cameras are already rectified.
            kitti_transforms["R0_rect"] = r0_rect.rotation_matrix
            if rotate:
                velo_to_cam_rot = np.dot(velo_to_cam_rot, norm.T)
            kitti_transforms["Tr_velo_to_cam"] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms["Tr_imu_to_velo"] = imu_to_velo_kitti
            calib_path = self.calib_folder.joinpath(f"{sample_name}.txt")

            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = "%.12e" % val[0]
                    for v in val[1:]:
                        val_str += " %.12e" % v
                    calib_file.write("%s: %s\n" % (key, val_str))

            # Write label file.
            label_path = self.label_folder.joinpath(f"{sample_name}.txt")
            if label_path.exists():
                # print("Skipping existing file: %s" % label_path)
                continue

            objects = []
            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = self.lyft_ds.get("sample_annotation", sample_annotation_token)

                # Get box in LIDAR frame.
                _, box_lidar_nusc, _ = self.lyft_ds.get_sample_data(
                    lidar_token, box_vis_level=BoxVisibility.NONE, selected_anntokens=[sample_annotation_token]
                )
                box_lidar_nusc = box_lidar_nusc[0]

                # Truncated: Set all objects to 0 which means untruncated.
                truncated = 0.0

                # Occluded: Set all objects to full visibility as this information is
                # not available in nuScenes.
                occluded = 0

                obj = dict()

                obj["detection_name"] = sample_annotation["category_name"]
                if obj["detection_name"] is None or obj["detection_name"] not in CLASS_MAP.keys():
                    continue

                obj["detection_name"] = CLASS_MAP[obj["detection_name"]]

                # Convert from nuScenes to KITTI box format.
                obj["box_cam_kitti"] = KittiDB.box_nuscenes_to_kitti(
                    box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect
                )

                # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                bbox_2d = project_to_2d(obj["box_cam_kitti"], p_left_kitti, cam_height, cam_width)
                if bbox_2d is None:
                    continue
                obj["bbox_2d"] = bbox_2d["bbox"]
                obj["truncated"] = bbox_2d["truncated"]

                # Set dummy score so we can use this file as result.
                obj["box_cam_kitti"].score = 0

                v = np.dot(obj["box_cam_kitti"].rotation_matrix, np.array([1, 0, 0]))
                rot_y = -np.arctan2(v[2], v[0])
                obj["alpha"] = -np.arctan2(obj["box_cam_kitti"].center[0], obj["box_cam_kitti"].center[2]) + rot_y
                obj["depth"] = np.linalg.norm(np.array(obj["box_cam_kitti"].center[:3]))
                objects.append(obj)

            objects = postprocessing(objects, cam_height, cam_width)

            with open(label_path, "w") as label_file:
                for obj in objects:
                    # Convert box to output string format.
                    output = box_to_string(name=obj["detection_name"], box=obj["box_cam_kitti"], bbox_2d=obj["bbox_2d"],
                                           truncation=obj["truncated"], occlusion=obj["occluded"], alpha=obj["alpha"])
                    label_file.write(output + '\n')
            with open(self.split_file, "a") as f:
                f.write(sample_name + "\n")

    def render_kitti(self, render_2d: bool = False) -> None:
        """Renders the annotations in the KITTI dataset from a lidar and a camera view.

        Args:
            render_2d: Whether to render 2d boxes (only works for camera data).

        Returns:

        """
        # if render_2d:
        #     print("Rendering 2d boxes from KITTI format")
        # else:
        #     print("Rendering 3d boxes projected from 3d KITTI format")

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.store_dir)

        # Create output folder.
        render_dir = self.store_dir.joinpath("render")
        if not render_dir.is_dir():
            render_dir.mkdir(parents=True)

        # Render each image.
        tokens = kitti.tokens

        # currently supports only single thread processing
        for token in tqdm(tokens):

            for sensor in ["lidar", "camera"]:
                out_path = render_dir.joinpath(f"{token}_{sensor}.png")
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                # Close the windows to avoid a warning of too many open windows.
                plt.close()

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """Convenience function to get the samples in a particular split.

        Args:
            split_logs: A list of the log names in this split.

        Returns: The list of samples.

        """
        samples = []
        str_to_write = ""
        for sample in self.lyft_ds.sample:
            scene = self.lyft_ds.get("scene", sample["scene_token"])
            log = self.lyft_ds.get("log", scene["log_token"])
            logfile = log["logfile"]
            if logfile in split_logs:
                samples.append(sample["token"])
            str_to_write += "{} {}\n".format(sample["token"], sample["scene_token"])
        with open(os.path.join(os.path.dirname(self.store_dir), "lyft_scenes.txt"), "w") as f:
            f.write(str_to_write)
        return samples


def lyft_to_kitti(lyft_path, kitti_path):
    kc = KittiConverter(store_dir=kitti_path)
    kc.nuscenes_gt_to_kitti(lyft_dataroot=os.path.join(lyft_path, "v1.02-train"),
                            table_folder=os.path.join(lyft_path, "v1.02-train", "v1.02-train"))
