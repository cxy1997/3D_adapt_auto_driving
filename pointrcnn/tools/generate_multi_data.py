import os
import sys
sys.path.insert(0, "../..")
from config_path import dataset_paths


def gen_data(src, dst):
    os.makedirs(os.path.join(dst, "KITTI/object/training"), exist_ok=True)
    dst_path = os.path.join(dst, "KITTI/ImageSets")
    if not os.path.isdir(dst_path):
        os.symlink(src, dst_path)
    for sub_dir in ["image_2", "label_2", "velodyne", "calib", "planes"]:
        src_path = os.path.join(src, "training", sub_dir)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, "KITTI/object/training", sub_dir)
            if not os.path.isdir(dst_path):
                os.symlink(src_path, dst_path)


def gen_all_data():
    for dataset in dataset_paths.keys():
        print(f"generating multi_data folder for {dataset} ...")
        gen_data(dataset_paths[dataset], os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "multi_data", dataset))


if __name__ == "__main__":
    gen_all_data()
