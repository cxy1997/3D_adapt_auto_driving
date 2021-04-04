import sys
sys.path.insert(0, "..")
import os
import numpy as np
from config_path import dataset_paths


def is_valid_car(x):
    if len(x) < 0 or x[0] != "Car":
        return False
    height = float(x[7]) - float(x[5]) + 1
    truncated = float(x[1])
    occluded = float(x[2])
    return height >= 25 and truncated <= 0.5 and occluded <= 2


def has_car(label_filename):
    with open(label_filename) as f:
        lines = list(filter(is_valid_car, map(lambda x: x.strip().split(" "), f.readlines())))
    return len(lines) > 0


path_dic = {"train": "training", "val": "training"}


if __name__ == '__main__':
    np.random.seed(19260817)

    for dataset, path in dataset_paths.items():
        for key, value in path_dic.items():
            with open(os.path.join(path, f"{key}.txt"), 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
            file_names = list(filter(lambda x: has_car(os.path.join(path, value, "label_2", f"{x}.txt")), file_names))
            np.random.shuffle(file_names)
            with open(os.path.join(path, f"{key}_car1.txt"), 'w') as f:
                f.write("\n".join(file_names))

