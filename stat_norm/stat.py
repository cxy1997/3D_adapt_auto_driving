import os
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import chain

import sys
sys.path.insert(0, "..")
from config_path import datasets, dataset_full_name, dataset_paths
from utils.object_3d import read_label


split_path_dic = {"train": "training",
                  "val": "training",
                  "test": "testing"}
stat_subjects = ["height", "width", "length"]


def get_stats(data):
    return {"mean": float(np.mean(data)),
            "std": float(np.std(data))}


def get_dataset_stats(root, split="train", force=False):
    assert split in split_path_dic.keys()
    stat_file = os.path.join(root, f"label_stats_{split}.json")

    if os.path.isfile(stat_file) and not force:
        with open(stat_file, "r") as f:
            stats = json.load(f)
    else:
        split_file = os.path.join(root, f"{split}.txt")
        with open(split_file, 'r') as f:
            data_ids = [x.strip() for x in f.readlines()]
        label_dir = os.path.join(root, split_path_dic[split], "label_2")

        stats = {x: [] for x in stat_subjects}
        for data_id in data_ids:
            objects = read_label(os.path.join(label_dir, "%s.txt" % data_id))
            for obj in objects:
                if obj.cls_type == "Car":
                    stats["height"].append(obj.h)
                    stats["width"].append(obj.w)
                    stats["length"].append(obj.l)

        for x in stat_subjects:
            stats[x] = get_stats(np.array(stats[x]))

        with open(stat_file, "w") as f:
            json.dump(stats, f, indent=4)

    return stats


def plot_stats(dataset_stats):
    ax = plt.subplot(1, 1, 1)
    table = ax.table(cellText=[[f"${dataset_stats[d][x]['mean']:0.2f}\pm{dataset_stats[d][x]['std']:0.2f}$" for x in stat_subjects] for d in datasets],
              rowLabels=[dataset_full_name[d] for d in datasets],
              bbox=[0, 0, 1, 1],
              colLabels=stat_subjects)
    ax.axis("off")
    table.set_fontsize(20)


def print_stats(dataset_stats):
    lines = [["mean (std)"] + stat_subjects]
    for d, stat in dataset_stats.items():
        lines.append([dataset_full_name[d]] + [f"{stat[x]['mean']:0.2f} ({stat[x]['std']:0.2f})" for x in stat_subjects])
    max_len = max(map(len, chain(*lines)))
    for line in lines:
        print("|".join([f"{s:{max_len}}" for s in line]))


if __name__ == "__main__":
    stats = {d: get_dataset_stats(dataset_paths[d]) for d in datasets}
    print_stats(stats)
