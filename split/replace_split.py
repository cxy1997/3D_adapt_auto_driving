import os
import shutil


datasets = ["kitti", "argo", "nusc", "lyft", "waymo"]
splits = ["train", "val"]


def replace_split(path_dic):
    """
    Replace all the splits files of 5 datasets with ours.
    The original ones will be renamed.
    """

    source_dir = os.path.dirname(os.path.realpath(__file__))
    for dataset in datasets:
        src = os.path.join(source_dir, dataset)
        dst = path_dic[dataset if dataset == "kitti" else dataset + "-in-kitti-format"]
        for split in splits:
            dst_file = os.path.join(dst, f"{split}.txt")
            dst_original = os.path.join(dst, f"{split}_original.txt")
            if os.path.isfile(dst_file) and not os.path.isfile(dst_original):
                os.rename(dst_file, dst_original)
            shutil.copyfile(os.path.join(src, f"{split}.txt"), dst_file)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config_path import raw_path_dic as path_dic

    replace_split(path_dic)
