import argparse
import sys
sys.path.insert(0, "..")
from config_path import raw_path_dic as path_dic
import convert
import multiprocessing as _mp
mp = _mp.get_context('spawn')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--datasets", type=str, help="datasets to be converted", default="argo+nusc+lyft+waymo")
    args = parser.parse_args()

    processes = []
    for dataset in args.datasets.split("+"):
        p = mp.Process(target=convert.__dict__[f"{dataset}_to_kitti"],
                       args=(path_dic[dataset], path_dic[f"{dataset}-in-kitti-format"]))
        p.start()
        processes.append(p)
        print(f"starting to convert {dataset} to KITTI format ...")

    for p in processes:
        p.join()
        del p
