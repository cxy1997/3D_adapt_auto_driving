import argparse
import sys
sys.path.insert(0, "..")
from config_path import raw_path_dic as path_dic
import download
import multiprocessing as _mp
mp = _mp.get_context('spawn')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--datasets", type=str, help="datasets to be downloaded", default="kitti+argo+waymo")
    args = parser.parse_args()

    processes = []
    for dataset in args.datasets.split("+"):
        p = mp.Process(target=download.__dict__[f"download_{dataset}"],
                       args=(path_dic[dataset],))
        p.start()
        processes.append(p)
        print(f"starting to download {dataset} ...")

    for p in processes:
        p.join()
        del p
