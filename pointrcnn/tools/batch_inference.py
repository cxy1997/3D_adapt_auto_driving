import os
import sys
import argparse
import shutil
from datetime import datetime
import torch
from itertools import islice, zip_longest

sys.path.insert(0, "../..")
from config_path import datasets, dataset_paths

import multiprocessing as _mp
mp = _mp.get_context('spawn')

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def gen_cmd(model, data, name, args, far_points):
    batch_size = 8
    if args.cfg == 'double':
        batch_size = 4
    dataset_name = "argo" if data.startswith("argo") else data
    cmd = f"python eval_rcnn.py --cfg_file cfgs/{args.cfg}.yaml --ckpt {model} --batch_size {batch_size} --eval_mode rcnn  --root ../multi_data/{data}/ --output_dir ../output/rcnn/{name}/ --within {args.within} --far-points {far_points} --dataset-name {dataset_name}"
    if args.inference:
        cmd += " --inference"
    if args.toground:
        cmd += " --toground"
    if args.align_size:
        cmd += " --align-size"
    if args.reverse_align:
        cmd += " --reverse-align"
    if args.noiou:
        cmd += " --noiou"
    return cmd

def gen_data(name):
    path = data_paths[name]["origin"]
    velodyne = data_paths[name]["path"]

    os.makedirs(f"/home/yw763/Deeplearning/Deeplearning/pointrcnn/multi_data/{name}/KITTI/object/training", exist_ok=True)

    assert os.path.isdir(path), f"{path} does not exist!"
    dst_path = f"/home/yw763/Deeplearning/Deeplearning/pointrcnn/multi_data/{name}/KITTI/ImageSets"
    if not os.path.isdir(dst_path):
        os.symlink(path, dst_path)

    for sub_dir in ["image_2", "label_2", "calib", "planes"]:
        src_path = os.path.join(path, "training", sub_dir)
        dst_path = f"/home/yw763/Deeplearning/Deeplearning/pointrcnn/multi_data/{name}/KITTI/object/training/{sub_dir}"
        assert os.path.isdir(src_path), f"{src_path} does not exist!"
        if not os.path.isdir(dst_path ):
            os.symlink(src_path, dst_path)

    assert os.path.isdir(velodyne), f"{velodyne} does not exist!"
    dst_path = f"/home/yw763/Deeplearning/Deeplearning/pointrcnn/multi_data/{name}/KITTI/object/training/velodyne"
    if not os.path.isdir(dst_path):
        os.symlink(velodyne, dst_path)


def gen_all_data(data_names):
    for name in data_names:
        gen_data(name)


def get_true_name(model_name, data_name, args, far_points):
    res =  f"{model_name}_{data_name}_within{args.within}_farsample{far_points}"
    if args.cfg != "default":
        res += f"_{args.cfg}"
    return res


def gen_commands(model_names, data_names, args, far_points):
    if args.one2one:
        for data_name, model_name in zip(data_names, model_names):
            for fp in far_points:
                yield gen_cmd(model_paths[model_name], data_name, get_true_name(model_name, data_name, args, fp), args, fp)
    else:
        for data_name in data_names:
            for model_name in model_names:
                for fp in far_points:
                    yield gen_cmd(model_paths[model_name], data_name, get_true_name(model_name, data_name, args, fp), args, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Inference')
    parser.add_argument('--cfg', type=str, default="default")
    parser.add_argument('--mp', action="store_true")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    # gen_all_data(data_names)
    #
    # if args.mp:
    #     for cmds in grouper(n_gpus, gen_commands(model_names, data_names, args, far_points)):
    #         processes = []
    #         for i, cmd in enumerate(cmds):
    #             if cmd is not None:
    #                 print(cmd)
    #                 p = mp.Process(target=os.system, args=(f"CUDA_VISIBLE_DEVICES={i} {cmd}",))
    #                 p.start()
    #                 processes.append(p)
    #
    #         for p in processes:
    #             p.join()
    #             del p
    # else:
    #     for c in gen_commands(model_names, data_names, args, far_points):
    #         os.system(c)

