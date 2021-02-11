import time
import os
import numpy as np
import json
import fire

import tools.kitti_object_eval_python.kitti_common as kitti
from tools.kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result, calculate_iou_partly


PATH = {
    "kitti": "/nfs01/data/yw763/driving/KITTI_object",
    "argo": "/home/yw763/driving/argoverse_kitti_1232x480_zerodisp",
    "nusc": "/nfs01/data/yw763/driving/nuscenes-in-kitti-format",
    "lyft": "/nfs01/data/yw763/driving/lyft_v1.02-in-kitti-format",
    "waymo": "/nfs01/data/yw763/driving/waymo-in-kitti-format",
}


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def read_plane(fname):
    with open(fname) as f:
        return np.array(list(map(eval, f.readlines()[-1].split(" "))))


def read_planes(dir, ids):
    return np.stack([read_plane(os.path.join(dir, "%06d.txt" % i)) for i in ids], axis=0)


def anno_to_ground(anno, plane):
    anno['location'][:, 1] -= (-plane[3] - plane[0] * anno['location'][:, 0] - plane[2]  * anno['location'][:, 2]) / plane[1]
    return anno


def annos_to_ground(annos, dir, ids):
    plane = read_planes(dir, ids)
    for i in range(len(annos)):
        annos[i] = anno_to_ground(annos[i], plane[i])
    return annos


def save_labels(annos, dir, ids):
    assert len(annos) == len(ids)
    os.makedirs(dir, exist_ok=True)
    for i in range(len(annos)):
        kitti.to_kitti_format(annos[i], os.path.join(dir, "%06d.txt" % ids[i]))


def get_model(s):
    data_names = ["kitti", "argo", "nusc", "lyft", "waymo"]
    loc = np.array([s.find(x) for x in data_names])
    loc[loc == -1] = 10000
    return data_names[int(np.argmin(loc))]


def get_data(s):
    data_names = ["kitti", "argo", "nusc", "lyft", "waymo"]
    loc = [s.lower().rfind(x) for x in data_names]
    return data_names[int(np.argmax(np.array(loc)))]


def get_scale_map(src, dst):
    return lambda x: np.stack([
        (x[:, 0] - src["length"]["mean"]) / src["length"]["std"] * dst["length"]["std"] + dst["length"]["mean"],
        (x[:, 1] - src["height"]["mean"]) / src["height"]["std"] * dst["height"]["std"] + dst["height"]["mean"],
        (x[:, 2] - src["width"]["mean"]) / src["width"]["std"] * dst["width"]["std"] + dst["width"]["mean"],
    ], axis=1)


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             toground=False,
             align_size=False,
             reverse_align=False):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    for i in range(len(dt_annos)):
        if len(dt_annos[i]['name']) > 0:
            assert np.max(dt_annos[i]['location'][:, 2]) < 80, f"{os.path.join(result_path, '%06d.txt' % val_image_ids[i])}, Some detection > 80m!!!"

    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    if toground:
        dt_annos = annos_to_ground(dt_annos, os.path.join(os.path.dirname(label_path), "planes"), val_image_ids)
        save_labels(dt_annos, os.path.join(os.path.dirname(result_path), "grounded"), val_image_ids)

    gt_annos = kitti.get_label_annos(label_path, val_image_ids)

    for i in range(len(gt_annos)):
        if len(gt_annos[i]['name']) > 0:
            assert np.max(gt_annos[i]['location'][:, 2]) < 70, f"{os.path.join(label_path, '%06d.txt' % val_image_ids[i])}, Some label > 70m!!!"

    if align_size:
        overlaps, _, _, _ = calculate_iou_partly(dt_annos, gt_annos, 1)
        assert len(overlaps) == len(dt_annos) == len(gt_annos)
        for i in range(len(overlaps)):
            assert overlaps[i].shape == (len(dt_annos[i]['name']), len(gt_annos[i]['name']))
            if len(dt_annos[i]['name']) > 0 and len(gt_annos[i]['name']) > 0:
                val = np.max(overlaps[i], axis=1)
                idx = np.argmax(overlaps[i], axis=1)
                for j in range(len(dt_annos[i]['name'])):
                    if val[j] > 0.2:
                        dt_annos[i]['dimensions'][j, :] = gt_annos[i]['dimensions'][idx[j], :]
        save_labels(dt_annos, os.path.join(os.path.dirname(result_path), "align_size"), val_image_ids)

    if reverse_align:
        src = get_data(label_path)
        dst = get_model(result_path)
        print(f"{src} -> {dst}")
        with open(os.path.join(PATH[src], "label_normal_val.json")) as f:
            src = json.load(f)
        with open(os.path.join(PATH[dst], "label_normal_val.json")) as f:
            dst = json.load(f)
        mapping = get_scale_map(src, dst)
        for i in range(len(gt_annos)):
            if len(gt_annos[i]['name']) > 0:
                gt_annos[i]["dimensions"] = mapping(gt_annos[i]["dimensions"])
        save_labels(gt_annos, os.path.join(os.path.dirname(result_path), "reverse_align"), val_image_ids)


    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)


if __name__ == '__main__':
    fire.Fire()
