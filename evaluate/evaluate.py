import os
import numpy as np
import json
import argparse
import pickle

import kitti_common as kitti
import pdb


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


def get_scale_map_gaussian(src, dst):
    return lambda x: np.stack([
        (x[:, 0] - src["length"]["mean"]) / src["length"]["std"] * dst["length"]["std"] + dst["length"]["mean"],
        (x[:, 1] - src["height"]["mean"]) / src["height"]["std"] * dst["height"]["std"] + dst["height"]["mean"],
        (x[:, 2] - src["width"]["mean"]) / src["width"]["std"] * dst["width"]["std"] + dst["width"]["mean"],
    ], axis=1)


def get_scale_map_regular(src, dst):
    return lambda x: np.stack([
        x[:, 0] - src["length"]["mean"] + dst["length"]["mean"],
        x[:, 1] - src["height"]["mean"] + dst["height"]["mean"],
        x[:, 2] - src["width"]["mean"] + dst["width"]["mean"],
    ], axis=1)


def get_scale_map_log(src, dst):
    return lambda x: np.stack([
        x[:, 0] / src["length"]["mean"] * dst["length"]["mean"],
        x[:, 1] / src["height"]["mean"] * dst["height"]["mean"],
        x[:, 2] / src["width"]["mean"] * dst["width"]["mean"],
    ], axis=1)

get_scale_map = get_scale_map_regular


def evaluate(result_path,
             dataset_path=None,
             label_split_file=None,
             label_path=None,
             metric="new",
             dataset="kitti",
             current_class=0,
             coco=False,
             score_thresh=-1,
             toground=False,
             rescale_pred=None,
             align_size=False,
             align_front=False,
             reverse_align=False,
             dense_sample=False,
             direct_save=False,
             output_iou=False,
             adapted=False):
    label_split_file = label_split_file or os.path.join(dataset_path, "val.txt")
    label_path = label_path or os.path.join(dataset_path, "training", "label_2")
    if metric == "old":
        from eval_old import get_official_eval_result, get_coco_eval_result, calculate_iou_partly
    else:
        from eval2 import get_official_eval_result, get_coco_eval_result, calculate_iou_partly
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    # for i in range(len(dt_annos)):
    #     if len(dt_annos[i]['name']) > 0:
    #         assert np.max(dt_annos[i]['location'][:, 2]) < 80, f"{os.path.join(result_path, '%06d.txt' % val_image_ids[i])}, Some detection > 80m!!!"

    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    if toground:
        dt_annos = annos_to_ground(dt_annos, os.path.join(os.path.dirname(label_path), "planes"), val_image_ids)
        save_labels(dt_annos, os.path.join(os.path.dirname(result_path), "grounded"), val_image_ids)

    if rescale_pred is not None:
        for anno in dt_annos:
            anno['dimensions'] *= rescale_pred

    gt_annos = kitti.get_label_annos(label_path, val_image_ids)


    # for i in range(len(gt_annos)):
    #     if len(gt_annos[i]['name']) > 0:
    #         assert np.max(gt_annos[i]['location'][:, 2]) < 70, f"{os.path.join(label_path, '%06d.txt' % val_image_ids[i])}, Some label > 70m!!!"
    if output_iou:
        target_dir = os.path.join(os.path.dirname(result_path), "with_iou")

        os.makedirs(target_dir, exist_ok=True)

        overlaps, _, _, _ = calculate_iou_partly(dt_annos, gt_annos, 1)
        assert len(overlaps) == len(dt_annos) == len(gt_annos)
        for i in range(len(overlaps)):
            assert overlaps[i].shape == (len(dt_annos[i]['name']), len(gt_annos[i]['name']))
            if len(dt_annos[i]['name']) > 0 and len(gt_annos[i]['name']) > 0:
                val = np.max(overlaps[i], axis=1)
            else:
                val = np.zeros(len(dt_annos[i]['name']))
            try:
                n = len(dt_annos[i]["name"])
                kitti_str = []
                for j in range(n):
                    kitti_str.append('%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (
                        dt_annos[i]['name'][j], dt_annos[i]['truncated'][j], dt_annos[i]['occluded'][j],
                        dt_annos[i]['alpha'][j],
                        dt_annos[i]['bbox'][j, 0], dt_annos[i]['bbox'][j, 1], dt_annos[i]['bbox'][j, 2],
                        dt_annos[i]['bbox'][j, 3],
                        dt_annos[i]['dimensions'][j, 1], dt_annos[i]['dimensions'][j, 2], dt_annos[i]['dimensions'][j, 0],
                        dt_annos[i]['location'][j, 0], dt_annos[i]['location'][j, 1], dt_annos[i]['location'][j, 2],
                        dt_annos[i]['rotation_y'][j], dt_annos[i]['score'][j], val[j]))
                with open(os.path.join(target_dir, "%06d.txt" % val_image_ids[i]), "w") as f:
                    f.write("\n".join(kitti_str))
            except:
                pdb.set_trace()

        target_dir = os.path.join(os.path.dirname(result_path), "with_iou_gt")

        os.makedirs(target_dir, exist_ok=True)

        for i in range(len(overlaps)):
            assert overlaps[i].shape == (len(dt_annos[i]['name']), len(gt_annos[i]['name']))
            if len(dt_annos[i]['name']) > 0 and len(gt_annos[i]['name']) > 0:
                val = np.max(overlaps[i], axis=0)
            else:
                val = np.zeros(len(gt_annos[i]['name']))
            try:
                n = len(gt_annos[i]["name"])
                kitti_str = []
                for j in range(n):
                    kitti_str.append('%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (
                        gt_annos[i]['name'][j], gt_annos[i]['truncated'][j], gt_annos[i]['occluded'][j],
                        gt_annos[i]['alpha'][j],
                        gt_annos[i]['bbox'][j, 0], gt_annos[i]['bbox'][j, 1], gt_annos[i]['bbox'][j, 2],
                        gt_annos[i]['bbox'][j, 3],
                        gt_annos[i]['dimensions'][j, 1], gt_annos[i]['dimensions'][j, 2], gt_annos[i]['dimensions'][j, 0],
                        gt_annos[i]['location'][j, 0], gt_annos[i]['location'][j, 1], gt_annos[i]['location'][j, 2],
                        gt_annos[i]['rotation_y'][j], gt_annos[i]['score'][j], val[j]))
                with open(os.path.join(target_dir, "%06d.txt" % val_image_ids[i]), "w") as f:
                    f.write("\n".join(kitti_str))
            except:
                pdb.set_trace()

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

    if align_front:
        overlaps, _, _, _ = calculate_iou_partly(dt_annos, gt_annos, 1)
        assert len(overlaps) == len(dt_annos) == len(gt_annos)
        for i in range(len(overlaps)):
            assert overlaps[i].shape == (len(dt_annos[i]['name']), len(gt_annos[i]['name']))
            if len(dt_annos[i]['name']) > 0 and len(gt_annos[i]['name']) > 0:
                val = np.max(overlaps[i], axis=1)
                idx = np.argmax(overlaps[i], axis=1)
                for j in range(len(dt_annos[i]['name'])):
                    if val[j] > 0.2:
                        dist = np.linalg.norm(dt_annos[i]['location'][j, :])
                        alpha = dt_annos[i]['alpha'][j]
                        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
                        if np.abs(np.sin(alpha)) * dist > dt_annos[i]['dimensions'][j, 2] / 2.0:
                            shift = (dt_annos[i]['dimensions'][j, 2] - gt_annos[i]['dimensions'][idx[j], 2]) / 2.0
                            if 0 < alpha:
                                angle = -dt_annos[i]['rotation_y'][j]
                            else:
                                angle = -dt_annos[i]['rotation_y'][j] + np.pi
                            dt_annos[i]['location'][j, 0] += shift * np.cos(angle)
                            dt_annos[i]['location'][j, 2] += shift * np.sin(angle)
                        if np.abs(np.cos(alpha)) * dist > dt_annos[i]['dimensions'][j, 1] / 2.0:
                            shift = (dt_annos[i]['dimensions'][j, 1] - gt_annos[i]['dimensions'][idx[j], 1]) / 2.0
                            if -np.pi / 2.0 < alpha < np.pi / 2.0:
                                angle = -dt_annos[i]['rotation_y'][j] - np.pi / 2.0
                            else:
                                angle = -dt_annos[i]['rotation_y'][j] + np.pi / 2.0
                            dt_annos[i]['location'][j, 0] += shift * np.cos(angle)
                            dt_annos[i]['location'][j, 2] += shift * np.sin(angle)
                        dt_annos[i]['dimensions'][j, :] = gt_annos[i]['dimensions'][idx[j], :]
        save_labels(dt_annos, os.path.join(os.path.dirname(result_path), "align_front"), val_image_ids)

    if reverse_align:
        import sys
        sys.path.insert(0, "..")
        from config_path import dataset_paths

        src = get_model(label_path)
        dst = get_model(result_path)
        print("label_path:", label_path)
        print("result_path:", result_path)
        print(f"{src} -> {dst}")
        with open(os.path.join(dataset_paths[src], "label_normal_val.json")) as f:
            src = json.load(f)
        with open(os.path.join(dataset_paths[dst], "label_normal_val.json")) as f:
            dst = json.load(f)
        mapping = get_scale_map(src, dst)
        for i in range(len(gt_annos)):
            if len(gt_annos[i]['name']) > 0:
                gt_annos[i]["dimensions"] = mapping(gt_annos[i]["dimensions"])
        save_labels(gt_annos, os.path.join(os.path.dirname(result_path), "reverse_align"), val_image_ids)


    if not output_iou:
        if coco:
            return get_coco_eval_result(gt_annos, dt_annos, current_class)
        else:
            ap_result_str, ap_dict = get_official_eval_result(gt_annos, dt_annos, current_class, dataset, dense_sample=dense_sample)
            if direct_save:
                result_path = os.path.dirname(result_path)
                fname = os.path.basename(result_path) + "_val20"
                if toground:
                    fname += "_ground"
                if align_size:
                    fname += "_align_size"
                if reverse_align:
                    fname += "_reverse_align"
                if adapted:
                    fname += "_adapted"

                print(f"Saving to {os.path.join(os.path.dirname(result_path), fname+'.pkl')}")
                with open(os.path.join(os.path.dirname(result_path), fname+'.pkl'), "wb") as fb:
                    pickle.dump(ap_dict["result"], fb)
                with open(os.path.join(os.path.dirname(result_path), fname+'.txt'), "w") as f:
                    f.write(ap_result_str)
            return ap_result_str, ap_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--result_path", type=str, help="predictions to be evaluated", required=True)
    parser.add_argument("--dataset_path", type=str, help="KITTI format dataset path", default=None)
    parser.add_argument("--label_split_file", type=str, help="split file containing data ids to be evaluated", default=None)
    parser.add_argument("--label_path", type=str, help="ground truth label files", default=None)
    parser.add_argument("--metric", type=str, default="new", choices=["new", "old"], help="determine difficulty with [old: bbox height, new: distance]")
    parser.add_argument("--current_class", type=int, default=0, choices=range(5), help="0: Car, 1: Pedestrian, 2: Cyclist, 3: Van, 4: Person_sitting")

    parser.add_argument("--toground", action="store_true", help="move predictions to ground plane")
    parser.add_argument("--rescale_pred", type=int, default=None, help="scale all prediction boxes with this ratio")
    parser.add_argument("--align_size", action="store_true", help="set prediction box size same as ground truth")
    parser.add_argument("--align_front", action="store_true", help="align bbox's face facing camera with ground truth")
    parser.add_argument("--reverse_align", action="store_true", help="apply statistical normalization to ground truth")
    args = parser.parse_args()

    assert args.dataset_path is not None or args.label_split_file is not None and args.label_path is not None
    info, data = evaluate(**vars(args))
    print(info)
