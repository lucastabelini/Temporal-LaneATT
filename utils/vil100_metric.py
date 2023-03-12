import os
import json

import numpy as np
from p_tqdm import t_map, p_map
from .culane_metric import culane_metric


def culane_metric_wrapper(prediction, annotation, img_shape):
    return culane_metric(prediction, annotation, img_shape=img_shape)


def get_macro_measure(img_list, measures):
    sequences = {}
    for img_path, iou in zip(img_list, measures):
        sequence = os.path.dirname(img_path)
        if sequence in sequences:
            sequences[sequence].append(iou)
        else:
            sequences[sequence] = [iou]
    macro_measure = 0
    for sequence in sequences:
        macro_measure += np.mean(sequences[sequence])
    return macro_measure / len(sequences)


def eval_predictions(preditions_basedir, dataset_root, img_list, sequential=False):
    print("Loading data ({} predictions/targets)...".format(len(img_list)))
    annotations, predictions, img_shapes = [], [], []
    annotations_dir = os.path.join(dataset_root, "Json")
    for img_path in img_list:
        annotation_path = img_path.replace("JPEGImages", annotations_dir) + ".json"
        prediction_path = img_path.replace("JPEGImages", preditions_basedir) + ".json"
        annotations.append(load_vil100_annotation(annotation_path))
        prediction = load_vil100_prediction(prediction_path)
        img_shapes.append((prediction["img_h"], prediction["img_w"], 3))
        predictions.append(prediction["lanes"])

    print(
        "Calculating metric {}...".format(
            "sequentially" if sequential else "in parallel"
        )
    )
    if sequential:
        results = t_map(culane_metric_wrapper, predictions, annotations, img_shapes)
    else:
        results = p_map(culane_metric_wrapper, predictions, annotations, img_shapes)
    total_tp = sum(tp for tp, _, _, _, _ in results)
    total_fp = sum(fp for _, fp, _, _, _ in results)
    total_fn = sum(fn for _, _, fn, _, _ in results)
    ious = [iou.max() if len(iou) > 0 else 0 for _, _, _, iou, _ in results]
    total_iou = get_macro_measure(img_list, ious)
    if total_tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": total_iou,
    }


def load_vil100_annotation(annotation_path):
    with open(annotation_path, "r") as annotation_file:
        annotation = json.load(annotation_file)
    lanes = [lane["points"] for lane in annotation["annotations"]["lane"]]
    lanes = remove_repeated(lanes)
    lanes = [lane for lane in lanes if len(lane) > 0]
    return lanes


def remove_repeated(lanes):
    filtered_lanes = []
    for lane in lanes:
        xs = [p[0] for p in lane]
        ys = [p[1] for p in lane]
        ys, indices = np.unique(ys, return_index=True)
        xs = np.array(xs)[indices]
        filtered_lanes.append(list(zip(xs, ys)))
    return filtered_lanes


def load_vil100_prediction(prediction_path):
    with open(prediction_path, "r") as prediction_file:
        prediction = json.load(prediction_file)
    return prediction


def load_img_list(img_list_path):
    with open(img_list_path, "r") as img_list_file:
        img_list = [img_path.strip()[1:] for img_path in img_list_file.readlines()]
    return img_list


def main():
    import sys

    print(
        eval_predictions(
            sys.argv[1],
            "/dados/tabelini/datasets/VIL100/",
            load_img_list("/dados/tabelini/datasets/VIL100/data/test.txt"),
        )
    )


if __name__ == "__main__":
    main()
