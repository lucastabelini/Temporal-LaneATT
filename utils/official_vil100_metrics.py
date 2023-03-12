import os
import json
import platform
import tempfile
import subprocess
import pickle as pkl

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from skimage.io import imread

# from .metrics.video_metrics.evaluation import db_eval
"""
    Line metrics
"""


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def get_pred_lanes(filename):
        with open(filename, "r") as file:
            data = json.load(file)
        img_h = data["img_h"]
        y_pred = [[img_h - p[1] for p in lane] for lane in data["lanes"]]
        x_pred = [[p[0] for p in lane] for lane in data["lanes"]]
        param = [
            np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))
        ]
        return param, img_h

    @staticmethod
    def get_gt_lanes(gt_dir, filename, height):
        gt_json = json.load(open(os.path.join(gt_dir, filename))).get("annotations")[
            "lane"
        ]
        img_height = height
        lanex_points = []
        laney_points = []
        for i in gt_json:
            for key, value in i.items():
                if key == "points" and value != []:
                    lanex = []
                    laney = []
                    for item in value:
                        lanex.append(item[0])
                        laney.append(img_height - item[1])
                    lanex_points.append(lanex)
                    laney_points.append(laney)
        return lanex_points, laney_points

    @staticmethod
    def calculate_results(param, gtx, gty):
        angles = [
            LaneEval.get_angle(np.array(gtx[i]), np.array(gty[i]))
            for i in range(len(gty))
        ]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0.0, 0.0
        matched = 0.0

        for index, (x_gts, thresh) in enumerate(zip(gtx, threshs)):
            accs = []
            for x_preds in param:
                x_pred = (
                    x_preds[0] * np.array(gty[index]) * np.array(gty[index])
                    + x_preds[1] * np.array(gty[index])
                    + x_preds[2]
                ).tolist()
                accs.append(
                    LaneEval.line_accuracy(np.array(x_pred), np.array(x_gts), thresh)
                )
            # print(accs)
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(param) - matched
        if len(gtx) > 8 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gtx) > 8:
            s -= min(line_accs)
        return (
            s / max(min(8.0, len(gtx)), 1.0),
            fp / len(param) if len(param) > 0 else 0.0,
            fn / max(min(len(gtx), 8.0), 1.0),
        )

    @staticmethod
    def calculate_return(pre_dir_name, json_dir_name):
        Preditction = pre_dir_name
        Json = json_dir_name
        num, accuracy, fp, fn = 0.0, 0.0, 0.0, 0.0
        list_preditction = os.listdir(Preditction)
        list_preditction.sort()
        for filename in list_preditction:
            pred_files = [
                x
                for x in os.listdir(os.path.join(Preditction, filename))
                if x.endswith(".json")
            ]
            json_files = os.listdir(os.path.join(Json, filename))
            pred_files.sort()
            json_files.sort()

            for pfile, jfile in zip(pred_files, json_files):
                pfile_name = os.path.join(Preditction, filename, pfile)
                param, height = LaneEval.get_pred_lanes(pfile_name)
                lanex_points, laney_points = LaneEval.get_gt_lanes(
                    os.path.join(Json, filename), jfile, height
                )

                try:
                    a, p, n = LaneEval.calculate_results(
                        param, lanex_points, laney_points
                    )
                except BaseException as e:
                    raise Exception("Format of lanes error.")
                accuracy += a
                fp += p
                fn += n
                num += 1

        accuracy = accuracy / num
        fp = fp / num
        fn = fn / num
        return accuracy, fp, fn


def official_vil100_line_metrics(pred_dir, json_anno_dir):
    accuracy, fp, fn = LaneEval.calculate_return(pred_dir, json_anno_dir)
    return {"Accuracy": accuracy, "FP": fp, "FN": fn}


"""
    Region metrics
"""


def read_helper(path):
    lines = open(path, "r").readlines()[1:]
    lines = " ".join(lines)
    values = lines.split(" ")[1::2]
    keys = lines.split(" ")[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res


def parse_anno_match(path):
    with open(path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.rstrip().split()
        datum = []
        datum.append(line[0])
        datum.append([int(d) for d in line[1:]])
        data.append(datum)
    return data


def call_culane_eval(
    data_dir, eval_cmd, output_path, temp_dir, result_dir, detect_dir, ano_dir_temp
):
    w_lane = 30
    iou = 0.5  # Set iou to 0.5 or 0.8
    frame = 1
    list = os.path.join(data_dir, "data", "test.txt")
    file = open(list)
    if not os.path.exists(os.path.join(output_path, temp_dir)):
        os.makedirs(os.path.join(output_path, temp_dir))
    if not os.path.exists(os.path.join(output_path, result_dir)):
        os.mkdir(os.path.join(output_path, result_dir))

    for line in file.readlines():  # [save_freq:]
        txt_path = os.path.join(
            output_path, temp_dir, line.strip().split("/")[2] + ".txt"
        )
        with open(txt_path, "a+") as f:
            f.write(line.strip().replace("/JPEGImages", "") + "\n")
        f.close()

    if platform.system() == "Windows":
        eval_cmd = eval_cmd.replace("/", os.sep)
    list_test_files = os.listdir(os.path.join(output_path, temp_dir))
    res_all = {}
    anno_match = []
    for list_file in list_test_files:
        txt_name = os.path.join(output_path, temp_dir, list_file)

        with open(txt_name, "r") as fp:
            frame_path = os.path.join(
                data_dir, "JPEGImages", fp.readlines()[0][1:].strip()
            )
            frame1 = imread(frame_path, as_gray=True)
        fp.close()

        out0 = os.path.join(output_path, result_dir, list_file)
        confusion_matrix_outpath = os.path.join(
            output_path, result_dir, list_file + "_cm.txt"
        )
        anno_match_outpath = os.path.join(
            output_path, result_dir, list_file + "_am.txt"
        )
        # open(out0, 'w')
        img_dir_temp = os.path.join(data_dir, "JPEGImages")

        subprocess.run(
            [
                eval_cmd,
                "-a",
                ano_dir_temp,
                "-d",
                detect_dir,
                "-i",
                img_dir_temp,
                "-l",
                txt_name,
                "-w",
                str(w_lane),
                "-t",
                str(iou),
                "-c",
                str(frame1.shape[1]),
                "-r",
                str(frame1.shape[0]),
                "-f",
                str(frame),
                "-o",
                out0,
                "-m",
                confusion_matrix_outpath,
                "-p",
                anno_match_outpath,
            ],
            capture_output=True,
        )
        res_all["out_" + str(list_file[:-4])] = read_helper(out0)
        confusion_matrix = np.loadtxt(confusion_matrix_outpath, dtype=np.int32)
        anno_match.extend(parse_anno_match(anno_match_outpath))
        res_all["out_" + str(list_file[:-4])]["cmatrix"] = confusion_matrix
        res_all["out_" + str(list_file[:-4])]["anno_match"] = anno_match

    with open("anno_match.pkl", "wb") as file:
        pkl.dump(anno_match, file)

    return res_all


def official_vil100_region_metrics(
    dataset_root, eval_exec_path, pred_dir, anno_txt_dir
):
    with tempfile.TemporaryDirectory() as work_dir:
        temp_dir = "temp_MMANet"
        result_dir = "results_MANet"

        res = call_culane_eval(
            dataset_root,
            eval_exec_path,
            work_dir,
            temp_dir,
            result_dir,
            pred_dir,
            anno_txt_dir,
        )
        TP, FP, FN, MIOU = 0, 0, 0, 0
        CLS_TP, CLS_FP, CLS_FN = 0, 0, 0
        cmatrix = np.zeros((15, 15), dtype=np.int32)
        for k, v in res.items():
            val = float(v["Fmeasure"]) if "nan" not in v["Fmeasure"] else 0
            val_tp, val_fp, val_fn, val_iou = (
                int(v["tp"]),
                int(v["fp"]),
                int(v["fn"]),
                float(v["miou"]),
            )
            val_cls_tp, val_cls_fp, val_cls_fn = (
                int(v["cls_tp"]),
                int(v["cls_fp"]),
                int(v["cls_fn"]),
            )
            cmatrix += v["cmatrix"]
            TP += val_tp
            FP += val_fp
            FN += val_fn
            CLS_TP += val_cls_tp
            CLS_FP += val_cls_fp
            CLS_FN += val_cls_fn
            MIOU += val_iou
        P = TP * 1.0 / (TP + FP)
        R = TP * 1.0 / (TP + FN)
        F = 2 * P * R / (P + R)
        CLS_P = CLS_TP * 1.0 / (CLS_TP + CLS_FP)
        CLS_R = CLS_TP * 1.0 / (CLS_TP + CLS_FN)
        CLS_F = (2 * CLS_P * CLS_R / (CLS_P + CLS_R)) if (CLS_P + CLS_R) > 0 else 0
        cls_accuracy = (CLS_TP / float(TP)) if TP > 0 else 0

        return {
            "F1": F,
            "mIoU": MIOU / len(res.items()),
            "CLS-F1": CLS_F,
            "CLS-Accuracy": cls_accuracy,
            "Confusion-Matrix": cmatrix.tolist(),
        }


def read_seg(path):
    seg = np.sum(cv2.imread(path), axis=2) > 0
    return seg


def read_points_pred(path):
    with open(path, "r") as file:
        data = json.load(file)
    img_h, img_w = data["img_h"], data["img_w"]
    ys = [[img_h - p[1] for p in lane] for lane in data["lanes"]]
    xs = [[p[0] for p in lane] for lane in data["lanes"]]
    return xs, ys, img_h, img_w


def points_to_seg(path):
    lanes_x, lanes_y, img_h, img_w = read_points_pred(path)
    seg = np.zeros((img_h, img_w), dtype=np.uint8)
    thickness = round(30 * (img_w / 1920.0))
    for idx, (xs, ys) in enumerate(zip(lanes_x, lanes_y)):
        for x1, y1, x2, y2 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            cv2.line(seg, (x1, y1), (x2, y2), (idx + 1,), thickness)
    return seg


def official_vil100_video_metrics(pred_dir, anno_dir):
    annotations = []
    predictions = []
    for clip_name in os.listdir(pred_dir):
        pred_clipdir = os.path.join(pred_dir, clip_name)
        for frame_filename in os.listdir(pred_clipdir):
            if not frame_filename.endswith(".json"):
                continue
            anno_path = os.path.join(
                anno_dir, clip_name, frame_filename.replace(".jpg.json", ".png")
            )
            pred_path = os.path.join(pred_clipdir, frame_filename)
            anno = read_seg(anno_path)
            pred = points_to_seg(pred_path)

            annotations.append(anno)
            predictions.append(pred)
            break
        break
    results = db_eval(annotations, predictions, measures="JFT")
    return results["dataset"]


def official_vil100_metrics(dataset_root, eval_exec_path, pred_dir, anno_txt_dir):
    region_metrics = official_vil100_region_metrics(
        dataset_root, eval_exec_path, pred_dir, anno_txt_dir
    )
    line_metrics = official_vil100_line_metrics(
        pred_dir, os.path.join(dataset_root, "Json")
    )
    # region_metrics = line_metrics = {}
    # video_metrics = official_vil100_video_metrics(pred_dir, os.path.join(dataset_root, 'Annotations'))

    return {**region_metrics, **line_metrics}  # , **video_metrics}


if __name__ == "__main__":
    import sys

    # print(official_vil100_video_metrics(sys.argv[1], '/mnt/ssd4/tabelini/datasets/VIL100/Annotations'))
    print(
        official_vil100_region_metrics(
            "../datasets/VIL100",
            "../MMA-Net/evaluation-cls/culane/culane_evaluator",
            sys.argv[1],
            "../MMA-Net/evaluation-cls/txt/anno_txt/",
        )
    )
