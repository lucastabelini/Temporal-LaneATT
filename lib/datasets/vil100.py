import os
import json
import logging

import numpy as np
from tqdm import tqdm

import utils.vil100_metric as vil100_metric
import utils.official_vil100_metrics as official_vil100_metrics
from .lane_dataset_loader import LaneDatasetLoader

ALL_ATTRIBUTES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13]
ATTRIBUTES_LABELS = [
    "single white solid",
    "single white dotted",
    "single yellow solid",
    "single yellow dotted",
    "double white solid",
    "double yellow solid",
    "double yellow dotted",
    "double white solid dotted",
    "double white dotted solid",
    "double solid white and yellow",
]
ATTRIBUTE_TO_CAT_ID = {
    attribute: cat_id
    for attribute, cat_id in zip(ALL_ATTRIBUTES, range(len(ALL_ATTRIBUTES)))
}
CAT_ID_TO_ATTRIBUTE = {
    cat_id: attribute for attribute, cat_id in ATTRIBUTE_TO_CAT_ID.items()
}
CAT_ID_TO_ATTRIBUTE[20] = 14


class VIL100(LaneDatasetLoader):
    def __init__(
        self,
        root,
        split="train",
        eval_exec_path=None,
        anno_txt_path=None,
        use_official_metric=True,
        max_lanes=None,
    ):
        self.root = root
        self.split = split
        self.max_lanes = max_lanes
        self.img_height = {}
        self.img_width = {}
        self.img_list = []
        self.annotations = []
        self.eval_exec_path = eval_exec_path
        self.anno_txt_path = anno_txt_path
        self.use_official_metric = use_official_metric
        self.logger = logging.getLogger(__name__)
        self.load_annotations()

    def get_img_clip(self, img_path):
        clip_dir = os.path.dirname(img_path)
        frame_name = os.path.basename(img_path)
        frame_number = int(frame_name.split(".")[0])
        return clip_dir, frame_number

    def get_img_heigth(self, path):
        """Returns the image's height in pixels"""
        return CLIP_SIZES[os.path.split(os.path.dirname(path))[1]][1]

    def get_img_width(self, path):
        """Returns the image's width in pixels"""
        return CLIP_SIZES[os.path.split(os.path.dirname(path))[1]][0]

    def get_metrics(self, lanes, idx):
        """Returns dataset's metrics for a prediction `lanes`

        A tuple `(fp, fn, matches, accs)` should be returned, where `fp` and `fn` indicate the number of false-positives
        and false-negatives, respectively, matches` is a list with a boolean value for each
        prediction in `lanes` indicating if the prediction is a true positive and `accs` is a metric indicating the
        quality of each prediction (e.g., the IoU with an annotation)

        If the metrics can't be computed, placeholder values should be returned.
        """
        img_path = self.annotations[idx]["path"]
        lanes = self.get_prediction_as_points(lanes, img_path)
        anno = vil100_metric.load_vil100_annotation(
            img_path.replace("JPEGImages", "Json") + ".json"
        )
        img_h, img_w = self.get_img_heigth(img_path), self.get_img_width(img_path)
        _, fp, fn, ious, matches = vil100_metric.culane_metric(
            lanes, anno, img_shape=(img_h, img_w, 3)
        )

        return fp, fn, matches, ious

    def load_annotations(self):
        """Loads all annotations from the dataset

        Should return a list where each item is a dictionary with keys `path` and `lanes`, where `path` is the path to
        the image and `lanes` is a list of lanes, represented by a list of points for example:

        return [{
            'path': 'example/path.png' # path to the image
            'lanes': [[10, 20], [20, 25]]
        }]
        """

        self.annotations = []
        self.img_height = {}
        self.img_width = {}
        self.max_lanes = 0
        os.makedirs("cache", exist_ok=True)
        cache_path = "cache/vil100_{}.json".format(self.split)

        if os.path.exists(cache_path):
            self.logger.info("Loading VIL100 annotations (cached)...")
            with open(cache_path, "r") as cache_file:
                data = json.load(cache_file)
                self.annotations = data["annotations"]
                self.max_lanes = data["max_lanes"]
                self.img_list = data["img_list"]
        else:
            img_list_path = os.path.join(self.root, "data", "{}.txt".format(self.split))
            with open(img_list_path, "r") as img_list_file:
                self.img_list = [
                    img_path.strip()[1:] for img_path in img_list_file.readlines()
                ]
            for img_path in self.img_list:
                json_path = os.path.join(
                    self.root, img_path.replace("JPEGImages", "Json") + ".json"
                )
                with open(json_path, "r") as json_file:
                    annotation_data = json.load(json_file)
                    full_img_path = os.path.join(self.root, img_path)
                    self.max_lanes = max(
                        self.max_lanes, len(annotation_data["annotations"]["lane"])
                    )
                    lanes = [
                        lane["points"]
                        for lane in annotation_data["annotations"]["lane"]
                    ]
                    categories = [
                        ATTRIBUTE_TO_CAT_ID[lane["attribute"]]
                        for lane in annotation_data["annotations"]["lane"]
                    ]
                    self.annotations.append(
                        {
                            "path": full_img_path,
                            "lanes": lanes,
                            "categories": categories,
                            "org_path": annotation_data["info"]["image_path"],
                        }
                    )

            with open(cache_path, "w") as cache_file:
                json.dump(
                    {
                        "annotations": self.annotations,
                        "max_lanes": self.max_lanes,
                        "img_width": self.img_width,
                        "img_height": self.img_height,
                        "img_list": self.img_list,
                    },
                    cache_file,
                )
        # self.annotations = self.annotations[:10]
        self.logger.info(
            "%d annotations loaded, with a maximum of %d lanes in an image.",
            len(self.annotations),
            self.max_lanes,
        )

    def get_prediction_as_points(self, pred, img_path):
        img_h = self.get_img_heigth(img_path)
        img_w = self.get_img_width(img_path)
        ys = np.arange(img_h) / img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane = list(zip(lane_xs, lane_ys))
            if len(lane) > 1:
                lanes.append(lane)

        return lanes

    def get_prediction_as_string(self, pred, img_path):
        img_h = self.get_img_heigth(img_path)
        img_w = self.get_img_width(img_path)
        ys = np.arange(img_h) / img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            if "category" in lane.metadata:
                category = CAT_ID_TO_ATTRIBUTE[lane.metadata["category"]]
            else:
                category = 0
            lane_str = (
                str(category)
                + " "
                + " ".join(
                    ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
                )
            )
            if lane_str != "":
                out.append(lane_str)

        return "\n".join(out)

    def eval_predictions(self, predictions, output_basedir):
        """Should return a dictionary with each metric's results
        Example:
        return {
            'F1': 0.9
            'Acc': 0.95
        }
        """
        print("Generating prediction output...")
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(
                output_basedir,
                os.path.dirname(self.annotations[idx]["old_anno"]["org_path"]),
            )
            output_filename = (
                os.path.basename(self.annotations[idx]["old_anno"]["org_path"])
                + ".json"
            )
            os.makedirs(output_dir, exist_ok=True)
            img_path = self.annotations[idx]["path"]
            lanes = self.get_prediction_as_points(pred, img_path)
            # save output in my format
            with open(os.path.join(output_dir, output_filename), "w") as out_file:
                output = {
                    "img_w": self.get_img_width(img_path),
                    "img_h": self.get_img_heigth(img_path),
                    "lanes": lanes,
                }
                json.dump(output, out_file)
            # save output in CULane's format
            culane_output_filename = os.path.basename(
                self.annotations[idx]["old_anno"]["org_path"]
            ).replace(".jpg", ".lines.txt")
            culane_format_output = self.get_prediction_as_string(pred, img_path)
            with open(
                os.path.join(output_dir, culane_output_filename), "w"
            ) as out_file:
                out_file.write(culane_format_output)
        if self.use_official_metric:
            self.logger.info("Computing metrics...")
            return official_vil100_metrics.official_vil100_metrics(
                self.root,
                self.eval_exec_path,
                os.path.abspath(output_basedir),
                self.anno_txt_path,
            )
        else:
            return vil100_metric.eval_predictions(
                output_basedir, self.root, self.img_list
            )

    def __getitem__(self, idx):
        """Should return the annotation with index idx"""
        return self.annotations[idx]

    def __len__(self):
        """Should return the number of samples in the dataset"""
        return len(self.annotations)


CLIP_SIZES = {
    "6_Road022_Trim001_frames": [1920, 1080],
    "4_Road027_Trim011_frames": [1920, 1080],
    "4_Road011_Trim001_frames": [1920, 1080],
    "2_Road015_Trim001_frames": [1280, 720],
    "6_Road024_Trim001_frames": [1920, 1080],
    "4_Road026_Trim001_frames": [1920, 1080],
    "0_Road029_Trim001_frames": [960, 480],
    "0_Road001_Trim003_frames": [1920, 1080],
    "0_Road001_Trim007_frames": [1920, 1080],
    "0_Road014_Trim004_frames": [1280, 720],
    "0_Road014_Trim005_frames": [1280, 720],
    "0_Road015_Trim008_frames": [1280, 720],
    "0_Road029_Trim002_frames": [960, 480],
    "0_Road029_Trim003_frames": [960, 480],
    "0_Road029_Trim004_frames": [960, 480],
    "0_Road029_Trim005_frames": [960, 480],
    "0_Road030_Trim001_frames": [960, 478],
    "0_Road030_Trim002_frames": [960, 478],
    "0_Road030_Trim003_frames": [960, 478],
    "0_Road031_Trim001_frames": [960, 480],
    "0_Road031_Trim003_frames": [960, 480],
    "0_Road031_Trim004_frames": [960, 480],
    "0_Road036_Trim004_frames": [960, 480],
    "0_Road036_Trim005_frames": [960, 480],
    "125_Road018_Trim005_frames": [1920, 1080],
    "125_Road018_Trim007_frames": [1920, 1080],
    "1269_Road022_Trim002_frames": [1920, 1080],
    "1269_Road023_Trim003_frames": [1920, 1080],
    "12_Road014_Trim002_frames": [1280, 720],
    "12_Road017_Trim005_frames": [1920, 1080],
    "12_Road018_Trim003_frames": [1920, 1080],
    "12_Road018_Trim004_frames": [1920, 1080],
    "15_Road001_Trim004_frames": [1920, 1080],
    "15_Road018_Trim008_frames": [1920, 1080],
    "1_Road001_Trim002_frames": [1920, 1080],
    "1_Road001_Trim005_frames": [1920, 1080],
    "1_Road001_Trim006_frames": [1920, 1080],
    "1_Road010_Trim002_frames": [1920, 1080],
    "1_Road012_Trim002_frames": [1920, 1080],
    "1_Road012_Trim003_frames": [1920, 1080],
    "1_Road012_Trim004_frames": [1920, 1080],
    "1_Road013_Trim003_frames": [1920, 1080],
    "1_Road013_Trim004_frames": [1920, 1080],
    "1_Road014_Trim001_frames": [1280, 720],
    "1_Road014_Trim007_frames": [1280, 720],
    "1_Road015_Trim008_frames": [1280, 720],
    "1_Road017_Trim002_frames": [1920, 1080],
    "1_Road017_Trim010_frames": [1920, 1080],
    "1_Road018_Trim002_frames": [1920, 1080],
    "1_Road018_Trim006_frames": [1920, 1080],
    "1_Road018_Trim009_frames": [1920, 1080],
    "1_Road018_Trim016_frames": [1920, 1080],
    "1_Road031_Trim005_frames": [960, 480],
    "1_Road034_Trim003_frames": [960, 478],
    "25_Road011_Trim005_frames": [1920, 1080],
    "25_Road015_Trim003_frames": [1280, 720],
    "25_Road015_Trim006_frames": [1280, 720],
    "25_Road026_Trim004_frames": [1920, 1080],
    "27_Road006_Trim001_frames": [672, 378],
    "2_Road001_Trim009_frames": [1920, 1080],
    "2_Road009_Trim002_frames": [1920, 1080],
    "2_Road010_Trim001_frames": [1920, 1080],
    "2_Road010_Trim003_frames": [1920, 1080],
    "2_Road011_Trim003_frames": [1920, 1080],
    "2_Road011_Trim004_frames": [1920, 1080],
    "2_Road012_Trim001_frames": [1920, 1080],
    "2_Road013_Trim002_frames": [1920, 1080],
    "2_Road014_Trim003_frames": [1280, 720],
    "2_Road015_Trim002_frames": [1280, 720],
    "2_Road017_Trim001_frames": [1920, 1080],
    "2_Road017_Trim004_frames": [1920, 1080],
    "2_Road018_Trim010_frames": [1920, 1080],
    "2_Road026_Trim003_frames": [1920, 1080],
    "2_Road036_Trim003_frames": [960, 480],
    "3_Road017_Trim007_frames": [1920, 1080],
    "3_Road017_Trim008_frames": [1920, 1080],
    "3_Road017_Trim009_frames": [1920, 1080],
    "49_Road028_Trim003_frames": [1920, 1080],
    "4_Road011_Trim002_frames": [1920, 1080],
    "4_Road017_Trim006_frames": [1920, 1080],
    "4_Road027_Trim005_frames": [1920, 1080],
    "4_Road027_Trim006_frames": [1920, 1080],
    "4_Road027_Trim013_frames": [1920, 1080],
    "4_Road027_Trim015_frames": [1920, 1080],
    "4_Road028_Trim012_frames": [1920, 1080],
    "4_Road028_Trim014_frames": [1920, 1080],
    "5_Road001_Trim001_frames": [1920, 1080],
    "5_Road001_Trim008_frames": [1920, 1080],
    "5_Road017_Trim003_frames": [1920, 1080],
    "78_Road002_Trim001_frames": [640, 368],
    "7_Road003_Trim001_frames": [960, 448],
    "7_Road005_Trim001_frames": [672, 378],
    "8_Road033_Trim001_frames": [960, 474],
    "8_Road033_Trim002_frames": [960, 474],
    "8_Road033_Trim003_frames": [960, 474],
    "8_Road033_Trim004_frames": [960, 474],
    "8_Road033_Trim005_frames": [960, 474],
    "9_Road026_Trim002_frames": [1920, 1080],
    "9_Road028_Trim001_frames": [1920, 1080],
    "9_Road028_Trim005_frames": [1920, 1080],
}


def main():
    from tabulate import tabulate

    table = [["Category", "Train", "Test", "Total"]]
    freqs = {}
    for split in ["train", "test"]:
        dataset = VIL100("../datasets/VIL100", split=split)
        freq = {i: 0 for i in ATTRIBUTES_LABELS}
        for idx in range(len(dataset)):
            for cat in dataset[idx]["categories"]:
                freq[ATTRIBUTES_LABELS[cat]] += 1
        freqs[split] = freq
    for cat in ATTRIBUTES_LABELS:
        table.append(
            [
                cat,
                freqs["train"][cat],
                freqs["test"][cat],
                freqs["train"][cat] + freqs["test"][cat],
            ]
        )
    print(tabulate(table, headers="firstrow"))


if __name__ == "__main__":
    main()
