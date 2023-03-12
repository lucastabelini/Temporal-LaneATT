import logging
from warnings import catch_warnings

import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from lib.lane import Lane

from .culane import CULane
from .tusimple import TuSimple
from .llamas import LLAMAS
from .vil100 import VIL100, CAT_ID_TO_ATTRIBUTE, ATTRIBUTE_TO_CAT_ID
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(
        self,
        S=72,
        dataset="tusimple",
        augmentations=None,
        normalize=False,
        img_size=(360, 640),
        aug_chance=1.0,
        **kwargs
    ):
        super(LaneDataset, self).__init__()
        if dataset == "tusimple":
            self.dataset = TuSimple(**kwargs)
        elif dataset == "culane":
            self.dataset = CULane(**kwargs)
        elif dataset == "llamas":
            self.dataset = LLAMAS(**kwargs)
        elif dataset == "vil100":
            self.dataset = VIL100(**kwargs)
        elif dataset == "nolabel_dataset":
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()
        self.n_strips = S - 1
        self.n_offsets = S
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips
        self.logger = logging.getLogger(__name__)

        # y at each x offset
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.transform_annotations()

        if augmentations is not None:
            # add augmentations
            augmentations = [
                getattr(iaa, aug["name"])(**aug["parameters"]) for aug in augmentations
            ]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential(
            [Resize({"height": self.img_h, "width": self.img_w})]
        )
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential(
            [iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations]
        )
        self.max_lanes = self.dataset.max_lanes

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        self.logger.info("Transforming annotations to the model's target format...")
        self.dataset.annotations = np.array(
            list(map(self.transform_annotation, self.dataset.annotations))
        )
        self.logger.info("Done.")

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno["path"])
            img_w = self.dataset.get_img_width(anno["path"])
        else:
            img_w, img_h = img_wh

        old_lanes = anno["lanes"]
        categories = anno[
            "categories"
        ]  # if 'categories' in anno else [1] * len(old_lanes)

        # removing lanes with less than 2 points
        categories = [cat for cat, lane in zip(categories, old_lanes) if len(lane) > 1]
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [
            [
                [x * self.img_w / float(img_w), y * self.img_h / float(img_h)]
                for x, y in lane
            ]
            for lane in old_lanes
        ]
        # old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
        #              for lane in [old_lanes[0]]]
        # create tranformed annotations
        lanes = (
            np.ones(
                (self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                dtype=np.float32,
            )
            * -1e5
        )  # 2 scores, 1 start_x, 1 start_y, 1 length, S+1 coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            start_y = round((1 - max([p[1] for p in lane]) / img_h) * self.n_strips)

            try:
                xs_extrap, xs_interp = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_interp) < 2:
                continue
            all_xs = np.hstack((xs_extrap, xs_interp))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = categories[lane_idx]
            lanes[lane_idx, 2] = xs_interp[0]
            lanes[lane_idx, 3] = start_y / self.n_strips
            lanes[lane_idx, 4] = (len(xs_interp) + start_y) / self.n_strips
            lanes[lane_idx, 5 : 5 + len(all_xs)] = all_xs

        new_anno = {"path": anno["path"], "label": lanes, "old_anno": anno}
        return new_anno

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3, len(points) - 1)
        )
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[
            (sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)
        ]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)

        return extrap_xs, interp_xs
        # all_xs = np.hstack((extrap_xs, interp_xs))
        #
        # # separate between inside and outside points
        # inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        # xs_inside_image = all_xs[inside_mask]
        # xs_outside_image = all_xs[~inside_mask]
        #
        # return xs_outside_image, xs_inside_image

    def label_to_lanes(self, label):
        lanes = []
        for l in label:
            if l[0] == 1:
                continue
            xs = l[5:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[3] * self.n_strips))
            end = int(round(l[4] * self.n_strips))
            xs = xs[start:end][::-1]
            ys = ys[start:end][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points, metadata={"category": int(l[1])}))
        return lanes

    def draw_annotation(self, idx, label=None, pred=None, img=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        if label is None:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
        img = cv2.resize(img, (self.img_w, self.img_h))

        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros(
                (self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8
            )
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            print("fp: {} | fn: {}".format(fp, fn))
            print(len(matches), "matches")
            print(matches, accs)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = l.points
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(
                        img,
                        tuple(curr_p),
                        tuple(next_p),
                        color=color,
                        thickness=1 if matches is None else 1,
                    )
                # if 'start_x' in l.metadata:
                #     start_x = l.metadata['start_x'] * img.shape[1]
                #     start_y = l.metadata['start_y'] * img.shape[0]
                #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                #                radius=5,
                #                color=(0, 0, 255),
                #                thickness=-1)
                # if len(xs) == 0:
                #     print("Empty pred")
                if "track_id" in l.metadata:
                    track_id = l.metadata["track_id"]
                    cv2.putText(
                        img,
                        "id={}".format(track_id),
                        (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.7,
                        color=color,
                    )
                if "category" in l.metadata:
                    category = CAT_ID_TO_ATTRIBUTE[l.metadata["category"]]
                    if color == GT_COLOR:
                        offset = 50
                    else:
                        offset = 0
                    cv2.putText(
                        img,
                        "cat={}".format(category),
                        (
                            int(xs[len(xs) // 2] + pad),
                            int(ys[len(xs) // 2] + pad + offset),
                        ),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.7,
                        color=color,
                    )
                #     cv2.putText(img,
                #                 '{:.0f}'.format(l.metadata['conf'] * 100),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=(255, 0, 255))
        return img, fp, fn

    def lane_to_linestrings(self, lanes, categories):
        lines = []
        for lane, category in zip(lanes, categories):
            lines.append(LineString(lane, category))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        categories = []
        for line in lines:
            lanes.append(line.coords)
            categories.append(line.label)

        return lanes, categories

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_org = cv2.imread(item["path"])
        # print(item['path'])
        line_strings_org = self.lane_to_linestrings(
            item["old_anno"]["lanes"], item["old_anno"]["categories"]
        )
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        for i in range(30):
            img, line_strings = self.transform(
                image=img_org.copy(), line_strings=line_strings_org
            )
            line_strings.clip_out_of_image_()
            lanes, categories = self.linestrings_to_lanes(line_strings)
            new_anno = {"path": item["path"], "lanes": lanes, "categories": categories}
            try:
                label = self.transform_annotation(
                    new_anno, img_wh=(self.img_w, self.img_h)
                )["label"]
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical("Transform annotation failed 30 times :(")
                    exit()

        img = img / 255.0
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)
