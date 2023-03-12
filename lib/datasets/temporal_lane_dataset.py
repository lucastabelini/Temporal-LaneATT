import cv2
import torch
import numpy as np
from imgaug.augmentables.lines import LineStringsOnImage

from .lane_dataset import LaneDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class TemporalLaneDataset(LaneDataset):
    def __init__(self, time_window_size=3, **kwargs):
        self.time_window_size = time_window_size
        self.sequence_idxs = []
        super().__init__(**kwargs)

    def get_sequence_idxs(self):
        clips = {}
        for anno_idx, anno in enumerate(self.dataset.annotations):
            img_path = anno["path"]
            img_clip, frame_idx = self.dataset.get_img_clip(img_path)
            if img_clip not in clips:
                clips[img_clip] = [(frame_idx, anno_idx)]
            else:
                clips[img_clip].append((frame_idx, anno_idx))
        sequences_idxs = []
        for clip in clips:
            clip_data = sorted(clips[clip], key=lambda x: x[0])
            clip_anno_idxs = [item[1] for item in clip_data]
            for i in range(self.time_window_size - 1):
                sequences_idxs.append([clip_anno_idxs[i]] * self.time_window_size)
            for i in range(len(clip_anno_idxs) - self.time_window_size + 1):
                sequences_idxs.append(clip_anno_idxs[i : i + self.time_window_size])
        return sequences_idxs

    def transform_annotations(self):
        self.logger.info("Transforming annotations to the model's target format...")
        self.dataset.annotations = np.array(
            list(map(self.transform_annotation, self.dataset.annotations))
        )
        self.logger.info("Done.")
        self.logger.info("Getting sequence indices...")
        self.sequence_idxs = self.get_sequence_idxs()
        self.logger.info("Done.")

    def draw_annotation(self, idx, label=None, pred=None, img=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            imgs, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            for i in range(len(imgs)):
                img = imgs[i].permute(1, 2, 0).numpy()
                if self.normalize:
                    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                img = (img * 255).astype(np.uint8)
                # if i < len(imgs) - 1:
                #     cv2.imshow('frame-{}'.format(i), img)
        if label is None:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
        return super().draw_annotation(idx, label, pred, img)

    def __getitem__(self, sequence_idx):
        sequence_idxs = self.sequence_idxs[sequence_idx]
        items = [self.dataset[idx] for idx in sequence_idxs]
        imgs = [cv2.imread(item["path"]) for item in items]
        target_item = self.dataset[sequence_idxs[-1]]
        target_img = imgs[-1]
        # print(item['path'])
        line_strings_org = self.lane_to_linestrings(
            target_item["old_anno"]["lanes"], target_item["old_anno"]["categories"]
        )
        line_strings_org = LineStringsOnImage(line_strings_org, shape=target_img.shape)
        for i in range(30):
            imgs_copy = [img.copy() for img in imgs]
            transform = self.transform.to_deterministic()
            for img_idx in range(self.time_window_size):
                imgs_copy[img_idx], line_strings = transform(
                    image=imgs_copy[img_idx], line_strings=line_strings_org
                )
            line_strings.clip_out_of_image_()
            lanes, categories = self.linestrings_to_lanes(line_strings)
            new_anno = {
                "path": target_item["path"],
                "lanes": lanes,
                "categories": categories,
            }
            try:
                label = self.transform_annotation(
                    new_anno, img_wh=(self.img_w, self.img_h)
                )["label"]
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical("Transform annotation failed 30 times :(")
                    exit()
        for i in range(self.time_window_size):
            imgs_copy[i] = imgs_copy[i] / 255.0
            if self.normalize:
                imgs_copy[i] = (imgs_copy[i] - IMAGENET_MEAN) / IMAGENET_STD
            imgs_copy[i] = self.to_tensor(imgs_copy[i].astype(np.float32))

        imgs = torch.stack(imgs_copy)

        return (imgs, label, sequence_idxs[-1])

    def __len__(self):
        return len(self.sequence_idxs)
