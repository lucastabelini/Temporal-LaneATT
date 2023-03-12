import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from nms import nms

from lib.focal_loss import FocalLoss
from lib.lane import Lane

from .classification_head import DirectPredictionHead, FeaturePredictionHead
from .matching import match_proposals_with_targets
from .resnet import resnet18, resnet34


class LaneATTTime(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained_backbone=True,
        S=72,
        img_w=640,
        img_h=360,
        cls_pred_type="disabled",
        time_window_size=None,
        anchors_freq_path=None,
        use_low_level_features=False,
        pretrain_path=None,
        topk_anchors=None,
        anchor_feat_channels=64,
        use_past_attention=False,
        **kwargs
    ):
        super(LaneATTTime, self).__init__()
        # Some definitions
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(
            backbone, pretrained_backbone
        )
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.use_low_level_features = use_low_level_features
        self.use_past_attention = use_past_attention
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(
            1, 0, steps=self.fmap_h, dtype=torch.float32
        )
        self.anchor_feat_channels = anchor_feat_channels
        if self.use_low_level_features:
            self.anchor_feat_channels *= 2

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72.0, 60.0, 49.0, 39.0, 30.0, 22.0]
        self.right_angles = [108.0, 120.0, 131.0, 141.0, 150.0, 158.0]
        self.bottom_angles = [
            165.0,
            150.0,
            141.0,
            131.0,
            120.0,
            108.0,
            100.0,
            90.0,
            80.0,
            72.0,
            60.0,
            49.0,
            39.0,
            30.0,
            15.0,
        ]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(
            lateral_n=72, bottom_n=128
        )

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        (
            self.cut_zs,
            self.cut_ys,
            self.cut_xs,
            self.invalid_mask,
        ) = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h
        )

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(
            backbone_nb_channels, anchor_feat_channels, kernel_size=1
        )
        if use_low_level_features:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 0)),
                nn.Conv2d(
                    backbone_nb_channels // 2, anchor_feat_channels, kernel_size=1
                ),
            )
        head_nb_inputs = (
            (time_window_size + 1) * self.anchor_feat_channels * self.fmap_h
        )
        if use_past_attention:
            head_nb_inputs = (
                (2 * time_window_size) * self.anchor_feat_channels * self.fmap_h
            )
        self.cls_layer = nn.Linear(head_nb_inputs, 2)
        self.reg_layer = nn.Linear(head_nb_inputs, self.n_offsets + 2)
        if cls_pred_type == "direct":
            self.cat_cls_layer = DirectPredictionHead(head_nb_inputs)
        elif cls_pred_type == "feature":
            self.cat_cls_layer = FeaturePredictionHead(head_nb_inputs)
        elif cls_pred_type == "disabled":
            self.cat_cls_layer = None
        else:
            raise Exception("Invalid cls_pred_type `{}`".format(cls_pred_type))
        # self.cat_cls_layer = nn.Linear(head_nb_inputs, num_categories)
        self.attention_layer = nn.Linear(
            self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1
        )
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

        # Load pretrain
        if pretrain_path is not None:
            model = torch.load(pretrain_path)["model"]
            self.load_state_dict(model, strict=False)
            for param in self.parameters():
                param.requires_grad = False
            for param in self.cat_cls_layer.parameters():
                param.requires_grad = True

    def attention_mechanism(self, batch_anchor_features, batch_size):
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(batch_size, len(self.anchors), -1)
        attention_matrix = torch.eye(
            attention.shape[1], device=batch_anchor_features.device
        ).repeat(batch_size, 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0.0, as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[
            non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]
        ] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(
            batch_size, len(self.anchors), -1
        )
        attention_features = torch.bmm(
            torch.transpose(batch_anchor_features, 1, 2),
            torch.transpose(attention_matrix, 1, 2),
        ).transpose(1, 2)
        attention_features = attention_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )

        return attention_features

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        # x.shape: (B, T, C, H, W) = batch, time, channels, height, width
        batch_size, time_window = x.shape[:2]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        if self.use_low_level_features:
            batch_features, batch_lowlevel_features = self.feature_extractor(
                x, get_lowlevel_features=True
            )
            batch_features = self.conv1(batch_features)
            batch_lowlevel_features = self.downsample(batch_lowlevel_features)
            batch_features = torch.cat((batch_features, batch_lowlevel_features), dim=1)
        else:
            batch_features = self.feature_extractor(x)
            batch_features = self.conv1(batch_features)
        all_batch_anchor_features = self.cut_anchor_features(batch_features)
        all_batch_anchor_features = all_batch_anchor_features.reshape(
            batch_size,
            time_window,
            len(self.anchors),
            self.anchor_feat_channels,
            self.fmap_h,
            1,
        )
        past_batch_anchor_features = all_batch_anchor_features[:, :-1].reshape(
            batch_size, time_window - 1, len(self.anchors), -1
        )

        batch_anchor_features = all_batch_anchor_features[:, -1]

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )

        # Compute attention features
        attention_features = self.attention_mechanism(batch_anchor_features, batch_size)
        if self.use_past_attention:
            past_attention_features = torch.zeros_like(past_batch_anchor_features)
            for t in range(time_window - 1):
                th_attention = self.attention_mechanism(
                    past_batch_anchor_features[:, t].reshape(
                        -1, self.anchor_feat_channels * self.fmap_h
                    ),
                    batch_size,
                )
                past_attention_features[:, t] = past_attention_features[:, t].reshape(
                    batch_size, len(self.anchors), -1
                )

        # Join global and local features
        past_batch_anchor_features = torch.permute(
            past_batch_anchor_features, dims=(0, 2, 1, 3)
        )
        past_batch_anchor_features = past_batch_anchor_features.reshape(
            batch_size * len(self.anchors), -1
        )
        batch_anchor_features = batch_anchor_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )
        if self.use_past_attention:
            past_attention_features = torch.permute(
                past_attention_features, dims=(0, 2, 1, 3)
            )
            past_attention_features = past_attention_features.reshape(
                batch_size * len(self.anchors), -1
            )
            batch_anchor_features = torch.cat(
                (
                    past_attention_features,
                    past_batch_anchor_features,
                    attention_features,
                    batch_anchor_features,
                ),
                dim=1,
            )
        else:
            batch_anchor_features = torch.cat(
                (past_batch_anchor_features, attention_features, batch_anchor_features),
                dim=1,
            )
        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)
        if self.cat_cls_layer is None:
            cat_cls_logits = torch.empty_like(cls_logits)
        else:
            cat_cls_logits = self.cat_cls_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
        reg = reg.reshape(batch_size, -1, reg.shape[1])
        cat_cls_logits = cat_cls_logits.reshape(batch_size, -1, cat_cls_logits.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros(
            (*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device
        )
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 3:] += reg

        # Apply nms
        proposals_list = self.nms(
            reg_proposals, cat_cls_logits, nms_thres, nms_topk, conf_threshold
        )

        return proposals_list

    def nms(
        self, batch_proposals, batch_cat_cls_logits, nms_thres, nms_topk, conf_threshold
    ):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, cat_cls_logits in zip(batch_proposals, batch_cat_cls_logits):
            anchor_inds = torch.arange(
                batch_proposals.shape[1], device=proposals.device
            )
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append(
                        (proposals[[]], self.anchors[[]], cat_cls_logits[[]])
                    )
                    continue
                keep, num_to_keep, _ = nms(
                    proposals, scores, overlap=nms_thres, top_k=nms_topk
                )
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            cat_cls_logits = cat_cls_logits[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], cat_cls_logits))

        return proposals_list

    def loss(
        self,
        proposals_list,
        targets,
        cls_loss_weight=1,
        reg_loss_weight=100,
        cat_cls_loss_weight=1,
    ):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        cat_cls_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, cat_cls_logits), target in zip(
            proposals_list, targets
        ):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 0] == 0]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                (
                    positives_mask,
                    invalid_offsets_mask,
                    negatives_mask,
                    target_positives_indices,
                ) = match_proposals_with_targets(self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            cat_cls_logits = cat_cls_logits[positives_mask]
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get lane existence classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.0
            cls_pred = all_proposals[:, :2]

            # Get lane category classification targets
            cat_cls_target = target[target_positives_indices][:, 1].long()

            # Regression targets
            reg_pred = positives[:, 3:]
            reg_pred[:, 2:] = reg_pred[:, 2:] / self.img_w
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = torch.clamp(
                    (target[:, 3] * self.n_strips).round().long(),
                    max=self.n_offsets - 1,
                    min=0,
                )
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = torch.clamp(
                    (target[:, 4] * self.n_strips).round().long(),
                    max=self.n_offsets - 1,
                    min=0,
                )
                ends[positive_starts > ends] = positive_starts[positive_starts > ends]
                invalid_offsets_mask = torch.zeros(
                    (num_positives, 1 + 1 + self.n_offsets + 1), dtype=torch.int
                )  # y_start + y_end + S + pad
                invalid_offsets_mask[all_indices, 2 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 2 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, :2] = False
                reg_target = target[:, 3:]
                reg_target[:, 2:] = reg_target[:, 2:] / self.img_w
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives
            if self.cat_cls_layer is not None:
                cat_cls_loss += self.cat_cls_layer.loss(cat_cls_logits, cat_cls_target)

        # Batch mean
        cls_loss = cls_loss_weight * cls_loss / valid_imgs
        reg_loss = reg_loss_weight * reg_loss / valid_imgs
        cat_cls_loss = cat_cls_loss_weight * cat_cls_loss / valid_imgs

        loss = cls_loss + reg_loss + cat_cls_loss
        return loss, {
            "cat_cls_loss": cat_cls_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "batch_positives": total_positives,
        }

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip(
            (self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,)
        )
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(
            -1, 1
        )
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(
            n_proposals, n_fmaps, fmaps_h
        )
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = (
            torch.arange(n_fmaps)
            .repeat_interleave(fmaps_h)
            .repeat(n_proposals)[:, None]
        )

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros(
            (batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device
        )

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(
                n_proposals, n_fmaps, self.fmap_h, 1
            )
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(
            self.left_angles, x=0.0, nb_origins=lateral_n
        )
        right_anchors, right_cut = self.generate_side_anchors(
            self.right_angles, x=1.0, nb_origins=lateral_n
        )
        bottom_anchors, bottom_cut = self.generate_side_anchors(
            self.bottom_angles, y=1.0, nb_origins=bottom_n
        )

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat(
            [left_cut, bottom_cut, right_cut]
        )

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1.0, 0.0, num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1.0, 0.0, num=nb_origins)]
        else:
            raise Exception(
                "Please define exactly one of `x` or `y` (not neither nor both)"
            )

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.0  # degrees to radians
        start_x, start_y = start
        anchor[2] = start_x
        anchor[3] = round((1 - start_y) * self.n_strips) / self.n_strips
        anchor[5:] = (
            start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)
        ) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(
                    img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5
                )

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals, cat_cls):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane, cat_pred in zip(proposals, cat_cls):
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[3].item() * self.n_strips))
            end = int(round(lane[4].item() * self.n_strips))
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            # mask = ~((((lane_xs[:start] >= 0.) &
            #            (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1 :] = -2
            lane_xs[:start] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1
            ).squeeze(2)
            metadata = {"start_x": lane[2], "start_y": lane[3], "conf": lane[1]}
            if self.cat_cls_layer is not None:
                metadata["category"] = (
                    self.cat_cls_layer.decode_prediction(cat_pred).cpu().item()
                )
            lane = Lane(points=points.cpu().numpy(), metadata=metadata)
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, cat_cls in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals, cat_cls)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == "resnet34":
        backbone = resnet34(pretrained=pretrained)
        fmap_c = 512
        stride = 32
    elif backbone == "resnet18":
        backbone = resnet18(pretrained=pretrained)
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError("Backbone not implemented: `{}`".format(backbone))

    return backbone, fmap_c, stride
