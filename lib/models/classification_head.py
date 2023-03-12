import torch
import torch.nn as nn
from lib.datasets.vil100 import (
    ATTRIBUTES_LABELS,
    ATTRIBUTE_TO_CAT_ID,
    ALL_ATTRIBUTES,
    CAT_ID_TO_ATTRIBUTE,
)

LABEL_TO_CAT_ID = {
    label: ATTRIBUTE_TO_CAT_ID[attribute]
    for label, attribute in zip(ATTRIBUTES_LABELS, ALL_ATTRIBUTES)
}
# LABEL_TO_CAT_ID['invalid'] = 20

DOUBLE_CATS = [
    LABEL_TO_CAT_ID["double white solid"],
    LABEL_TO_CAT_ID["double white solid dotted"],
    LABEL_TO_CAT_ID["double white dotted solid"],
    LABEL_TO_CAT_ID["double yellow solid"],
    LABEL_TO_CAT_ID["double yellow dotted"],
    LABEL_TO_CAT_ID["double solid white and yellow"],
]
LEFT_YELLOW_CATS = [
    LABEL_TO_CAT_ID["single yellow solid"],
    LABEL_TO_CAT_ID["single yellow dotted"],
    LABEL_TO_CAT_ID["double yellow solid"],
    LABEL_TO_CAT_ID["single yellow dotted"],
]
LEFT_DOTTED_CATS = [
    LABEL_TO_CAT_ID["single white dotted"],
    LABEL_TO_CAT_ID["single yellow dotted"],
    LABEL_TO_CAT_ID["single yellow dotted"],
    LABEL_TO_CAT_ID["double white dotted solid"],
    LABEL_TO_CAT_ID["double yellow dotted"],
]
RIGHT_YELLOW_CATS = [
    LABEL_TO_CAT_ID["double solid white and yellow"],
    LABEL_TO_CAT_ID["double yellow solid"],
    LABEL_TO_CAT_ID["double yellow dotted"],
]
RIGHT_DOTTED_CATS = [
    LABEL_TO_CAT_ID["double white solid dotted"],
    LABEL_TO_CAT_ID["double yellow dotted"],
]


class FeaturePredictionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer_left = nn.Sequential(
            nn.Linear(in_features, in_features // 2), nn.ReLU()
        )
        self.layer_right = nn.Sequential(
            nn.Linear(in_features, in_features // 2), nn.ReLU()
        )
        self.layer_double = nn.Linear(in_features, 2)  # single, double
        self.layer_color = nn.Linear(in_features // 2, 2)  # white, yellow
        self.layer_filling = nn.Linear(in_features // 2, 2)  # solid, dotted
        self.mapping = torch.zeros((2, 2, 2, 2, 2), dtype=torch.int8)
        # self.mapping[:] = LABEL_TO_CAT_ID['invalid']
        self.mapping[0, 0, 0] = LABEL_TO_CAT_ID["single white solid"]
        self.mapping[0, 0, 1] = LABEL_TO_CAT_ID["single white dotted"]
        self.mapping[0, 1, 0] = LABEL_TO_CAT_ID["single yellow solid"]
        self.mapping[0, 1, 1] = LABEL_TO_CAT_ID["single yellow dotted"]
        self.mapping[1, 0, 0, 0, 0] = LABEL_TO_CAT_ID["double white solid"]
        self.mapping[1, 0, 0, 0, 1] = LABEL_TO_CAT_ID["double white solid dotted"]
        self.mapping[1, 0, 1, 0, 0] = LABEL_TO_CAT_ID["double white dotted solid"]
        self.mapping[1, 1, 0, 1, 0] = LABEL_TO_CAT_ID["double yellow solid"]
        self.mapping[1, 1, 1, 1, 1] = LABEL_TO_CAT_ID["double yellow dotted"]
        self.mapping[1, 0, 0, 1, 0] = LABEL_TO_CAT_ID["double solid white and yellow"]
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        left_features = self.layer_left(x)
        left_color = self.layer_color(left_features)
        left_filling = self.layer_filling(left_features)
        right_features = self.layer_right(x)
        right_color = self.layer_color(right_features)
        right_filling = self.layer_filling(right_features)
        double_features = torch.cat((left_features, right_features), dim=1)
        double = self.layer_double(double_features)

        return torch.cat(
            (double, left_color, left_filling, right_color, right_filling), dim=1
        )

    def loss(self, prediction, label):
        double = prediction[:, [0, 1]]
        left_color, left_filling = prediction[:, [2, 3]], prediction[:, [4, 5]]
        right_color, right_filling = prediction[:, [6, 7]], prediction[:, [8, 9]]
        (
            double_target,
            left_color_target,
            left_filling_target,
            right_color_target,
            right_filling_target,
        ) = self.encode_label(label)
        double_loss = self.cross_entropy(double, double_target)
        left_color_loss = self.cross_entropy(left_color, left_color_target)
        left_filling_loss = self.cross_entropy(left_filling, left_filling_target)
        is_double = double_target == 1
        if is_double.sum() > 0:
            right_color_loss = self.cross_entropy(
                right_color[is_double], right_color_target[is_double]
            )
            right_filling_loss = self.cross_entropy(
                right_filling[is_double], right_filling_target[is_double]
            )
        else:
            right_color_loss = right_filling_loss = 0

        return (
            double_loss
            + left_color_loss
            + left_filling_loss
            + right_color_loss
            + right_filling_loss
        )

    def encode_label(self, label):
        double_target = label == DOUBLE_CATS[0]
        for cat in DOUBLE_CATS[1:]:
            double_target |= label == cat
        double_target = double_target.long()

        left_color_target = label == LEFT_YELLOW_CATS[0]
        for cat in LEFT_YELLOW_CATS[1:]:
            left_color_target |= label == cat
        left_color_target = left_color_target.long()

        left_filling_target = label == LEFT_DOTTED_CATS[0]
        for cat in LEFT_DOTTED_CATS[1:]:
            left_filling_target |= label == cat
        left_filling_target = left_filling_target.long()

        right_color_target = label == RIGHT_YELLOW_CATS[0]
        for cat in RIGHT_YELLOW_CATS[1:]:
            right_color_target |= label == cat
        right_color_target = right_color_target.long()

        right_filling_target = label == RIGHT_DOTTED_CATS[0]
        for cat in RIGHT_DOTTED_CATS[1:]:
            right_filling_target |= label == cat
        right_filling_target = right_filling_target.long()

        return (
            double_target,
            left_color_target,
            left_filling_target,
            right_color_target,
            right_filling_target,
        )

    def decode_prediction(self, prediction):
        double = prediction[[0, 1]]
        left_color, left_filling = prediction[[2, 3]], prediction[[4, 5]]
        right_color, right_filling = prediction[[6, 7]], prediction[[8, 9]]
        double = torch.argmax(double)
        left_color = torch.argmax(left_color)
        left_filling = torch.argmax(left_filling)
        right_color = torch.argmax(right_color)
        right_filling = torch.argmax(right_filling)

        return self.mapping[
            double, left_color, left_filling, right_color, right_filling
        ]


class DirectPredictionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, len(CAT_ID_TO_ATTRIBUTE)),
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def loss(self, prediction, label):
        return self.cross_entropy(prediction, label)

    def decode_prediction(self, prediction):
        return torch.argmax(prediction)
