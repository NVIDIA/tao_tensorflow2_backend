"""EfficientDet losses."""
import numpy as np
import tensorflow as tf
import math
from typing import Union, Text
from cv.efficientdet.model import anchors
FloatType = Union[tf.Tensor, float, np.float32, np.float64]


class FocalLoss(tf.keras.losses.Loss):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    """

    def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
        """Initialize focal loss.

        Args:
            alpha: A float32 scalar multiplying alpha to the loss from positive
                examples and (1-alpha) to the loss from negative examples.
            gamma: A float32 scalar modulating loss from hard and easy examples.
            label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
            **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    @tf.autograph.experimental.do_not_convert
    def call(self, y, y_pred):
        """Compute focal loss for y and y_pred.

        Args:
            y: A tuple of (normalizer, y_true), where y_true is the target class.
            y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

        Returns:
            the focal loss.
        """
        normalizer, y_true = y
        alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        pred_prob = tf.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t)**gamma

        # apply label smoothing for cross_entropy for each entry.
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # compute the final loss and return
        return alpha_factor * modulating_factor * ce / normalizer


class StableFocalLoss(tf.keras.losses.Loss):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Below are comments/derivations for computing modulator.
    For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    for positive samples and 1 - sigmoid(x) for negative examples.

    The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    computation. For r > 0, it puts more weights on hard examples, and less
    weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    its back-propagation is not stable when r < 1. The implementation here
    resolves the issue.

    For positive samples (labels being 1),
         (1 - p_t)^r
     = (1 - sigmoid(x))^r
     = (1 - (1 / (1 + exp(-x))))^r
     = (exp(-x) / (1 + exp(-x)))^r
     = exp(log((exp(-x) / (1 + exp(-x)))^r))
     = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
     = exp(- r * x - r * log(1 + exp(-x)))

    For negative samples (labels being 0),
         (1 - p_t)^r
     = (sigmoid(x))^r
     = (1 / (1 + exp(-x)))^r
     = exp(log((1 / (1 + exp(-x)))^r))
     = exp(-r * log(1 + exp(-x)))

    Therefore one unified form for positive (z = 1) and negative (z = 0)
    samples is: (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    """

    def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
        """Initialize focal loss.

        Args:
            alpha: A float32 scalar multiplying alpha to the loss from positive
                examples and (1-alpha) to the loss from negative examples.
            gamma: A float32 scalar modulating loss from hard and easy examples.
            label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
            **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    @tf.autograph.experimental.do_not_convert
    def call(self, y, y_pred):
        """Compute focal loss for y and y_pred.

        Args:
            y: A tuple of (normalizer, y_true), where y_true is the target class.
            y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

        Returns:
            the focal loss.
        """
        normalizer, y_true = y
        alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

        positive_label_mask = tf.equal(y_true, 1.0)
        negative_pred = -1.0 * y_pred
        modulator = tf.exp(gamma * y_true * negative_pred - gamma * tf.math.log1p(tf.exp(negative_pred)))

        # apply label smoothing for cross_entropy for each entry.
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        loss = modulator * ce
        weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
        weighted_loss /= normalizer
        return weighted_loss


class BoxLoss(tf.keras.losses.Loss):
    """L2 box regression loss."""

    def __init__(self, delta=0.1, **kwargs):
        """Initialize box loss.

        Args:
            delta: `float`, the point where the huber loss function changes from a
                quadratic to linear. It is typically around the mean value of regression
                target. For instances, the regression targets of 512x512 input with 6
                anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
            **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.huber = tf.keras.losses.Huber(delta, reduction=tf.keras.losses.Reduction.NONE)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, box_outputs):
        """Call."""
        num_positives, box_targets = y_true
        normalizer = num_positives * 4.0
        mask = tf.cast(box_targets != 0.0, tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self.huber(box_targets, box_outputs) * mask
        box_loss = tf.reduce_sum(box_loss)
        box_loss /= normalizer
        return box_loss


class BoxIouLoss(tf.keras.losses.Loss):
    """Box iou loss."""

    def __init__(self, iou_loss_type, min_level, max_level, num_scales,
                 aspect_ratios, anchor_scale, image_size, **kwargs):
        """Init BoxIOU Loss."""
        super().__init__(**kwargs)
        self.iou_loss_type = iou_loss_type
        self.input_anchors = anchors.Anchors(min_level, max_level, num_scales,
                                             aspect_ratios, anchor_scale,
                                             image_size)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, box_outputs):
        """Call."""
        anchor_boxes = tf.tile(
            self.input_anchors.boxes,
            [box_outputs.shape[0] // self.input_anchors.boxes.shape[0], 1])
        num_positives, box_targets = y_true
        mask = tf.cast(box_targets != 0.0, box_targets.dtype)
        box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes) * mask
        box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes) * mask
        normalizer = num_positives * 4.0
        box_iou_loss = iou_loss(box_outputs, box_targets, self.iou_loss_type)
        box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
        return box_iou_loss


def _get_v(b1_height: FloatType, b1_width: FloatType, b2_height: FloatType,
           b2_width: FloatType) -> tf.Tensor:
    """Get the consistency measurement of aspect ratio for ciou."""

    @tf.custom_gradient
    def _get_grad_v(height, width):
        """backpropogate gradient."""
        arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
            tf.math.divide_no_nan(width, height))
        v = 4 * ((arctan / math.pi)**2)

        def _grad_v(dv):
            """Grad for eager mode."""
            gdw = dv * 8 * arctan * height / (math.pi**2)
            gdh = -dv * 8 * arctan * width / (math.pi**2)
            return [gdh, gdw]

        def _grad_v_graph(dv, variables):
            """Grad for graph mode."""
            gdw = dv * 8 * arctan * height / (math.pi**2)
            gdh = -dv * 8 * arctan * width / (math.pi**2)
            return [gdh, gdw], tf.gradients(v, variables, grad_ys=dv)

        if tf.compat.v1.executing_eagerly_outside_functions():
            return v, _grad_v
        return v, _grad_v_graph

    return _get_grad_v(b2_height, b2_width)


def _iou_per_anchor(pred_boxes: FloatType,
                    target_boxes: FloatType,
                    iou_type: Text = 'iou') -> tf.Tensor:
    """Computing the IoU for a single anchor.

    Args:
        pred_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max].
        target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max].
        iou_type: one of ['iou', 'ciou', 'diou', 'giou'].

    Returns:
        IoU loss float `Tensor`.
    """
    # t_ denotes target boxes and p_ denotes predicted boxes.
    t_ymin, t_xmin, t_ymax, t_xmax = target_boxes
    p_ymin, p_xmin, p_ymax, p_xmax = pred_boxes

    zero = tf.convert_to_tensor(0.0, t_ymin.dtype)
    p_width = tf.maximum(zero, p_xmax - p_xmin)
    p_height = tf.maximum(zero, p_ymax - p_ymin)
    t_width = tf.maximum(zero, t_xmax - t_xmin)
    t_height = tf.maximum(zero, t_ymax - t_ymin)
    p_area = p_width * p_height
    t_area = t_width * t_height

    intersect_ymin = tf.maximum(p_ymin, t_ymin)
    intersect_xmin = tf.maximum(p_xmin, t_xmin)
    intersect_ymax = tf.minimum(p_ymax, t_ymax)
    intersect_xmax = tf.minimum(p_xmax, t_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = p_area + t_area - intersect_area
    iou_v = tf.math.divide_no_nan(intersect_area, union_area)
    if iou_type == 'iou':
        return iou_v  # iou is the simplest form.

    enclose_ymin = tf.minimum(p_ymin, t_ymin)
    enclose_xmin = tf.minimum(p_xmin, t_xmin)
    enclose_ymax = tf.maximum(p_ymax, t_ymax)
    enclose_xmax = tf.maximum(p_xmax, t_xmax)

    assert iou_type in ('giou', 'diou', 'ciou')
    if iou_type == 'giou':  # giou is the generalized iou.
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        giou_v = iou_v - tf.math.divide_no_nan(
            (enclose_area - union_area), enclose_area)
        return giou_v

    assert iou_type in ('diou', 'ciou')
    p_center = tf.stack([(p_ymin + p_ymax) / 2, (p_xmin + p_xmax) / 2], axis=-1)
    t_center = tf.stack([(t_ymin + t_ymax) / 2, (t_xmin + t_xmax) / 2], axis=-1)
    euclidean = tf.linalg.norm(t_center - p_center, axis=-1)
    diag_length = tf.linalg.norm(
        tf.stack(
            [enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin],
            axis=-1),
        axis=-1)
    diou_v = iou_v - tf.math.divide_no_nan(euclidean**2, diag_length**2)
    if iou_type == 'diou':  # diou is the distance iou.
        return diou_v

    assert iou_type == 'ciou'
    v = _get_v(p_height, p_width, t_height, t_width)
    alpha = tf.math.divide_no_nan(v, ((1 - iou_v) + v))
    return diou_v - alpha * v  # the last one is ciou.


def iou_loss(pred_boxes: FloatType,
             target_boxes: FloatType,
             iou_type: Text = 'iou') -> tf.Tensor:
    """A unified interface for computing various IoU losses.

    Let B and B_gt denotes the pred_box and B_gt is the target box (ground truth):

        IoU = |B & B_gt| / |B | B_gt|

        GIoU = IoU - |C - B U B_gt| / C, where C is the smallest box covering B and
        B_gt.

        DIoU = IoU - E(B, B_gt)^2 / c^2, E is the Euclidean distance of the center
        points of B and B_gt, and c is the diagonal length of the smallest box
        covering the two boxes

        CIoU = IoU - DIoU - a * v, where a is a positive trade-off parameter, and
        v measures the consistency of aspect ratio:
            v = (arctan(w_gt / h_gt) - arctan(w / h)) * 4 / pi^2
        where (w_gt, h_gt) and (w, h) are the width and height of the target and
        predicted box respectively.

    The returned loss is computed as 1 - one of {IoU, GIoU, DIoU, CIoU}.

    Args:
        pred_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max]*.
            It can be multiple anchors, with each anchor box has four coordinates.
        target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max]*.
            It can be multiple anchors, with each anchor box has four coordinates.
        iou_type: one of ['iou', 'ciou', 'diou', 'giou'].

    Returns:
        IoU loss float `Tensor`.
    """
    if iou_type not in ('iou', 'ciou', 'diou', 'giou'):
        raise ValueError(
            f'Unknown loss_type {iou_type}, not iou/ciou/diou/giou')

    pred_boxes = tf.convert_to_tensor(pred_boxes, tf.float32)
    target_boxes = tf.cast(target_boxes, pred_boxes.dtype)

    # t_ denotes target boxes and p_ denotes predicted boxes: (y, x, y_max, x_max)
    pred_boxes_list = tf.unstack(pred_boxes, None, axis=-1)
    target_boxes_list = tf.unstack(target_boxes, None, axis=-1)
    assert len(pred_boxes_list) == len(target_boxes_list)
    assert len(pred_boxes_list) % 4 == 0

    iou_loss_list = []
    for i in range(0, len(pred_boxes_list), 4):
        pred_boxes = pred_boxes_list[i:i + 4]
        target_boxes = target_boxes_list[i:i + 4]

        # Compute mask.
        t_ymin, t_xmin, t_ymax, t_xmax = target_boxes
        mask = tf.math.logical_and(t_ymax > t_ymin, t_xmax > t_xmin)
        mask = tf.cast(mask, t_ymin.dtype)
        # Loss should be mask * (1 - iou) = mask - masked_iou.
        pred_boxes = [b * mask for b in pred_boxes]
        target_boxes = [b * mask for b in target_boxes]
        iou_loss_list.append(
            mask *
            (1 - tf.squeeze(_iou_per_anchor(pred_boxes, target_boxes, iou_type))))
    if len(iou_loss_list) == 1:
        return iou_loss_list[0]
    return tf.reduce_sum(tf.stack(iou_loss_list), 0)
