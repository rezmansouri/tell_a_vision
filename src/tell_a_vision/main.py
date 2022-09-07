import numpy as np


def locate(boxes, scene_width, scene_height, v_point=.66, h_point=.66, horizontal_only=True):
    """
    :param boxes: array-like object of shape (n_boxes, 4) with each element containing [ymin, xmin, ymax, xmax] coordinates of bounding boxes
    :param scene_width: the pixel-width of the scene to be analyzed
    :param scene_height: the pixel-height of the scene to be analyzed
    :param v_point: the portion of a box's width, as a threshold to determine its vertical location
    :param h_point: the portion of a box's height, as a threshold to determine its horizontal location
    :param horizontal_only: defaults to True, if False, horizontal location of the boxes will be determined as well
    :return: an array of [vertical, horizontal] location of each box, 0 for left (above), 1 for middle (midst), 2 for right (bottom), and None in horizontal index if horizontal_only=True
    """
    where = np.ones((boxes.shape[0], 2))

    v_margin = scene_width / 2
    both_left = (boxes[:, 1] <= v_margin) & (boxes[:, 3] <= v_margin)
    both_right = np.full(both_left.shape, False)
    both_right[~both_left] = (boxes[~both_left, 1] >= v_margin) & (boxes[~both_left, 3] >= v_margin)
    v_threshold = (boxes[:, 3] - boxes[:, 1]) * h_point
    left_h_margin = v_margin - v_threshold
    left = np.full(both_left.shape, False)
    left[~both_left & ~both_right] = boxes[~both_left & ~both_right, 1] < left_h_margin[~both_left & ~both_right]
    right_h_margin = v_margin + v_threshold
    right = np.full(both_left.shape, False)
    right[~both_left & ~both_right & ~left] = boxes[~both_left & ~both_right & ~left, 3] > right_h_margin[
        ~both_left & ~both_right & ~left]
    where[np.where(both_left | left), 0] = 0
    where[np.where(both_right | right), 0] = 2

    if not horizontal_only:
        v_margin = scene_height / 2
        both_above = (boxes[:, 0] <= v_margin) & (boxes[:, 2] <= v_margin)
        both_bottom = np.full(both_above.shape, False)
        both_bottom[~both_above] = (boxes[~both_above, 0] >= v_margin) & (boxes[~both_above, 2] >= v_margin)
        v_threshold = (boxes[:, 2] - boxes[:, 0]) * v_point
        bottom_v_margin = v_margin - v_threshold
        above = np.full(both_above.shape, False)
        above[~both_above & ~both_bottom] = boxes[~both_above & ~both_bottom, 0] < bottom_v_margin[~both_above & ~both_bottom]
        above_v_margin = v_margin + v_threshold
        bottom = np.full(both_bottom.shape, False)
        bottom[~both_above & ~both_bottom & ~bottom] = boxes[~both_above & ~both_bottom & ~bottom, 2] > above_v_margin[
            ~both_above & ~both_bottom & ~bottom]
        where[np.where(both_above | above), 1] = 0
        where[np.where(both_bottom | bottom), 1] = 2
    else:
        where[..., 1] = None

    return where
