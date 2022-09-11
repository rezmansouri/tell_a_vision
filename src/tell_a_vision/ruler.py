import numpy as np


class Ruler:
    def __init__(self, images: list, class_labels: list, coords_key='box', class_key='class', xmin_key='x1', ymin_key='y1',
                 xmax_key='x2', ymax_key='y2'):
        """
        fit the ruler on a collection of bounding box annotations for each class to get quantiles of the area of objects
        :param images: list of dicts where each one corresponds to bounding box annotations of an image i.e. [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param class_labels: list of class labels
        :param coords_key: alternative key for 'box' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param class_key: alternative key for 'class' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param xmin_key: alternative key for 'x1' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param ymin_key: alternative key for 'y1' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param xmax_key: alternative key for 'x2' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        :param ymax_key: alternative key for 'y2' in [{'box':{'x1': 0, 'y1': 10, 'x2': 10, 'y2':20}, 'class': 'car'}]
        """
        self._quantiles = {c: [] for c in class_labels}
        self._class_labels = class_labels
        for image in images:
            for obj in image:
                coords = obj[coords_key]
                class_ = obj[class_key]
                w = coords[xmin_key] - coords[xmax_key]
                h = coords[ymin_key] - coords[ymax_key]
                self._quantiles[class_].append(w * h)
        for c in self._quantiles:
            self._quantiles[c] = np.quantile(self._quantiles[c], [.25, .5, .75])

    def get_ranks(self, boxes, classes):
        """
        :param boxes: array-like object of shape (n_boxes, 4) with each element containing [ymin, xmin, ymax, xmax] coordinates of bounding boxes
        :param classes: array-like object of shape (n_boxes, ) with each element corresponding to the index of the object's class in class labels
        :return: an array of objects' ranks where a rank corresponds to the quantile interval of the object's size among the objects fitted in initialization -> 0, 1, 2, or 3
        """
        ranks = []
        for box, class_index in zip(boxes, classes):
            ymin, xmin, ymax, xmax = box
            class_ = self._class_labels[class_index]
            size = (ymax - ymin) * (xmax - xmin)
            ranks.append(np.searchsorted(self._quantiles[class_], size))
        return ranks
