def locate(boxes, scene_width, scene_height, v_point=.3, h_point=.3, horizontal_only=True):
    """
    :param boxes: array-like object of shape (n_boxes, 4) with each element containing [ymin, xmin, ymax, xmax] coordinates of bounding boxes
    :param scene_width: the pixel-width of the scene to be analyzed
    :param scene_height: the pixel-height of the scene to be analyzed
    :param v_point: the portion of a box's width, as a threshold to determine its vertical location
    :param h_point: the portion of a box's height, as a threshold to determine its horizontal location
    :param horizontal_only: defaults to True, if False, horizontal location of the boxes will be determined as well
    :return: an array of [vertical, horizontal] location of each box, 0 for left (above), 1 for right (bottom), None in horizontal index if lr_only=True
    """
    where = []
    v_margin = scene_width / 2
    for box in boxes:
        placement = [None, None]
        xmin, xmax = box[1], box[3]
        if xmin <= v_margin and xmax <= v_margin:
            placement[0] = 0
        elif xmin >= v_margin and xmax >= v_margin:
            placement[0] = 2
        else:
            width = xmax - xmin
            threshold = width * v_point
            if v_margin - xmin >= threshold:
                placement[0] = 0
            elif xmax - v_margin >= threshold:
                placement[0] = 2
            else:
                placement[0] = 1
        where.append(placement)
    if not horizontal_only:
        h_margin = scene_height / 2
        for i, box in enumerate(boxes):
            ymin, ymax = box[0], box[2]
            if ymin <= h_margin and ymax <= h_margin:
                where[i][1] = 0
            elif ymin >= h_margin and ymax >= h_margin:
                where[i][1] = 2
            else:
                height = ymax - ymin
                threshold = height * h_point
                if v_margin - ymin >= threshold:
                    where[i][1] = 0
                elif ymax - v_margin >= threshold:
                    where[i][1] = 2
                else:
                    where[i][1] = 1
    return where
