import random

import numpy as np
import torch
import torch.nn.functional as F

from utils import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(
    linewidth=320, formatter={
        'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open('data/coco.names', 'r')
    names = fp.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters()
              if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' %
          ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print(
            '%5g %50s %9s %12g %20s %12.3g %12.3g' %
            (i,
             name,
             p.requires_grad,
             p.numel(),
             list(
                 p.shape),
                p.mean(),
                p.std()))
    print(
        'Model Summary: %g layers, %g parameters, %g gradients\n' %
        (i + 1, n_p, n_g))


def class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor([187437,
                                     4955,
                                     30920,
                                     6033,
                                     3838,
                                     4332,
                                     3160,
                                     7051,
                                     7677,
                                     9167,
                                     1316,
                                     1372,
                                     833,
                                     6757,
                                     7355,
                                     3302,
                                     3776,
                                     4671,
                                     6769,
                                     5706,
                                     3908,
                                     903,
                                     3686,
                                     3596,
                                     6200,
                                     7920,
                                     8779,
                                     4505,
                                     4272,
                                     1862,
                                     4698,
                                     1962,
                                     4403,
                                     6659,
                                     2402,
                                     2689,
                                     4012,
                                     4175,
                                     3411,
                                     17048,
                                     5637,
                                     14553,
                                     3923,
                                     5539,
                                     4289,
                                     10084,
                                     7018,
                                     4314,
                                     3099,
                                     4638,
                                     4939,
                                     5543,
                                     2038,
                                     4004,
                                     5053,
                                     4578,
                                     27292,
                                     4113,
                                     5931,
                                     2905,
                                     11174,
                                     2873,
                                     4036,
                                     3415,
                                     1517,
                                     4122,
                                     1980,
                                     4464,
                                     1190,
                                     2302,
                                     156,
                                     3933,
                                     1877,
                                     17630,
                                     4337,
                                     4624,
                                     1075,
                                     3468,
                                     135,
                                     1380])
    weights /= weights.sum()
    return weights


# Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
def xyxy2xywh(x):
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


# Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
def xywh2xyxy(x):
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
            tp:    True positives (list).
            conf:  Objectness value from 0-1 (list).
            pred_cls: Predicted object classes (list).
            target_cls: True object classes (list).
    # Returns
            The average precision.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(
        conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype(
        'int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
    # Returns
            The average precision.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets(pred_boxes, pred_conf, pred_cls, targets, anchor_wh,
                  anchor_number, class_number, grid_number, batch_report):
    """
    returns targets_per_image, class_numberorrect, tx, ty, tw, th, tconf, tcls
    """
    batch_size = len(targets)  # number of images in batch
    targets_per_image = [len(x) for x in targets]

    shape = (batch_size, anchor_number, grid_number, grid_number)
    tx = torch.zeros(shape)
    ty = torch.zeros(shape)
    tw = torch.zeros(shape)
    th = torch.zeros(shape)

    tconf = torch.ByteTensor(
        batch_size,
        anchor_number,
        grid_number,
        grid_number).fill_(0)
    tcls = torch.ByteTensor(
        batch_size,
        anchor_number,
        grid_number,
        grid_number,
        class_number).fill_(0)

    TP = torch.ByteTensor(batch_size, max(targets_per_image)).fill_(0)
    FP = torch.ByteTensor(batch_size, max(targets_per_image)).fill_(0)
    FN = torch.ByteTensor(batch_size, max(targets_per_image)).fill_(0)
    TC = torch.ShortTensor(batch_size, max(
        targets_per_image)).fill_(-1)  # target category

    for b in range(batch_size):
        target_number = targets_per_image[b]

        if target_number == 0:
            continue

        target = targets[b]  # target is targets in one image

        # if batch_report:
        # 	FN[b, :target_number] = 1

        # Record target category in TC
        TC[b, :target_number] = target[:, 0].long()

        # gx, gy, gw, gh are scaled target x, y, w, h in the output feature
        gx, gy, gw, gh = target[:, 1] * grid_number, target[:, 2] * \
            grid_number, target[:, 3] * grid_number, target[:, 4] * grid_number

        # Get grid box indices and prevent overflows
        # self.long() is equivalent to self.to(torch.int64), namely convert gx
        # and gy to integar coordinates
        gi = torch.clamp(gx.long(), min=0, max=grid_number - 1)
        gj = torch.clamp(gy.long(), min=0, max=grid_number - 1)

        # iou between targets and anchors (only use w and h)
        # box1 has shape [target_number, 2]
        box1 = target[:, 3:5] * grid_number

        # anchor_wh has shape [3, 2], use unsqueeze to add one dimension at axis 1, then becomes [3, 1, 2]
        # repeat (1, target_number, 1), then shape becomes [3, target_number, 2]
        # box2 = anchor_wh.unsqueeze(1).repeat(1, target_number, 1)
        box2 = anchor_wh.unsqueeze(1)

        # min function firstly enlarge box1 to [3, target_number, 2], and finally output [3, target_number, 2]
        # prod returns the product of each row of the given dimension
        # inter_area has shape [3, target_number]
        inter_area = torch.min(box1, box2).prod(2)
        iou_anchor_target = inter_area / \
            (gw * gh + box2.prod(2) - inter_area + 1e-16)

        # Select best anchor for each target, a is a vector containing the
        # index of best anchors
        iou_anchor_best, a = iou_anchor_target.max(0)

        # Select best unique target-anchor combinations
        if target_number > 1:
            # np.argsort returns the indices that would sort an array.
            # iou_order = np.argsort(-iou_anchor_best)  # best to worst
            iou_order = torch.argsort(-iou_anchor_best)

            # Unique anchor selection (slower but retains original order)
            # gi, gj, a have same shape (target_number, 1)
            # u = torch.cat((gi, gj, a), 0).view(3, -1).numpy()
            u = torch.cat((gi, gj, a), 0).view(3, -1)

            # u has shape (3, target_number), use iou_order to sort the second dimension
            # np.unique with axis=1 here are used to find the index of unique columns
            # np.unique will change the matrix but we don't use the matrix
            # the key idea is when different bbox are matched to the same anchor, we use the one with larger iou
            # first_unique_index has shape (target_number - identical number,
            # 1)
            ordered_u = u[:, iou_order]
            _, first_unique_index = np.unique(
                ordered_u, axis=1, return_index=True)  # first unique indices

            # ordered_u is sorted by iou_order, so we need to use i = iou_order[first_unique_index]
            # to recover and match the sequence in iou_anchor_best
            i = iou_order[first_unique_index]

            # best anchors must share good iou with target bbox
            # else we assume it looks too unsimilar to anchors and thus not a good bbox to be used during training
            # iou_anchor_best is a vector of index of target bbox
            i = i[iou_anchor_best[i] > 0.10]

            if len(i) == 0:
                continue

            a, gj, gi, target = a[i], gj[i], gi[i], target[i]

            # if only one bbox left, we need to expand target to two-dimension
            # to be used later
            if len(target.shape) == 1:
                target = target.view(1, 5)
        else:
            if iou_anchor_best < 0.10:
                continue
            i = 0

        # since some bbox in target are not matched to anchors due to low iou,
        # we need to update tc, gx, gy, gw and gh.
        tc, gx, gy, gw, gh = target[:, 0].long(), target[:, 1] * grid_number, target[:, 2] \
            * grid_number, target[:, 3] * grid_number, target[:, 4] * grid_number

        # The key idea is to fit the groundtruth gx, gy, gw, gh to anchors,
        # in order to find out the ground truth tx, ty, tw, th for our
        # anchor-based outputs

        # Coordinates
        # tx, ty, tw, th are the ground truths that used to calculate loss
        # coordinates not related to ground truth bbox should keep value 0.
        # a is the vector of selected anchors
        # tx and ty are with shape [8, 3, 13, 13]
        tx[b, a, gj, gi] = gx - gi.float()
        ty[b, a, gj, gi] = gy - gj.float()

        # Width and height (yolo method)
        tw[b, a, gj, gi] = torch.log(gw / anchor_wh[a, 0])
        th[b, a, gj, gi] = torch.log(gh / anchor_wh[a, 1])

        # Width and height (power method)
        # tw[b, a, gj, gi] = torch.sqrt(gw / anchor_wh[a, 0]) / 2
        # th[b, a, gj, gi] = torch.sqrt(gh / anchor_wh[a, 1]) / 2

        # One-hot encoding of label
        # tcls has shape [8, 3, 13, 13, 80]
        # tconf has shape [8, 3, 13, 13]
        tcls[b, a, gj, gi, tc] = 1

        # tconf is used as mask later
        # tconf = 1 means there is a target matched to an anchor at the place
        tconf[b, a, gj, gi] = 1

        if batch_report:
            tb = torch.cat((gx - gw / 2, gy - gh / 2, gx + gw / 2,
                            gy + gh / 2)).view(4, -1).t()  # target boxes

            # predicted classes and confidence
            pcls = torch.argmax(pred_cls[b, a, gj, gi], 1).cpu()
            pconf = torch.sigmoid(pred_conf[b, a, gj, gi]).cpu()
            iou_pred = bbox_iou(tb, pred_boxes[b, a, gj, gi].cpu())

            # shape (batch_size, max(targets_per_image))
            TP[b, i] = (pconf > 0.5) & (iou_pred > 0.5) & (pcls == tc)
            # coordinates or class are wrong
            FP[b, i] = (pconf > 0.5) & (TP[b, i] == 0)
            # confidence score is too low (set to zero)
            FN[b, i] = pconf <= 0.5

    return tx, ty, tw, th, tconf, tcls, TP, FP, FN, TC


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            # thresh = 0.85
            thresh = nms_thres
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i,
                                     0] - a[i + 1:,
                                            0]) < radius) & (torch.abs(a[i,
                                                                         1] - a[i + 1:,
                                                                                1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(
                        a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > thresh]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        a = w * h  # area
        ar = w / (h + 1e-16)  # aspect ratio

        log_w, log_h, log_a, log_ar = torch.log(
            w), torch.log(h), torch.log(a), torch.log(ar)

        # n = len(w)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] = multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        v = ((pred[:, 4] > conf_thres) & (class_prob > .3))
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob,
        # class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float(
        ).unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        nms_style = 'OR'  # 'AND' or 'OR' (classical)
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []

            if nms_style == 'OR':  # Classical NMS
                while detections_class.shape[0]:
                    # Get detection with highest confidence and save as max
                    # detection
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(detections_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(max_detections[-1], detections_class[1:])

                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]

            elif nms_style == 'AND':  # 'AND'-style NMS, at least two boxes must share commonality to pass, single boxes erased
                while detections_class.shape[0]:
                    if len(detections_class) == 1:
                        break

                    ious = bbox_iou(detections_class[:1], detections_class[1:])

                    if ious.max() > 0.5:
                        max_detections.append(detections_class[0].unsqueeze(0))

                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]

            if len(max_detections) > 0:
                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None else torch.cat(
                    (output[image_i], max_detections))

    return output
