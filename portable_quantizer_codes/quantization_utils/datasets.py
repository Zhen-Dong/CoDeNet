import math
import os
import random

import cv2
import numpy as np
import torch

from utils.utils import xyxy2xywh


class load_images_and_labels():  # for training
    def __init__(
            self,
            path,
            batch_size,
            img_size,
            multi_scale=False,
            augment=True):
        # for MS COCO training, file is trainvalno5k.txt
        self.path = path
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]
        # we edit image_path to become label_path and get labels (They are in
        # different file directories)
        self.label_files = [
            path.replace(
                'images',
                'labels').replace(
                '.png',
                '.txt').replace(
                '.jpg',
                '.txt') for path in self.img_files]

        self.images_number = len(self.img_files)
        self.batch_number = math.ceil(self.images_number / batch_size)
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.batch_number > 0, 'No images found in path %s' % path

    def __iter__(self):
        self.count = -1
        # if augmentation, then we shuffle the index
        # np.random.permutation can generate a random index sequence
        self.shuffled_vector = np.random.permutation(self.images_number) \
            if self.augment else np.arange(self.images_number)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.batch_number:
            raise StopIteration

        # i_start and i_end are the start and end index of this batch
        i_start = self.count * self.batch_size
        i_end = min((self.count + 1) * self.batch_size, self.images_number)

        if self.multi_scale:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            # Fixed-Scale YOLO Training
            height = self.height

        img_all = []
        labels_all = []

        # index is current image index in this batch
        # file_index is for the real img_path and label_path and thus is
        # shuffled (if we use augmentation)
        for index, files_index in enumerate(range(i_start, i_end)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            label_path = self.label_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR format
            if img is None:
                continue

            augment_hsv = True

            # Convert to HSV image
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                # here dst=img means img = output of cvtColor
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            # Resize Square
            h, w, _ = img.shape
            img, ratio, padw, padh = resize_square(
                img, height=height, color=(127.5, 127.5, 127.5))

            # Load labels in a single image and then convert format
            if os.path.isfile(label_path):
                # Use reshape(-1, 5) to make labels0 has 5 columns, number of
                # rows is automatically computed
                labels0 = np.loadtxt(
                    label_path, dtype=np.float32).reshape(-1, 5)

                # Normalized xywh to pixel xyxy format
                # labels is a 2-dimensional array
                # here padw and padh have already been divided by 2 in
                # resize-square function
                labels = labels0.copy()
                labels[:, 1] = ratio * w * \
                    (labels0[:, 1] - labels0[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * \
                    (labels0[:, 2] - labels0[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * \
                    (labels0[:, 1] + labels0[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * \
                    (labels0[:, 2] + labels0[:, 4] / 2) + padh
            else:
                labels = np.array([])

            # Augment Image and Labels
            if self.augment:
                img, labels, M = random_affine(
                    img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

            plotFlag = False
            if plotFlag:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10)) if index == 0 else None
                # change channels to RGB to be plotted
                plt.subplot(4, 4, index + 1).imshow(img[:, :, ::-1])
                # .T is the same as self.transpose()
                # after transpose, combinations are (x1,y1), (x2,y1), (x2,y2),
                # (x1,y2), (x1,y1)
                plt.plot(labels[:, [1, 3, 3, 1, 1]].T,
                         labels[:, [2, 2, 4, 4, 2]].T, '.-')
                plt.axis('off')
                plt.show()

            labels_number = len(labels)

            # convert xyxy labels to xywh labels
            if labels_number > 0:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / height

            # random left-right flip and up-down flip
            if self.augment:
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    if labels_number > 0:
                        # change the x coordinate
                        labels[:, 1] = 1 - labels[:, 1]

                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    if labels_number > 0:
                        # change the y coordinate
                        labels[:, 2] = 1 - labels[:, 2]

            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        # BGR to RGB using ::-1
        # transpose function makes img_all to be Batch-Channal-X-Y shaped,
        # which is required by PyTorch
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # Return a contiguous array in memory (C order)
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

    def __len__(self):
        return self.batch_number  # number of batches


# resize a rectangular image to a padded square
def resize_square(img, height=416, color=(0, 0, 0)):
    # these operations make the large one between height and width in the original image
    # become height-we-want after resize
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]

    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    # the sum of top padding and bottom padding is height padding
    top, bottom = dh // 2, dh - (dh // 2)
    # the sum of left padding and right padding is width padding
    left, right = dw // 2, dw - (dw // 2)

    img = cv2.resize(
        img,
        (new_shape[1],
         new_shape[0]),
        interpolation=cv2.INTER_AREA)  # resized, no border

    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1),
                  scale=(.9, 1.1), shear=(-2, 2), borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    # np.eye() returns a diagonal matrix
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small
    # rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
        img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
        img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    # warpPerspective is a form of augmentation that combines S, T, R, and
    # overall change the perspective
    imw = cv2.warpPerspective(
        img,
        M,
        dsize=(
            height,
            height),
        flags=cv2.INTER_LINEAR,
        borderValue=borderValue)  # BGR order borderValue

    # targets are groundtruth labels
    # calculate labels after augmentation
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            # xy is a matrix with (4*number of bbox in this image) rows
            # each four rows contains (x1,y1), (x2,y2), (x1,y2), (x2,y1)
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            # xy is a matrix with (number of bbox) rows
            # each row contains 4 values, x_min, y_min, x_max, y_max, which
            # describle the bbox after augmentation M
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(
                4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)),
                            abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy is a matrix with (number of bbox) rows
            # each row contains 4 values, x_min, y_min, x_max, y_max, which
            # describle the bbox after angle-based reduction
            xy = np.concatenate(
                (x -
                 w /
                 2,
                 y -
                 h /
                 2,
                 x +
                 w /
                 2,
                 y +
                 h /
                 2)).reshape(
                4,
                n).T

            # clip points in bbox that are outside the image after augmentation
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            # i is a vector, length is the number of bbox in this image
            # i is used to determine if a bbox after augmentation is a good one
            # for training
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            # we get rid of bad bbox in the augmented image for training
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets, M
    else:
        return imw
