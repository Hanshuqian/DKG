import torch
import numpy as np
import medpy.metric.binary as medmetric

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(1,2))
    # same for the ground truth
    # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(1,2))
    # same for the intersection
    intersection = np.sum(intersection.astype(float), axis=(1,2))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_numpy_medpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    return medmetric.dc(binary_segmentation, binary_gt_label)


    # if get_hd:
    #     if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
    #         return medmetric.assd(binary_segmentation, binary_gt_label)
    #         # return medmetric.hd(binary_segmentation, binary_gt_label)
    #     else:
    #         return np.nan
    # else:
    #     return 0.0


def hd_numpy(binary_segmentation, binary_gt_label, get_hd):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    if get_hd:
        if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
            return medmetric.assd(binary_segmentation, binary_gt_label)
        else:
            return np.nan
    else:
        return 0.0


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    # print target.shape
    # print pred.shape
    return dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), dice_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...])


def DiceLoss(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
