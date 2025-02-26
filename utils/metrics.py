import torch


def dice_coefficient(pred, gt, smooth=1e-5):
    """
    computational formula:
        dice = 2TP/(FP + 2TP + FN)
    """

    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # if (pred.sum() + gt.sum()) == 0:
    #     return 1
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient(pred, gt, smooth=1e-5):
    """
    computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """

    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N


def sespiou_coefficient2(pred, gt, all=False, smooth=1e-5):
    """
    computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """

    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth)/(TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2*Precision*Recall/(Recall + Precision + smooth)
    if all:
        return SE.sum() / N, SP.sum() / N, IOU.sum() / N, Acc.sum()/N, F1.sum()/N, Precision.sum()/N, Recall.sum()/N
    else:
        return IOU.sum() / N, Acc.sum()/N, SE.sum() / N, SP.sum() / N


# ========== get matrix ==========

def get_matrix(pred, gt, smooth=1e-5):
    """
    computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """

    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    return TP, FP, TN, FN


def get_HD(x, y, p=2):  # input should be like (Batch, 1, width, height) or (Batch, width, height)
    # print(x.shape, y.shape)
    x = x[0, :, :] > 0.5
    y = y[0, :, :] > 0.5
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)

    distance_matrix = torch.cdist(x, y, p=p)  # p=2 means Euclidean Distance

    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    hd = value.max(1)[0].mean().data.cpu().numpy()

    return hd
