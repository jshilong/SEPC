from abc import ABCMeta, abstractmethod

import torch


class BaseCoders(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        pass

    @abstractmethod
    def decode(self, priors, deltas, max_shape=None):
        pass


def xyxy_to_tblr(xyxy, centers=None):
    """Convert [x1 y1 x2 y2] box format to [t, b, l, r] format.

    if center is not given, the center point is used
    """
    if centers is None:
        centers = (xyxy[:, :2] + xyxy[:, 2:] + 1) / 2
    assert len(centers) == len(xyxy)
    xmin, ymin, xmax, ymax = xyxy.split(1, dim=1)
    t = centers[:, 1].unsqueeze(1) - ymin
    b = ymax - centers[:, 1].unsqueeze(1) + 1
    l = centers[:, 0].unsqueeze(1) - xmin  # noqa E741
    r = xmax - centers[:, 0].unsqueeze(1) + 1
    # return torch.cat((t, b, l, r), dim=1) # true one
    return torch.cat((b, t, l, r), dim=1)


def tblr_to_xyxy(tblr, centers):
    """Convert [x1 y1 x2 y2] box format to [t, b, l, r] format.

    if center is not given, the center point is used
    """
    assert len(centers) == len(tblr)
    # t, b, l, r = tblr.split(1, dim=1) ##true one
    b, t, l, r = tblr.split(1, dim=1)
    xmin = centers[:, 0].unsqueeze(1) - l
    xmax = centers[:, 0].unsqueeze(1) + r - 1
    ymin = centers[:, 1].unsqueeze(1) - t
    ymax = centers[:, 1].unsqueeze(1) + b - 1
    return torch.cat((xmin, ymin, xmax, ymax), dim=1)


class TBLRCoder(BaseCoders):
    def __init__(self, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
        self.means = means
        self.stds = stds

    def encode(self, priors, gt):
        """Encode the variances from the priorbox layers into the ground truth
        boxes we have matched (based on jaccard overlap) with the prior boxes.

        Args:
            gt: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_gt, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_proposals,4].
        Return:
            encoded boxes (tensor), Shape: [num_proposals, 4]
        """

        # dist b/t match center and prior's center
        prior_centers = (priors[:, 0:2] + priors[:, 2:4] + 1) / 2
        wh = priors[:, 2:4] - priors[:, 0:2] + 1
        loc = xyxy_to_tblr(gt, centers=prior_centers)
        w, h = torch.split(wh, 1, dim=1)
        loc[:, :2] /= h
        loc[:, 2:] /= w

        means = loc.new_tensor(self.means).unsqueeze(0)
        stds = loc.new_tensor(self.stds).unsqueeze(0)
        loc = loc.sub_(means).div_(stds)
        return loc

    def decode(self, priors, deltas, max_shape=None):
        """Encode the variances from the priorbox layers into the ground truth
        boxes we have matched (based on jaccard overlap) with the prior boxes.

        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
        """
        # dist b/t match center and prior's center

        means = deltas.new_tensor(self.means).repeat(1, deltas.size(1) // 4)
        stds = deltas.new_tensor(self.stds).repeat(1, deltas.size(1) // 4)
        loc_decode = deltas * stds + means
        prior_centers = (priors[:, 0:2] + priors[:, 2:4] + 1) / 2
        wh = priors[:, 2:4] - priors[:, 0:2] + 1
        w, h = torch.split(wh, 1, dim=1)
        loc_decode[:, :2] *= h
        loc_decode[:, 2:] *= w
        boxes = tblr_to_xyxy(loc_decode, centers=prior_centers)
        if max_shape is not None:
            boxes[:, 0].clamp_(min=0, max=max_shape[1] - 1)
            boxes[:, 1].clamp_(min=0, max=max_shape[0] - 1)
            boxes[:, 2].clamp_(min=0, max=max_shape[1] - 1)
            boxes[:, 3].clamp_(min=0, max=max_shape[0] - 1)
        return boxes


def build_coder(cfg, **kwargs):
    temp_cfg = cfg.copy()
    type = temp_cfg.pop('type', None)
    if type != 'TBLRCoder':
        raise TypeError('fsaf only support TBLRCoder')
    return TBLRCoder(**temp_cfg)
