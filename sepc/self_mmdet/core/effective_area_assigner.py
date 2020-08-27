import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.geometry import bbox_overlaps


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.
        Args:
            bboxes (Tensor): shape (m, 4)
            scale (float): the scale factor of bboxes

        Returns:
            (Tensor): shape (m, 4) scaled bboxes
        """
    w_half = (bboxes[:, 2] - bboxes[:, 0] + 1) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1] + 1) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0] + 1) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1] + 1) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(bboxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half - 1
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half - 1
    return boxes_exp


def is_located_in(points, bboxes, is_aligned=False):
    """is center a locates in box b Then we compute the area of intersect
    between box_a and box_b.

    Args:
      points: (tensor) bounding boxes, Shape: [m,2].
      bboxes: (tensor)  bounding boxes, Shape: [n,4].
        If is_aligned is ``True``, then m mush be equal to n
    Return:
        intersection area (Tensor), Shape: [m, n]. If is_aligned
        is ``True``, shape = [m]
    """
    if not is_aligned:
        return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
               (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))
    else:
        return (points[:, 0] > bboxes[:, 0]) & \
               (points[:, 0] < bboxes[:, 2]) & \
               (points[:, 1] > bboxes[:, 1]) & \
               (points[:, 1] < bboxes[:, 3])


def bboxes_area(bboxes):
    """Compute the area of an array of boxes."""
    w = (bboxes[:, 2] - bboxes[:, 0] + 1)
    h = (bboxes[:, 3] - bboxes[:, 1] + 1)
    areas = w * h

    return areas


class EffectiveAreaAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or
    a positive integer indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_area_thr (float): threshold within which pixels
            are labelled as positive.
        neg_area_thr (float): threshold above which pixels
            are labelled as positive.
        min_pos_iof (float): minimum iof of a pixel with a gt
            to be labelled as positive
    """
    def __init__(self, pos_area_thr, neg_area_thr, min_pos_iof=1e-2):
        self.pos_area_thr = pos_area_thr
        self.neg_area_thr = neg_area_thr
        self.min_pos_iof = min_pos_iof

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]

        # constructing effective gt areas
        gt_eff = scale_boxes(
            gt_bboxes,
            self.pos_area_thr)  # effective bboxes, i.e. center 0.2 part
        bbox_centers = (bboxes[:, 2:4] + bboxes[:, 0:2] + 1) / 2
        is_bbox_in_gt = is_located_in(
            bbox_centers,
            gt_bboxes)  # the center points lie within the gt boxes
        bbox_and_gt_eff_overlaps = bbox_overlaps(bboxes, gt_eff, mode='iof')
        is_bbox_in_gt_eff = is_bbox_in_gt & (
            bbox_and_gt_eff_overlaps > self.min_pos_iof)  # shape (n, k)
        # the center point of effective priors should be within the gt box

        # constructing ignored gt areas
        gt_ignore = scale_boxes(gt_bboxes, self.neg_area_thr)
        is_bbox_in_gt_ignore = (bbox_overlaps(bboxes, gt_ignore, mode='iof') >
                                self.min_pos_iof)
        is_bbox_in_gt_ignore &= (~is_bbox_in_gt_eff
                                 )  # rule out center effective pixels

        gt_areas = bboxes_area(gt_bboxes)
        _, sort_idx = gt_areas.sort(
            descending=True)  # smaller instances can overlay larger ones
        assign_result = self.assign_wrt_areas(is_bbox_in_gt_eff,
                                              is_bbox_in_gt_ignore,
                                              gt_labels,
                                              gt_priority=sort_idx)
        return assign_result

    def assign_wrt_areas(self,
                         is_bbox_in_gt_eff,
                         is_bbox_in_gt_ignore,
                         gt_labels=None,
                         gt_priority=None):
        num_bboxes, num_gts = is_bbox_in_gt_eff.size(
            0), is_bbox_in_gt_eff.size(1)
        if gt_priority is None:
            gt_priority = torch.arange(num_gts).to(is_bbox_in_gt_eff.device)
            # the bigger, the more preferable to be assigned

        assigned_gt_inds = is_bbox_in_gt_eff.new_full((num_bboxes, ),
                                                      0,
                                                      dtype=torch.long)
        # ignored indices
        inds_of_ignore = torch.any(is_bbox_in_gt_ignore, dim=1)
        assigned_gt_inds[inds_of_ignore] = -1
        if is_bbox_in_gt_eff.sum() == 0:  # No gt match
            return AssignResult(num_gts, assigned_gt_inds, None, labels=None)

        bbox_priority = is_bbox_in_gt_eff.new_full((num_bboxes, num_gts),
                                                   -1,
                                                   dtype=torch.long)
        inds_of_match = torch.any(
            is_bbox_in_gt_eff,
            dim=1)  # whether the bbox is matched (to any gt)

        # Each bbox could match with multiple gts.
        # The following codes deals with this
        matched_bbox_and_gt_correspondence = is_bbox_in_gt_eff[
            inds_of_match]  # shape [nmatch, k]
        matched_bbox_gt_inds = torch.nonzero(
            matched_bbox_and_gt_correspondence)[:, 1]
        # the matched gt index of each positive bbox. shape [nmatch]
        bbox_priority[is_bbox_in_gt_eff] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = bbox_priority[inds_of_match].max(
            dim=1)  # the maximum shape [nmatch]
        # effective indices
        assigned_gt_inds[inds_of_match] = argmax_priority + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_labels)
