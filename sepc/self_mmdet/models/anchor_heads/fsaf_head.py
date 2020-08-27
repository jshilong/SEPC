import numpy as np
import torch
from mmcv.cnn import normal_init
from mmdet.core import force_fp32, multiclass_nms
from mmdet.models.anchor_heads import RetinaHead
from mmdet.models.losses import IoULoss
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmdet.models.registry import HEADS, LOSSES
from mmdet.models.utils import bias_init_with_prob

from sepc.self_mmdet.core.build_coders import build_coder
from sepc.self_mmdet.core.fsaf_anchor_target import (fsaf_anchor_target,
                                                     multi_apply)


@weighted_loss
def iou_loss_tblr(pred, target, eps=1e-6):
    """Calculate the iou loss when both the prediction and targets are encoded
    in TBLR format.

    Args:
        pred (Tensor): With shape (num_anchor, 4)
        target (Tensor): With shape (num_anchor, 4)
        eps: the minimum iou threshold

    Returns:
        loss (Tensor): With shape (num_anchors)
    """
    xt, xb, xl, xr = torch.split(pred, 1, dim=-1)

    # the ground truth position
    gt, gb, gl, gr = torch.split(target, 1, dim=-1)

    # compute the bounding box size
    X = (xt + xb) * (xl + xr)  # AreaX
    G = (gt + gb) * (gl + gr)  # AreaG

    # compute the IOU
    Ih = torch.min(xt, gt) + torch.min(xb, gb)
    Iw = torch.min(xl, gl) + torch.min(xr, gr)

    In = Ih * Iw
    U = (X + G - In).clamp(min=1)  # minimum area should be 1

    IoU = In / U
    IoU = IoU.squeeze()
    ious = IoU.clamp(min=eps)
    loss = -ious.log()
    return loss


@LOSSES.register_module
class IoULossTBLR(IoULoss):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULossTBLR, self).__init__(eps, reduction, loss_weight)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        weight = weight.sum(dim=-1) / 4.  # iou loss is a scalar!
        loss = self.loss_weight * iou_loss_tblr(pred,
                                                target,
                                                weight,
                                                eps=self.eps,
                                                reduction=reduction,
                                                avg_factor=avg_factor,
                                                **kwargs)
        return loss


@HEADS.register_module
class FSAFHead(RetinaHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 effective_threshold=0.2,
                 ignore_threshold=0.2,
                 coder=dict(type='TBLRCoder', ),
                 **kwargs):

        self.coder = build_coder(coder)

        self.effective_threshold = effective_threshold
        self.ignore_threshold = ignore_threshold
        super(FSAFHead, self).__init__(num_classes, in_channels, stacked_convs,
                                       octave_base_scale, scales_per_octave,
                                       conv_cfg, norm_cfg, **kwargs)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01, bias=1)

    def forward_single(self, x):
        if not isinstance(x, list):
            x = [x, x]
        cls_feat = x[0]
        reg_feat = x[1]
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, self.relu(
            bbox_pred)  # TBLR encoder only accepts positive bbox_pred

    def loss_single(self,
                    cls_score,
                    bbox_pred,
                    labels,
                    label_weights,
                    bbox_targets,
                    bbox_weights,
                    num_total_samples,
                    cfg,
                    reduction_override=None):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score,
                                 labels,
                                 label_weights,
                                 avg_factor=num_total_samples,
                                 reduction_override=reduction_override)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(bbox_pred,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_samples,
                                   reduction_override=reduction_override)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes,
        gt_labels,
        img_metas,
        cfg,
        gt_bboxes_ignore=None,
    ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        batch_size = len(gt_bboxes)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes,
                                                        img_metas,
                                                        device=device)

        cls_reg_targets = fsaf_anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            sampling=self.sampling,
            encoder=self.coder.encode)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg,
         pos_assigned_gt_inds_list) = cls_reg_targets

        num_gts = np.array(list(map(len, gt_labels)))
        num_total_samples = (num_total_pos +
                             num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg,
            reduction_override='none')
        cum_num_gts = list(np.cumsum(num_gts))
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            for j in range(1, batch_size):
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = sum(map(len, gt_labels))
        with torch.no_grad():
            loss_levels, = multi_apply(self.collect_loss_level_single,
                                       losses_cls,
                                       losses_bbox,
                                       pos_assigned_gt_inds_list,
                                       labels_seq=torch.arange(num_gts,
                                                               device=device))
            loss_levels = torch.stack(loss_levels, dim=0)
            loss, argmin = loss_levels.min(dim=0)
        losses_cls, losses_bbox, pos_inds = multi_apply(
            self.reassign_loss_single,
            losses_cls,
            losses_bbox,
            pos_assigned_gt_inds_list,
            labels_list,
            list(range(len(losses_cls))),
            min_levels=argmin)
        num_pos = torch.cat(pos_inds, 0).sum().float()
        accuracy = self.calculate_accuracy(cls_scores, labels_list, pos_inds)
        for i in range(len(losses_cls)):
            losses_cls[i] /= num_pos
            losses_bbox[i] /= num_pos
        return dict(loss_cls=losses_cls,
                    loss_bbox=losses_bbox,
                    num_pos=num_pos / batch_size,
                    accuracy=accuracy)

    def calculate_accuracy(self, cls_scores, labels_list, pos_inds):
        with torch.no_grad():
            num_pos = torch.cat(pos_inds, 0).sum().float()
            num_class = cls_scores[0].size(1)
            scores = [
                cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos]
                for cls, pos in zip(cls_scores, pos_inds)
            ]
            labels = [
                label.reshape(-1)[pos]
                for label, pos in zip(labels_list, pos_inds)
            ]
            argmax = lambda x: x.argmax(1) if x.numel(  # noqa E731
            ) > 0 else -100
            num_correct = sum([(argmax(score) + 1 == label).sum()
                               for score, label in zip(scores, labels)])
            return num_correct.float() / (num_pos + 1e-3)

    def collect_loss_level_single(self, cls_loss, reg_loss,
                                  pos_assigned_gt_inds, labels_seq):
        """Get the average loss in each FPN level w.r.t. each gt label.

        Args:
            cls_loss (tensor): classification loss of each feature
                map pixel, shape (num_anchor, num_class)
            reg_loss (tensor): regression loss of each feature map
                pixel, shape (num_anchor)
            pos_assigned_gt_inds (tensor): shape (num_anchor), indicating
                which gt the prior is assigned to (-1: no assignment)
            labels_seq: The rank of labels
        """
        loss = cls_loss.sum(
            dim=-1) + reg_loss  # total loss at each feature map point
        match = pos_assigned_gt_inds.reshape(-1).unsqueeze(
            1) == labels_seq.unsqueeze(0)
        loss_ceiling = loss.new_zeros(1).squeeze(
        ) + 1e6  # default loss value for a layer where no anchor is positive
        losses_ = torch.stack([
            torch.mean(loss[match[:, i]])
            if match[:, i].sum() > 0 else loss_ceiling for i in labels_seq
        ])
        return losses_,

    def reassign_loss_single(self, cls_loss, reg_loss, pos_assigned_gt_inds,
                             labels, level, min_levels):
        """Reassign loss values at each level by masking those where the pre-
        calculated loss is too large.

        Args:
            cls_loss (Tensor): With shape (num_anchors, num_classes)
                classification loss
            reg_loss (Tensor): With shape (num_anchors) regression loss
            pos_assigned_gt_inds (Tensor): With shape (num_anchors),
                the gt indices that each positive anchor corresponds
                to. (-1 if it is a negative one)
            labels (Tensor): With shape (num_anchors). Label assigned
                to each pixel
            level (int): the current level index in the pyramid
                (0-4 for RetinaNet)
            min_levels (Tensor): shape (num_gts), the best-matching
                level for each gt

        Returns:
            cls_loss (Tensor): With shape (num_anchors, num_classes).
                Corrected classification loss
            reg_loss (Tensor): With shape (num_anchors). Corrected
                regression loss
            keep_indices (Tensor): With shape (num_anchors). Indicating final
                postive anchors
        """

        unmatch_gt_inds = torch.nonzero(
            min_levels !=
            level)  # gts indices that unmatch with the current level
        match_gt_inds = torch.nonzero(min_levels == level)
        loc_weight = cls_loss.new_ones(cls_loss.size(0))
        cls_weight = cls_loss.new_ones(cls_loss.size(0), cls_loss.size(1))
        zeroing_indices = (pos_assigned_gt_inds.view(
            -1, 1) == unmatch_gt_inds.view(1, -1)).any(dim=-1)
        keep_indices = (pos_assigned_gt_inds.view(-1, 1) == match_gt_inds.view(
            1, -1)).any(dim=-1)
        loc_weight[zeroing_indices] = 0

        # only the weight corresponding to the label is
        # zeroed out if not selected
        zeroing_labels = labels[zeroing_indices] - 1
        assert (zeroing_labels >= 0).all()
        cls_weight[zeroing_indices, zeroing_labels] = 0

        # weighted loss for both cls and reg loss
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')
        return cls_loss, reg_loss, keep_indices

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """Transform outputs for a single batch item into labeled boxes."""
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.coder.decode(anchors, bbox_pred, max_shape=img_shape)
            # bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
            #                     self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
