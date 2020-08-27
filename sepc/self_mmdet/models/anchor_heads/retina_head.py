from mmdet.models.anchor_heads import RetinaHead
from mmdet.models.registry import HEADS


@HEADS.register_module
class SepcRetinaHead(RetinaHead):
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
        return cls_score, bbox_pred
