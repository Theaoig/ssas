import torch
try:
    from . cbam import CBAM
    from . dbnet import DBHead,FCDBHead
except:
    from cbam import CBAM
    from dbnet import DBHead,FCDBHead
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class Merge(torch.nn.Module):
    """upsample the features and concant them
    
    Args:
        torch (int): feaure channel number
    """
    def __init__(self,):
        super().__init__()
        self.upsample_2=torch.nn.Sequential(torch.nn.UpsamplingBilinear2d(scale_factor=2))
        self.upsample_4=torch.nn.Sequential(torch.nn.UpsamplingBilinear2d(scale_factor=4))
        self.upsample_8=torch.nn.Sequential(torch.nn.UpsamplingBilinear2d(scale_factor=8))

    def forward(self, features):
        fea0=features['0']
        fea1=self.upsample_2(features['1'])
        fea2=self.upsample_4(features['2'])
        fea3=self.upsample_8(features['3'])
        feature=torch.cat([fea0,fea1,fea2,fea3],dim=1)
        return feature

class Predictor(torch.nn.Module):
    """predict img label and anomaly mask

    Args:
        torch (int): feature channel number
    """
    def __init__(self, dim, db_K=50):
        super().__init__()
        self.mask_predictor=DBHead(dim,1,db_K)
        self.label_predictor=FCDBHead(dim,1,db_K)
                
    def forward(self,x):
        mask=self.mask_predictor(x)
        x=x+(torch.nn.AvgPool2d(4,4)(mask)*x)
        label=self.label_predictor(x)
        return mask,label

class PMAS(torch.nn.Module):
    """presude anomaly segementation

    Args:
        torch (None Args): None
    """
    def __init__(self,feature_dim=256,trainable_layers=0):
        super(PMAS, self).__init__()
        self.backbone=resnet_fpn_backbone('resnet50', True, trainable_layers=trainable_layers)
        self.merge=Merge()
        self.predictor=Predictor(dim=4*feature_dim)
        
    def forward(self,x):
        
        multiscal_features=self.backbone(x)
        feature=self.merge(multiscal_features)
        mask,label=self.predictor(feature)
        return mask,label
        
if __name__=="__main__":
    net=PMAS().to('cuda')
    x=torch.randn((10,3,256,256)).to('cuda')
    mask,label=net(x)
    print(mask.shape,label.shape)
    
    label_loss=torch.nn.BCELoss()
    mask_loss=torch.nn.SmoothL1Loss()
    
    label_target=torch.empty(10, 1).random_(2).to('cuda')
    mask_target=torch.empty(10,1,256,256).random_(2).to('cuda')
    
    print(mask_loss(mask,mask_target),label_loss(label,label_target))
    
    
    