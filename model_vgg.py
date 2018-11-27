from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
from layers import SoftEncodingLayer, NonGrayMaskLayer, ReweighingLayer
from torchvision.models import vgg
import numpy as np

class ColorizationNetwork_L(nn.Module):
    def __init__(self):
        super(ColorizationNetwork_L, self).__init__()
        self.VGG_19 = vgg.vgg19_bn(pretrained = True) #[batch, 256, 56*, 56*]
        self.VGG_19.classifier = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.Relu = nn.ReLU(inplace = True)#[batch, 128, 56*, 56*]
        self.conv_8 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(inplace = True), #[batch, 128, 56*, 56*]
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(inplace = True), #[batch, 128, 56*, 56*]
            nn.Conv2d(n_channels = 128, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1)
            #[batch, 313, 56*, 56*] 
        )
        

    def forward(self, img_L):
        vgg_output = self.Relu(self.VGG_19(img_L))
        Z_pred = self.conv_8(vgg_output)
        return Z_pred


class ColorizationNetwork(nn.Module):
    """
    This class represents the Colorisation Network
    """
    def __init__(self, batchNorm=True, pretrained=False):
        super(ColorizationNetwork, self).__init__()
        
        """
        self.nnecnclayer = NNEncLayer()
        self.priorboostlayer = PriorBoostLayer()
        self.nongraymasklayer = NonGrayMaskLayer()
        # self.rebalancelayer = ClassRebalanceMultLayer()
        self.rebalancelayer = Rebalance_Op.apply
        # Rebalance_Op.apply
        self.pool = nn.AvgPool2d(4,4)
        self.upsample = nn.Upsample(scale_factor=4)
        self.bw_conv = nn.Conv2d(1,64,3, padding=1)
        self.main = VGG(make_layers(cfg, batch_norm=batchNorm))
        self.main.classifier = nn.ConvTranspose2d(512,256,4,2, padding=1)
        self.relu = nn.ReLU()
        self.conv_8 = conv(256,256,2,[1,1], batchNorm=False)
        self.conv313 = nn.Conv2d(256,313,1,1)
        """
        
         #Components required to process L component
        self.ColorizationNetwork_L = ColorizationNetwork_L()
        self.upsample = nn.Upsample(scale_factor = 4)
        
        #Components required to process ab component
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.soft_encoding_layer = SoftEncodingLayer()
        self.non_gray_mask_layer = NonGrayMaskLayer()
        self.reweighting_layer = ReweighingLayer()
        
    def forward(self, img):
        
        #Processing the L component
        #gt_img_l = (gt_img[:,:1,:,:] - 50.) * 0.02
        #print('Input Shape', img.shape)
        img_h = img.shape[2]
        img_w = img.shape[3]
        img_L = img #[batch, 1, 224, 224]
        img_L[:, 1:,:,:] = torch.zeros([32, 2, img_h, img_w])
        Z_pred = self.ColorizationNetwork_L(img_L) #[batch, 313, 56, 56]
        
        # Processing the ab component 
        img_ab = img[:,1:,:,:] # [batch, 2, 224, 224]
        img_ab_downsample = self.pool(img_ab).cpu().data.numpy() # [batch, 2, 56, 56] #numpy
        
        #groundtruth Z
        img_ab_prob_dist = self.soft_encoding_layer.evaluate(img_ab_downsample) # [batch, 313, 56, 56]
        img_ab_prob_dist_argmax = np.argmax(img_ab_prob_dist, axis = 1).astype(np.int32)
        nongray_flag = self.non_gray_mask_layer.evaluate(img_ab_downsample) #[batch, 1, 1, 1]
        
        #Weight for class rebalancing 
        weight_per_pixel = self.reweighting_layer.evaluate(img_ab_prob_dist) # [batch, 1, 56, 56]
        
        #Final Weight per pixel
        weight_per_pixel_mask = (weight_per_pixel * nongray_flag).astype('float32') #[batch, 1, 56, 56]
        
        #Convert to tensors, ALL MUST BE float32 type
        weights = Variable(torch.from_numpy(weight_per_pixel_mask)).cuda()
        Z_groundtruth_argmax = Variable(torch.from_numpy(img_ab_prob_dist_argmax))
        Z_groundtruth_argmax = Z_groundtruth_argmax.type(torch.LongTensor).cuda()
        #Z_pred
        
        #Return is different for train and test mode
        if self.training:
            return weights, Z_groundtruth_argmax, Z_pred
        else:
            return weights, Z_groundtruth_argmax, Z_pred, self.upsample(Z_pred)
        
