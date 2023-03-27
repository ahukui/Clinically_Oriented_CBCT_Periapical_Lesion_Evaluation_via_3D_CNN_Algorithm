import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn.functional import interpolate
import os


norm_func = nn.InstanceNorm3d
norm_func2d = nn.InstanceNorm2d
act_func = nn.CELU

import torch.nn.functional
from torch.nn.functional import fold, unfold
from math import *

class NonLocal(nn.Module):
    def __init__(self, inchannel):
        super(NonLocal, self).__init__()
        self.q = nn.Sequential(
            nn.Conv3d(inchannel, inchannel//2, kernel_size=1, bias=False))
        self.key = nn.Sequential(
            nn.Conv3d(inchannel, inchannel//2, kernel_size=1, bias=False))
        self.val = nn.Sequential(
            nn.Conv3d(inchannel, inchannel//2, kernel_size=1, bias=False))
        self.post = nn.Sequential(
            nn.Conv3d(inchannel//2, inchannel, kernel_size=1, bias=False))
    def forward(self, x):
        q = self.q(x)
        key = self.key(x)
        val = self.val(x)
        n, c, d, h, w = q.shape
        q = q.view((n, c, d*h*w))
        key = key.view((n, c, d*h*w)).permute((0, 2, 1)).contiguous()
        val = val.view((n, c, d*h*w))
        atten_map = torch.matmul(key, q).softmax(dim=2)
        val = torch.matmul(val, atten_map).view((n, c, d, h, w))
        out = self.post(val) + x
        return out

def unfold3d(data, kernel_size, stride, padding):
    n, c, d, h, w = data.shape 
    data = data.view(n, c*d, h, w)
    data = unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    print(data.shape)
    data = data.view(n, c, d, kernel_size*kernel_size, -1).permute(0, 1, 3, 4, 2).contiguous().view(n, c*kernel_size*kernel_size, -1, d)
    print(data.shape)
    data = unfold(data, kernel_size=(kernel_size,1), stride=(stride,1), padding=0)
    print(data.shape)
    return data

def fold3d(data, kernel_size, stride, output_size):
    n, c, dim = data.shape
    if type(output_size)==list:
        d, h, w = output_size
    else: 
        d = output_size
        h = output_size
        w = output_size
    data = fold(data, output_size=(dim*stride//d, d), kernel_size=(1, kernel_size), stride=(1, stride))
    data = data.view(n, -1, stride*stride, dim*stride//d, d).permute(0, 1, 4, 2, 3).contiguous().view(n, -1, dim*stride//d)
    data = fold(data, output_size=(h, w), stride=stride, kernel_size=kernel_size).view(n, -1, d, h, w)
    return data

class Attention(nn.Module):
    def __init__(self, n, channel):
        super(Attention, self).__init__()
        self.position = nn.Parameter(torch.randn((n,n)), requires_grad=True)
        self.to_qkv = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, 3*channel)
        )
        self.scale = 1/sqrt(n)

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        q, k, v = self.to_qkv(x).chunk(3, 2)
        k = k.permute(0,2,1).contiguous()
        qk = torch.matmul(q, k)
        qk = qk * self.scale + self.position
        qk = qk.softmax(dim=2)
        out = torch.matmul(qk, v).permute(0,2,1).contiguous()
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class SwinBlock(nn.Module):
    def __init__(self, ww, inchannel):
        super(SwinBlock, self).__init__()
        self.atten_0 = Attention(ww*ww*ww, inchannel)
        self.mlp_0 = FeedForward(inchannel, inchannel//2)
        self.ww = ww

    def forward(self, x):
        n,c,d,h,w=x.shape
        print(x.shape)
        _x = unfold3d(x, self.ww, self.ww, 0)
        print('unfold3d', _x.shape)
        _x = _x.permute(0,2,1).contiguous().view(-1, c, self.ww*self.ww*self.ww)
        _x = self.atten_0(_x) + _x
        _x = _x.permute(0,2,1).contiguous()
        _x = self.mlp_0(_x)
        _x = _x.view(n, -1, c*self.ww*self.ww*self.ww)
        _x = _x.permute(0,2,1).contiguous()
        print('before flod', _x.shape)
        x = fold3d(_x, self.ww, self.ww, [d, h, w]) + x
        return x


class Swish(nn.Module):
    def __init__(self, inplace):
        super(Swish, self).__init__() 
    def forward(self, x):
        return x*x.sigmoid()

class PassNorm(nn.Module):
    def __init__(self, inplace):
        super(PassNorm, self).__init__() 
    def forward(self, x):
        return x

act_func = Swish
# norm_func = PassNorm

class SoftPool(nn.Module):
    def __init__(self, pool_op=nn.AvgPool3d, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super(SoftPool, self).__init__()
        self.pool_op = pool_op(kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        x_exp = x.exp()
        x = x * x_exp
        x_pool = self.pool_op(x)
        x_exp_pool = self.pool_op(x_exp)
        x = x_pool/x_exp_pool
        return x

class ResNormActConv3d(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNormActConv3d, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv = nn.Sequential(
            norm_func(inchannel, affine=True),
            nn.GELU(),
            nn.Conv3d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
        )

        self.conv0 = nn.Sequential(
            norm_func(inchannel, affine=True),
            nn.GELU(),
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        if self.inchannel == self.outchannel:
            return self.conv(x) + x
        else:
            return self.conv(x) + self.conv0(x)

            
        

class ResConv3dNormAct(nn.Module):
    def __init__(self, inchannel, outchannel, att=False, num_heads=2):
        super(ResConv3dNormAct, self).__init__()
        self.num_heads = num_heads
        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU())
        if att:
            self.conv1 = nn.ModuleList()
            for i in range(num_heads):
                print(i)
                self.conv1.append(SwinBlock(4, outchannel//num_heads))
        else: 
            self.conv1 = nn.Sequential(
                nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
                norm_func(outchannel, affine=True),
                nn.GELU())
        self.att = att
    
    def forward(self, x):
        out = self.conv_0(x)
        if self.att:
            out = out.chunk(self.num_heads, 1)
            outs = []
            for index in range(self.num_heads):
                outs.append(self.conv1[index](out[index]))
            out = torch.cat(outs, dim=1)
        else: 
            out = out + self.conv1(out)
        return out

class DownConv(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, if_att=True, num_heads=2):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(kernel_size, stride),
            ResConv3dNormAct(inchannel, outchannel, if_att, num_heads)
        )
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, inchannel, outchannel, stride, if_att=False, num_heads=2):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True),
            nn.Conv3d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )
    def forward(self, x):
        return self.up_conv(x)

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


class sSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class cSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.Conv_Squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv3d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse


class scSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE3D(in_channels)
        self.sSE = sSE3D(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse






class ResConv2dNormAct(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, groups):
        super(ResConv2dNormAct, self).__init__()
        self.falg = (inchannel==outchannel)
        self.conv0 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
        )
        self.norm_act0 = nn.Sequential(
            norm_func2d(outchannel, affine=True),
            nn.GELU()
        )
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(outchannel//2, outchannel//2, kernel_size=kernel_size, padding=kernel_size//2, groups=groups, bias=False)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(outchannel//2, outchannel//2, kernel_size=kernel_size-2, padding=kernel_size//2-1, groups=groups, bias=False)
        )

        self.fusion0 = nn.Sequential(
            norm_func2d(outchannel, affine=True),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, bias=False),
        )

        self.conv0_1 = nn.Sequential(
            norm_func2d(outchannel, affine=True),
            nn.GELU(),
            scSE(outchannel),
        )


        self.conv1 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=1, bias=False),
        )
        self.norm_act1 = nn.Sequential(
            norm_func2d(outchannel, affine=True),
            nn.GELU()
        )
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(outchannel//2, outchannel//2, kernel_size=kernel_size, padding=kernel_size//2, groups=groups, bias=False)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(outchannel//2, outchannel//2, kernel_size=kernel_size-2, padding=kernel_size//2-1, groups=groups, bias=False)
        )

        self.fusion1 = nn.Sequential(
            norm_func2d(outchannel, affine=True),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.conv0(x)
        norm = self.norm_act0(x).chunk(2, 1)
        x1 = torch.cat([self.conv1_3(norm[0]), self.conv1_5(norm[1])], dim=1)
        x1 = self.fusion0(x1)


        x1 = self.conv0_1(x1)

        x2 = self.conv1(x1)
        norm = self.norm_act1(x2).chunk(2, 1)
        x2 = torch.cat([self.conv2_3(norm[0]), self.conv2_5(norm[1])], dim=1)
        x2 = self.fusion1(x2)
        return x + x2

class Conv2dUnit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(Conv2dUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            norm_func2d(outchannel, affine=True),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)


class VesselNet(nn.Module):

    def __init__(self,
                 inchannel=1,
                 num_seg_classes=1,
                 size = 16):
        super(VesselNet, self).__init__()
        base_channel = 16
        self.pre_conv = nn.Sequential(
            norm_func(inchannel, affine=True),
            nn.Conv3d(inchannel, base_channel, kernel_size=3, padding=1, bias=False),
            norm_func(base_channel, affine=True),
            nn.GELU(),
        )
        self.conv0 = nn.Sequential(
            nn.AvgPool3d(2, 2),
            ResNormActConv3d(base_channel, base_channel*2),
            cSE3D(base_channel*2),
            ResNormActConv3d(base_channel*2, base_channel*2),
        )
        
        self.conv1 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResNormActConv3d(base_channel*2, base_channel*4),
            ResNormActConv3d(base_channel*4, base_channel*4),
        ) #128x128


        self.conv2 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResNormActConv3d(base_channel*4, base_channel*8),
            cSE3D(base_channel*8),
            ResNormActConv3d(base_channel*8, base_channel*8),
        ) #64x64


        self.conv3 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResNormActConv3d(base_channel*8, base_channel*16),
            ResNormActConv3d(base_channel*16, base_channel*16),
        ) #32x32


        self.conv4 = nn.Sequential(
            nn.MaxPool3d((1,2,2),(1,2,2)),
            ResNormActConv3d(base_channel*16, base_channel*16),
            cSE3D(base_channel*16),            
            ResNormActConv3d(base_channel*16, base_channel*16),
        ) #16x16


        self.conv5 = nn.Sequential(
            NonLocal(base_channel*16)
        )

        self.up5 = UpConv(base_channel*16, base_channel*16, (2,2,2))

        # self.up4 = UpConv(base_channel*16, base_channel*8, (2,2,2))
        self.p4 = nn.Sequential(
            ResNormActConv3d(base_channel*16, base_channel*16),
            cSE3D(base_channel*16),
            ResNormActConv3d(base_channel*16, base_channel*16)
        )

        self.up4 = UpConv(base_channel*16, base_channel*16, (1,2,2))

        self.p3 = nn.Sequential(
            ResNormActConv3d(base_channel*16, base_channel*8),
            ResNormActConv3d(base_channel*8, base_channel*8)
        )

        self.up3 = UpConv(base_channel*8, base_channel*8, (2,2,2))

        self.p2 = nn.Sequential(
            ResNormActConv3d(base_channel*8, base_channel*4),
            cSE3D(base_channel*4),
            ResNormActConv3d(base_channel*4, base_channel*4)

        )

        self.up2 = UpConv(base_channel*4, base_channel*4, (2,2,2))



        self.p1 = nn.Sequential(
            ResNormActConv3d(base_channel*4, base_channel*2),
            ResNormActConv3d(base_channel*2, base_channel*2)
        )


        self.up1 = UpConv(base_channel*2, base_channel*2, (2,2,2))


        self.p0 = nn.Sequential(
            ResNormActConv3d(base_channel*2, base_channel*1),
            cSE3D(base_channel*1),            
            ResNormActConv3d(base_channel*1, base_channel*1)
        )

        self.up0 = UpConv(base_channel*1, base_channel*1, (2,2,2))


        self.p_last = nn.Sequential(
            ResNormActConv3d(base_channel*1, base_channel*1),
            ResNormActConv3d(base_channel*1, base_channel*1)
        )

        # self.efficient_1 = Efficient(base_channel*1, 16, 320, 320)
        
        
        self.conv_last = nn.Sequential(
            nn.Conv3d(base_channel*1, base_channel*1, kernel_size=3, padding=1, bias=False),
            norm_func(base_channel*1, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channel*1, num_seg_classes, kernel_size=3, padding=1, bias=True)
        )
        self.conv_seg_2 = nn.Sequential(
            nn.Conv3d(base_channel*1, base_channel*1, kernel_size=3, padding=1, bias=False),
            norm_func(base_channel*1, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channel*1, num_seg_classes, kernel_size=3, padding=1, bias=True)
        )

        self.conv_seg_3 = nn.Sequential(
            nn.Conv3d(base_channel*2, base_channel*1, kernel_size=3, padding=1, bias=False),
            norm_func(base_channel*1, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channel*1, num_seg_classes, kernel_size=3, padding=1, bias=True)
        )


        self.conv_output = nn.Sequential(
            nn.Conv3d(base_channel, num_seg_classes, kernel_size=3, padding=1, bias=False),
            norm_func(num_seg_classes, affine=True),
       )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
    
    def upsample_depth(self, x, n):
        N, C, H, W = x.shape
        depth = N//n 
        x = x.view(n, depth, C, H, W).transpose(1,2).contiguous()
        x = F.upsample(x, scale_factor=(2,1,1), mode='trilinear', align_corners=True)
        x = x.transpose(1,2).contiguous().view(2*N, C, H, W)
        return x


    def forward(self, x):
        #print("input", x.size())
        x = self.pre_conv(x)  # [3, 16, 32, 512, 512] B, C, D, W, H
        #print("pre_conv", x.size())

        conv0 = self.conv0(x) #32x256x256
        #print(conv0.size())

        conv1 = self.conv1(conv0) #16x128x128
        #print("conv1",  conv1.size()) 


        conv2 = self.conv2(conv1) #8x64x64
        #print("_conv1, conv2", conv1.size(), conv2.size()) 

        conv3 = self.conv3(conv2) #4x32x32
        #print("_conv2, conv3", conv2.size(), conv3.size())    

        conv4 = self.conv4(conv3)
        #print('conv4 shape', conv4.shape)   
        conv5 = self.conv5(conv4)
        #print('conv5 shape', conv5.shape)        



        p4 = self.p4(torch.add(conv5, conv4))
        #print('p4 shape', p4.shape)  
              
      
        p3 = self.p3(torch.add(self.up4(p4), conv3))
        #print('p3 shape', p3.shape)        
                
        p2 = self.p2(torch.add(self.up3(p3), conv2))
        #print('p31 shape', p3.shape)           
                       
        p1 = self.p1(torch.add(self.up2(p2), conv1))
        #print('p1 shape, conv0', p1.shape, conv0.shape)         
        #            
        p0 = self.p0(torch.add(self.up1(p1), conv0))
        #print('p1 shape', p1.shape)   

        last_p = self.p_last(torch.add(self.up0(p0), x))        

        seg_3 = self.conv_seg_3(p1)

        seg_2 = self.conv_seg_2(p0)

        seg_1 = self.conv_last(last_p)
        #print('output', seg_3.shape, seg_2.shape, seg_1.shape)
        out_last = [seg_3, seg_2, seg_1]
        #print('output', seg_3.shape, seg_2.shape, seg_1.shape, out_last.shape)
        return out_last


def vesselnet(**kwargs):
    model = VesselNet(**kwargs)
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    input = torch.randn(1, 1, 32, 32, 32).cuda()
    model = vesselnet()
    model.cuda()
    outputs = model(input)
    for index, output in enumerate(outputs):
        print(output.size())
    
    
    
    
    
    
    