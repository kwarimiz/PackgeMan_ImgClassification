'''
Description: 
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py
'''
import torch
import torch.nn.functional as F
import torch.nn as nn 
import torchvision.models as models
from torchvision.models.resnet import ResNet,_resnet,Bottleneck


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Residual_ScConv(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        norm_layer = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.scconv1 = ScConv(planes)
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.scconv1(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义一个函数来修改ResNet18模型，将ResidualBlock替换为包含ScConv的版本
def modify_resnet18():
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    for name, module in model.named_children():
        if name == 'layer1':
            new_layer1 = nn.Sequential()
            for child_name, child_module in module.named_children():
                if isinstance(child_module, models.resnet.BasicBlock):
                    in_channels = child_module.conv1.in_channels
                    out_channels = child_module.conv1.out_channels
                    stride = child_module.conv2.stride[0]
                    downsample = None
                    new_block = Residual_ScConv(in_channels, out_channels, stride, downsample)
                    new_layer1.add_module(child_name, new_block)
                else:
                    new_layer1.add_module(child_name, child_module)
            setattr(model, name, new_layer1)
    return model
    # 加载预训练的ResNet18模型
    model = models.resnet34()
    # 遍历模型的所有子模块，并替换残差块
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, models.resnet.BasicBlock):
                    # 假设in_channels和out_channels是已知的或者从child_module中获取的
                    in_channels = child_module.conv1.in_channels
                    out_channels = child_module.conv1.out_channels
                    stride = child_module.conv2.stride[0]
                    downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels)
                        )
                    # 创建新的残差块并包含ScConv
                    new_block = ResidualBlockWithSCConv(in_channels, out_channels, stride, downsample)
                    # 用新的残差块替换原有的残差块
                    setattr(module, child_name, new_block)
    return model

def modify_resnet34():
    pass


# class Bottleneck_ScConv(Bottleneck):
#     expansion: int = 4

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample= None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer=None,
#     ) -> None:
#         super().__init__(inplanes, planes, stride, groups, base_width, dilation, norm_layer)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups
        
#         # new conv
#         self.conv1 = ScConv(width)
#     def forward(self, x):

#         identity = x
        
#         out = self.conv1(x)
       
#         out = self.relu(out)


#         # Replace the original conv2 with ScConv
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
       
#         out += identity

#         out = self.relu(out)

#         return out
    
class Bottleneck_ScConv(Bottleneck):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample= None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__(inplanes, planes, stride, groups, base_width, dilation, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # new conv
        self.conv2 = ScConv(width)
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        identity = out

        # Replace the original conv2 with ScConv
        out = self.conv2(out)

        out = self.bn2(out)

        out += identity
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out
    
def modify_resnet50(weights = None):
    if weights is not None:
        net = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    else:
        net = models.resnet50(weights=None)

    net.layer3[4] = Bottleneck_ScConv(1024,256)
    net.layer4[2] = Bottleneck_ScConv(2048,512)
    
    return net
    # weights = None

    # return _resnet(Bottleneck_ScConv, [3, 4, 6, 3], weights,True)


if __name__ == '__main__':
    x       = torch.randn(1,32,16,16)
    model   = ScConv(32)
    print(model(x).shape)
