import torch.nn as nn
import torchvision.models as models
from models.ScConv import modify_resnet18,modify_resnet34,modify_resnet50

def get_model(model_name:str, num_classes:int,weights=None):

    if model_name == 'vgg16':
        if weights=='pretrained':
            net = models.vgg16(pretrained=True)
        else:
            net = models.vgg16()
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features,num_classes)

    # ConvNext
    elif model_name == 'convnext':
        if weights=='pretrained':
            net = models.convnext_base(weights='IMAGENET1K_V1')
        else:
            net = models.convnext_base()
        net = models.convnext_base()
        net.classifier[2]=nn.Linear(1024,num_classes)

    # EfficientNet
    elif model_name == 'efficientnet':
        if weights=='pretrained':
            net = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        else:
            net = models.efficientnet_v2_m()
        dropout = net.classifier[0]
        in_features = net.classifier[1].in_features

        net.classifier = nn.Sequential(
                        dropout,
                        nn.Linear(in_features, num_classes)
                    )
    # MaxViT
    elif model_name == 'maxvit':
        if weights=='pretrained':
            net = models.maxvit_t(weights='IMAGENET1K_V1')
        else:
            net = models.maxvit_t()
        net.classifier[5]=nn.Linear(512, num_classes)

    # RegNet
    elif model_name=='regnet':
        if weights=='pretrained':
            net = models.regnet_y_16gf(weights='IMAGENET1K_V2')
        else:
            net = models.regnet_y_16gf()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)
    
    # ResNet
    elif model_name=='resnet101':
        if weights=='pretrained':
            net = models.resnet101(weights='IMAGENET1K_V2')
        else:
            net = models.resnet101()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    # Swin
    elif model_name == 'swin':
        if weights=='pretrained':
            net = models.swin_v2_b(weights='IMAGENET1K_V1')
        else:
            net = models.swin_v2_b()
        in_features = net.head.in_features
        net.head = nn.Linear(in_features, num_classes)

    # ViT
    elif model_name =='vit':
        if weights=='pretrained':
            net = models.vit_b_16(weights='IMAGENET1K_V1')
        else:
            net = models.vit_b_16()
        inchannel = net.heads.head.in_features
        net.heads.head = nn.Linear(inchannel, num_classes)
        
    # WideResNet
    elif model_name =='wide_resnet':
        if weights=='pretrained':
            net = models.wide_resnet50_2(pretrained=True)
        else:
            net = models.wide_resnet50_2(weights='IMAGENET1K_V2')
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name == 'mnasnet':
        net = models.mnasnet0_75()
        net.classifier[1]=nn.Linear(1280,num_classes)

# Customized Models
    elif model_name == 'ScConv18':
        net = modify_resnet18() 
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name == 'ScConv50':
        net = modify_resnet50(weights) 
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name == 'inception':
        net = models.inception_v3(pretrained=True)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)
    
    elif model_name.startswith('resnet50'):
        # net = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        net = models.resnet50()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name.startswith('resnet34'):
        if weights=='pretrained':
            net = models.resnet34(pretrained=True)
        else:
            net = models.resnet34() 
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name.startswith('resnet18'):
        # net = models.resnet18(weights = 'ResNet18_Weights.DEFAULT') #
        net = models.resnet18()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name == 'resnet34_ScConv':
        # net = modify_resnet34(weights = 'ResNet34_Weights.DEFAULT') # 不写的话，无参数
        net = modify_resnet34() # 不写的话，无参数
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    return net