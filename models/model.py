import torch.nn as nn
import torchvision.models as models
from models.ScConv import modify_resnet18,modify_resnet34,modify_resnet50

def get_model(model_name:str, num_classes:int,weights=None):
    if model_name == 'vgg16':
        net = models.vgg16()
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features,num_classes)
    elif model_name == 'ScConv18':
        net = modify_resnet18() 
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)
    elif model_name == 'ScConv50':
        net = modify_resnet50(weights) 
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name == 'inception_v3':
        net = models.inception_v3(pretrained=True)
        in_features = net.fc.in_features
        net.last_linear = nn.Linear(in_features,num_classes)
    elif model_name == 'efficientnet':
        net = models.efficientnet_v2_s()
        dropout = net.classifier[0]
        in_features = net.classifier[1].in_features

        net.classifier = nn.Sequential(
                        dropout,
                        nn.Linear(in_features, num_classes)
                    )
    elif model_name.startswith('resnet50'):
        # net = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        net = models.resnet50()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)

    elif model_name.startswith('resnet34'):
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
    elif model_name =='vit':
        net = models.vit_b_16()
        inchannel = net.heads.head.in_features
        net.heads.head = nn.Linear(inchannel, num_classes)
    elif model_name=='regnet':
        net = models.regnet_y_3_2gf()
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features,num_classes)
    elif model_name == 'convnext':
        net = models.convnext_small()
        net.classifier[2]=nn.Linear(768,num_classes)
    elif model_name == 'swin':
        net = models.swin_v2_t()
        in_features = net.head.in_features
        net.head = nn.Linear(in_features, num_classes)
    elif model_name == 'maxvit':
        net = models.maxvit_t()
        net.classifier[5]=nn.Linear(512, num_classes)
    return net