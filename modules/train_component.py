import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts
from torch import nn

def get_loss_function(loss_function_name):
    if loss_function_name == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()
    elif loss_function_name == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Invalid loss function name: {loss_function_name}')
    return loss_function


def get_optimizer(optimizer_name, net, lr, momentum=0.9, weight_decay=1e-4):
    if optimizer_name == 'SGD':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')
    return optimizer

def get_scheduler(scheduler_name, optimizer, T_0=10, T_mult=2, eta_min=5e-5):
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'Cos':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    else:
        raise ValueError(f'Invalid scheduler name: {scheduler_name}')
    return scheduler