#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from models import resnet34
import torch.distributed.launch
import argparse
import pickle
from modules.focal_loss import FocalLoss
from config.root_path import DATA_ROOT
import time
device =torch.device('cuda:0')


parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('-e','--epoch',type=int,metavar='',help='the number of epoch',default=100)
parser.add_argument('-d','--dataset',type=str,metavar='',help='the name of dataset')
parser.add_argument('-b','--batch_size',type=int,metavar='',help='the number of batch size',default=128)
parser.add_argument('-g','--gpu_num',type=int,metavar='',help='the number of gpu',default=8)
parser.add_argument('-n','--net',type=str,metavar='',help='network',default='resnet34')
parser.add_argument('-l','--loss_function',type=str,metavar='',help='loss function',default='CrossEntropyLoss')
parser.add_argument('-r','--result_folder',type=str,metavar='',help='the path to save result')

args = parser.parse_args()

root_path = os.path.join(DATA_ROOT,args.dataset)

print(f'Loss function = {args.loss_function}')

data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(256),
                                     transforms.Resize(152),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
        "val": transforms.Compose([
                                   transforms.CenterCrop(256),
                                   transforms.Resize(152),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
                }

batch_size = args.batch_size
gpu_num = args.gpu_num
lr = 0.0001
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process\n'.format(nw))


train_dataset = datasets.ImageFolder(os.path.join(root_path,'train'),
                                     transform=data_transform['train'])

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,                         
                          shuffle=True,
                          num_workers = nw)

val_dataset = datasets.ImageFolder(os.path.join(root_path,'val'),
                                   transform=data_transform['val'])
val_loader = DataLoader(val_dataset,
                        batch_size = batch_size,
                        shuffle=False,
                        num_workers = nw)


print(f'val_dataset length = {len(val_dataset)}')
print(f'class = {val_dataset.class_to_idx}\n')

net = resnet34()

latest_weight = f'./result/{args.result_folder}/pth/latest_8gpu.pth'
best_weight = f'./result/{args.result_folder}/pth/best_8gpu.pth'
best_acc_path = f'./result/{args.result_folder}/pth/best_8gpu.txt'


if not os.path.isfile(latest_weight):
    weight_path = 'pth/resnet34.pth'
    net.load_state_dict(torch.load(weight_path))
    print('weight = resnet34\n')

inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel,len(val_dataset.class_to_idx))# the number of class

net.to(device)

if args.loss_function == 'FocalLoss':
    loss_function = FocalLoss()
elif args.loss_function == 'CrossEntropyLoss':
    loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr = lr)


net = nn.DataParallel(net,device_ids=list( range(gpu_num) ) )


if  os.path.isfile(best_weight):
    weight_path = best_weight
    net.load_state_dict(torch.load(weight_path))
    print('weight = best\n')


with open(best_acc_path) as f:
    best_acc = f.read()

best_acc = float(best_acc)
save_path = f'result/{args.result_folder}/pth/best_{gpu_num}gpu.pth'

train_loss_list = []
val_acc_list = []

def train_epoch(epoch, train_loader, net, optimizer, loss_function, device):
    start_time = time.time()  # 记录当前 epoch 开始时间
    running_loss = 0.0
    
    for step, data in enumerate(train_loader):
        images, labels = data
        optimizer.zero_grad()
        output = net(images)
        loss = loss_function(output, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()  # 记录当前 epoch 结束时间
    epoch_time_seconds = end_time - start_time

    return running_loss, epoch_time_seconds

# 验证一个 epoch
def validate_epoch(val_loader, net, device):
    net.eval()
    acc = 0.0
    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            output = net(images.to(device))
            predict = torch.max(output, dim=1)[1]
            acc += torch.eq(predict, labels.to(device)).sum().item()

    val_acc = acc / len(val_dataset)
    return val_acc

# 主训练循环
def train(args, train_loader, val_loader, net, optimizer, loss_function, device):
    
    train_loss_list = []
    val_acc_list = []
    best_acc = 0.0
    time_list = []

    for epoch in range(args.epoch):
        print(f'Epoch:{epoch+1}')
        net.train()
        running_loss, train_time = train_epoch(epoch, train_loader, net, optimizer, loss_function, device)
        train_loss_list.append(running_loss)
        time_list.append(train_time)
        print(f"Traning time :   {train_time:.2f} seconds")

        # 验证部分
        val_acc = validate_epoch(val_loader, net, device)
        val_acc_list.append(val_acc)
        print(f'Loss :       {running_loss}')
        print(f'Val Acc :    {val_acc:.6f}')

        # 保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

            with open(best_acc_path, 'w') as f:
                f.write(str(val_acc))

        torch.save(net.state_dict(), latest_weight)

        # 保存训练进度
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
            with open(f'result/{args.result_folder}/acc_loss/train_loss_list_epoch_{epoch+1}.pkl', 'wb') as f:
                pickle.dump(train_loss_list, f)

            with open(f'result/{args.result_folder}/acc_loss/val_acc_list_epoch_{epoch+1}.pkl', 'wb') as f:
                pickle.dump(val_acc_list, f)

        # 输出训练总时长
        total_time = sum(time_list)/60
        print(f"Total time :    {total_time:.2f} min\n")
                
    


if __name__ == "__main__":

    train(args, train_loader, val_loader, net, optimizer, loss_function, device)
    print('train over')