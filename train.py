#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torch.nn as nn
from models.model import get_model
import torch.distributed.launch
import pickle
from config.root_path import DATA_ROOT,WEIGHT_ROOT
import time
from config.cmd_args import parse_args
device =torch.device('cuda:0')
from modules.train_utils import DataHandler
from modules.train_component import get_optimizer,get_loss_function

args = parse_args()

data_root = os.path.join(DATA_ROOT,args.dataset)
batch_size = args.batch_size
gpu_num = args.gpu_num
lr = 0.0001
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process\n'.format(nw))

# load data
data_loader = DataHandler(data_root,batch_size,nw,args.net,args.sampler)

train_loader, val_loader = data_loader.train_loader, data_loader.val_loader

val_dataset = data_loader.get_val_dataset()
val_len = len(val_dataset)

print(f'val_dataset length = {len(val_dataset)}')
print(f'class = {val_dataset.class_to_idx}\n')

net = get_model(args.net,len(val_dataset.class_to_idx),args.weight)
loss_function = get_loss_function(args.loss_function)
optimizer = get_optimizer(args.optimizer,net,lr)

# define save path

weight_root = os.path.join(WEIGHT_ROOT,args.result_folder)

def get_weight_path(metric):
    weight_pth = os.path.join(weight_root,f'best_{metric}.pth')
    return weight_pth

best_acc_weight = get_weight_path('acc')

latest_weight = os.path.join(weight_root,f'latest.pth')

best_score_path = os.path.join(weight_root,'best_score.csv')


net.to(device)


net = nn.DataParallel(net,device_ids=list( range(gpu_num) ) )



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

    for epoch in range(args.start_epoch,args.end_epoch):
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
            torch.save(net.state_dict(), best_acc_weight)

            with open(best_score_path, 'w') as f:
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
    