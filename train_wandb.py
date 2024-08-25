#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torch.nn as nn
from models.model import get_model
import torch.distributed.launch
from modules.focal_loss import FocalLoss
import time
from accelerate import Accelerator
# from accelerate.utils import LoggerType
import logging
from accelerate.logging import get_logger
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from config.cmd_args import parse_args
from config.root_path import DATA_ROOT,WEIGHT_ROOT
from modules.train_utils import DataHandler
from modules.predict_utils import read_metrics
from modules.train_component import get_optimizer,get_scheduler,get_loss_function
import evaluate
# import wandb
gpu_name = torch.cuda.current_device()

args = parse_args()

config = {
    "model": args.net,
    "start_epoch": args.start_epoch,
    'end_epoch': args.end_epoch,
    "start_lr": 1e-4,
    "loss_function": args.loss_function,
    "batch_size": args.batch_size,
    'optimizer': args.optimizer,
    'scheduler': args.scheduler,
    'data_sampler': args.sampler,
    'weight': args.weight,
}
accelerator = Accelerator(log_with="wandb")

accelerator.init_trackers(project_name='pretrain_model',
                          config=config
                          )


device = accelerator.device
num_processes = accelerator.state.num_processes

logging.basicConfig(
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename=f'{args.result_root}/{args.result_folder}/train.log',
                    filemode='w')

logger = get_logger('train.log',log_level='INFO')
# logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)     # 设置日志级别，把不许要的log信息隔离掉


data_root = os.path.join(DATA_ROOT,args.dataset)

batch_size = args.batch_size
gpu_num = num_processes
lr = 1e-4
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

# load data
data_loader = DataHandler(data_root,batch_size,nw,args.net,args.sampler)
train_loader, val_loader = data_loader.train_loader, data_loader.val_loader
# accelerator.log({'class_weight':data_loader.sample_weights})
val_dataset = data_loader.get_val_dataset()
val_len = len(val_dataset)

# log base information
logger.info('Data Information:\n')
logger.info(f'num_workers = {nw}')
logger.info(f'val_dataset length = {len(val_dataset)}')
logger.info(f'class = {val_dataset.class_to_idx}\n')
logger.info(f'start epoch = {args.start_epoch}')
logger.info(f'gup num = {gpu_num}')

net =get_model(args.net,len(val_dataset.class_to_idx),args.weight)

# define loss function ,optimizer,scheduler

loss_function = get_loss_function(args.loss_function)

optimizer = get_optimizer(args.optimizer,net,lr)

if args.scheduler:
    scheduler = get_scheduler(args.scheduler,optimizer)
else:
    scheduler = None

# define save path

weight_root = os.path.join(WEIGHT_ROOT,args.result_folder)

def get_weight_path(metric):
    weight_pth = os.path.join(weight_root,f'best_{metric}.pth')
    return weight_pth

best_precision_weight = get_weight_path('precision')

latest_weight = os.path.join(weight_root,f'latest.pth')

best_score_path = os.path.join(weight_root,'best_score.csv')

# pack all thing in accelerator 

if scheduler:
    net, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(
        net, optimizer, train_loader, scheduler, val_loader
    )
else:
    net, optimizer, train_loader, val_loader = accelerator.prepare(
        net, optimizer, train_loader, val_loader
    )

# load weight
weight_info = args.weight
if args.weight == 'default':
    if os.path.exists(best_precision_weight):
        net.load_state_dict(torch.load(best_precision_weight))
        weight_info = 'Best'
    elif os.path.exists(latest_weight):
        net.load_state_dict(torch.load(latest_weight))
        weight_info = 'Latest'
logger.info(f'Weight:         :   {weight_info}\n')


# traning one  epoch
def train_epoch(train_loader, net, optimizer, loss_function):

    start_time = time.time()  # 记录当前 epoch 开始时间
    running_loss = 0.0
    
    for data in train_loader:
        images, labels = data
        optimizer.zero_grad()
        output = net(images)
        loss = loss_function(output, labels)
        accelerator.backward(loss)
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()

    end_time = time.time()  # 记录当前 epoch 结束时间
    epoch_time_seconds = end_time - start_time

    current_lr = optimizer.param_groups[0]['lr'] if not scheduler else scheduler.get_last_lr()[0]

    average_loss = running_loss / len(train_loader) # 计算平均 loss
    return average_loss, epoch_time_seconds,current_lr

# 验证一个 epoch
def validate_epoch(val_loader, net):

    net.eval()
    acc = evaluate.load("./evaluate/metrics/accuracy")
    precision = evaluate.load("./evaluate/metrics/precision")
    recall = evaluate.load("./evaluate/metrics/recall")
    
    with torch.no_grad():
        for images,labels in val_loader:
            outputs = net(images)
            predictions = outputs.argmax(-1)
        
            all_predictions, all_labels = accelerator.gather_for_metrics((predictions, labels))
            
            val_acc = acc.compute(references=all_labels,predictions=all_predictions)['accuracy']
            val_precision = precision.compute(predictions=all_predictions, references=all_labels,average='weighted',zero_division=0)['precision']
            val_recall = recall.compute(predictions=all_predictions, references=all_labels,average='weighted',zero_division=0)['recall']
           
    return val_acc,val_precision,val_recall

# 主训练循环
def train(train_loader, val_loader, net, optimizer, loss_function): 
        
    time_list = []

    for epoch in range(args.start_epoch,args.end_epoch):
        logger.info(f'Epoch:           {epoch+1}')
        net.train()
        average_loss, train_time,current_lr= train_epoch(train_loader, net, optimizer, loss_function)

        accelerator.log({'Train loss':average_loss}, epoch)
        accelerator.log({'Train time': train_time}, epoch)
        accelerator.log({'Learning Rate': current_lr},epoch)  

        time_list.append(train_time)
        logger.info(f"Traning time :   {train_time:.2f} seconds")

        # 验证部分
        val_accuracy,val_precision,val_recall= validate_epoch(val_loader, net)

    
        accelerator.log({'Accuracy':val_accuracy}, epoch)
        accelerator.log({'Precision': val_precision}, epoch)
        accelerator.log({'Recall':val_recall}, epoch)

        
        logger.info(f'Train Loss      :   {average_loss:.6f}')
        logger.info(f'Val Accuracy    :   {val_accuracy:.6f}')
        logger.info(f'Val Precision   :   {val_precision:.6f}')
        logger.info(f'Val Recall      :   {val_recall:.6f}')

        # 保存模型和得分
        current_scores = {'accuracy': val_accuracy, 'precision': val_precision, 'recall': val_recall}
        

        if accelerator.is_main_process:
            best_scores = pd.read_csv(best_score_path)
            for metric in ['accuracy', 'precision', 'recall']:
                if current_scores[metric] > best_scores.loc[0,metric]:

                    # 更新CSV文件
                    best_scores[metric] = current_scores[metric]  
                    best_scores.to_csv(best_score_path, index=False)

                    # 保存模型
                    weight_path = get_weight_path(metric)
                    accelerator.save(net.state_dict(), weight_path)
                # 保存最小loss对应的权重
            if average_loss < best_scores.loc[0, 'loss']:
                best_scores['loss'] = average_loss
                best_scores.to_csv(best_score_path, index=False)
                weight_path = get_weight_path('loss')
                accelerator.save(net.state_dict(), weight_path)

        accelerator.save(net.state_dict(), latest_weight)

        # 输出训练总时长
        total_time = sum(time_list)/60
        logger.info(f"Total time :    {total_time:.2f} min\n")
    
    accelerator.end_training()
    
if __name__ == "__main__":
    accelerator.print('start training')
    try:
        train(train_loader, val_loader, net, optimizer, loss_function)
    except Exception as e:
        print(e)
        logger.error(e,exc_info=True)
    logger.info('train over')
