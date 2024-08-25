import os
import time
import torch
from sklearn.metrics import accuracy_score
import torch.distributed.launch
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from models.model import get_model
from modules.train_utils import DataHandler
from modules.predict_utils import performance,top_k_accuracy,save_cm
import argparse
parser = argparse.ArgumentParser(description='预测脚本参数')
parser.add_argument('-r','--result_folder',type=str,metavar='',help='the path to save result')
parser.add_argument('-w','--weight',type=str,metavar='',help='the weight name',default='loss')
parser.add_argument('--result_root',type=str,metavar='',help='the root path to save result',default='result_pretrain')
parser.add_argument('-b','--batch_size',type=int,metavar='',help='the batch size',default=256)
parser.add_argument('-n','--net',type=str,metavar='',help='the net name',default=None)
args = parser.parse_args()

from config.root_path import DATA_ROOT,WEIGHT_ROOT

accelerator = Accelerator()
device = accelerator.device

dataset_name = '8_class_select'
data_root = os.path.join(DATA_ROOT,dataset_name)

batch_size = args.batch_size
gpu_num = accelerator.state.num_processes
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
accelerator.print('Using {} dataloader workers every process'.format(nw))

if args.net is None:
    net_name = args.result_folder.split('_')[0]
else:
    net_name = args.net
data_loader = DataHandler(data_root,batch_size,nw,net_name,args.result_root)
test_loader ,test_dataset= data_loader.test_loader,data_loader.test_dataset
class_to_idx = test_dataset.class_to_idx

accelerator.print(f'val_len: {len(test_dataset)}')

result_root = args.result_root
result_folder = args.result_folder


net =get_model(net_name,len(test_dataset.class_to_idx))

net,test_loader = accelerator.prepare(net,test_loader)

weight_root =os.path.join(WEIGHT_ROOT,result_folder)

def get_weight_path(metric):
    weight_pth = os.path.join(weight_root,f'best_{metric}.pth')
    return weight_pth

weight_path = get_weight_path(args.weight)
net.load_state_dict(torch.load(weight_path))

true_labels = []
predicted_labels = []
top3_predicted_labels = []
class_num = len(test_dataset.class_to_idx)

net.eval()
start_time = time.time()
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        _, top3_predicted = outputs.topk(3, 1, True, True)

         # 使用 accelerate 收集预测和标签
        all_predictions, all_labels = accelerator.gather_for_metrics((predicted, labels))
        all_top3_predicted = accelerator.gather_for_metrics(top3_predicted)

        true_labels.extend(all_labels.cpu().numpy())
        predicted_labels.extend(all_predictions.cpu().numpy())
        top3_predicted_labels.extend(all_top3_predicted.cpu().numpy())

if accelerator.is_main_process:
    end_time = time.time()
    print(f'Inference Time: {end_time - start_time}')

    with open('predict_time.csv','a') as f:
        f.write(f'{result_folder},{end_time - start_time}\n')

# 计算 Top-1 精确率
top1_accuracy = accuracy_score(true_labels, predicted_labels)
accelerator.print("Top-1 Accuracy:", top1_accuracy)

# 计算 Top-3 精确率
# 需要将 top3_predicted_labels 转换为每个样本的 top-3 列表
top3_lists = np.array(top3_predicted_labels).reshape(-1, 3)
top3_accuracy = top_k_accuracy(true_labels, top3_lists, 3)
accelerator.print("Top-3 Accuracy:", top3_accuracy)

save_cm(result_root,true_labels, predicted_labels, class_num, class_to_idx,result_folder)

df,per_df= performance(
    result_root,
    result_folder,
    true_labels,
    predicted_labels,
    class_num,
    class_to_idx,
    top1_accuracy,
    top3_accuracy)
accelerator.print(f'{df}\n')
accelerator.print(per_df)