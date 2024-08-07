import os
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
args = parser.parse_args()
from config.root_path import DATA_ROOT,WEIGHT_ROOT

accelerator = Accelerator()
device = accelerator.device

dataset_name = '8_class_select'
data_root = os.path.join(DATA_ROOT,dataset_name)

batch_size = 512
gpu_num = accelerator.state.num_processes
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
accelerator.print('Using {} dataloader workers every process'.format(nw))

net = args.result_folder.split('_')[0]
data_loader = DataHandler(data_root,batch_size,nw,net)
test_loader ,test_dataset= data_loader.test_loader,data_loader.test_dataset
class_to_idx = test_dataset.class_to_idx

accelerator.print(f'val_len: {len(test_dataset)}')

result_folder = args.result_folder
net_name = result_folder.split('_')[0]


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
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        _, top3_predicted = outputs.topk(3, 1, True, True)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        top3_predicted_labels.extend(top3_predicted.cpu().numpy())

# 计算 Top-1 精确率
top1_accuracy = accuracy_score(true_labels, predicted_labels)
print("Top-1 Accuracy:", top1_accuracy)

# 计算 Top-3 精确率
# 需要将 top3_predicted_labels 转换为每个样本的 top-3 列表
top3_lists = np.array(top3_predicted_labels).reshape(-1, 3)
top3_accuracy = top_k_accuracy(true_labels, top3_lists, 3)
print("Top-3 Accuracy:", top3_accuracy)

save_cm(true_labels, predicted_labels, class_num, class_to_idx,result_folder)

df,per_df= performance(
    result_folder,true_labels, predicted_labels,class_num,class_to_idx,top1_accuracy,top3_accuracy)
print(f'{df}\n')
print(per_df)