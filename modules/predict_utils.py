
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import shutil
import torch
import torch.distributed.launch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import get_model
from modules.train_utils import DataHandler,GetNameDataset
from config.root_path import DATA_ROOT,WEIGHT_ROOT

device =torch.device('cuda:0')

class ModelEvaluator:
    def __init__(
        self, 
        result_folder: str , 
        loader_type:  str,
        data_root: str = DATA_ROOT, 
        batch_size: int = 512,
        nw: int = 8, 
        device = torch.device('cuda:0'), 
        gpu_num: int =8,
        weight: str ='best'
    ):
        self.result_folder = result_folder
        self.loader_type = loader_type
        self.data_root = data_root
        self.batch_size = batch_size
        self.nw = nw
        self.device = device
        self.gpu_num = gpu_num
        self.weight = weight
        self.model_name = result_folder.split('_')[0]
        self.val_dataset, self.val_loader = self.get_data_loader(loader_type)

    def get_data_loader(self, loader_type):
        if loader_type == 'test':               # 只取测试集
            data_loader = DataHandler(self.data_root, self.batch_size, self.nw,self.model_name)
            val_dataset = data_loader.test_dataset
            val_loader = data_loader.test_loader
        elif loader_type == 'del_img':          # 取所有数据集和数据集的图片名
            val_dataset = GetNameDataset(os.path.join(self.data_root, 'total'))
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.nw)
        return val_dataset, val_loader

    def load_model(self,model_name=None):
        if model_name == None:
            model_name = self.result_folder.split('_')[0]
        net = get_model(model_name, len(self.val_dataset.class_to_idx))
        net.to(self.device)
        net = torch.nn.DataParallel(net, device_ids=list(range(self.gpu_num)))
        weight_path = self.get_weight_path()
        print(f'Loading model from {weight_path.split("/")[-1]}')   
        net.load_state_dict(torch.load(weight_path))
        return net
    
    def get_weight_path(self):
        weight_root = os.path.join(WEIGHT_ROOT, self.result_folder)
        if self.weight=='latest':
            weight_path = os.path.join(weight_root, 'latest.pth')
        else:
            weight_path = os.path.join(weight_root, f'best_{self.weight}.pth')
        return weight_path
    
    def evaluate(self):
        net = self.load_model()
        if self.loader_type == 'test':
            true_labels = []
            predicted_labels = []
            top3_predicted_labels = []
            class_num = len(self.val_dataset.class_to_idx)

            net.eval()
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    _, top3_predicted = outputs.topk(3, 1, True, True)

                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())
                    top3_predicted_labels.extend(top3_predicted.cpu().numpy())

            return true_labels, predicted_labels, top3_predicted_labels, class_num, self.val_dataset.class_to_idx

        elif self.loader_type == 'del_img':
            wrong_img = []
            class_dict = {v: k for k, v in self.val_dataset.class_to_idx.items()}

            net.eval()
            with torch.no_grad():
                for data in tqdm(self.val_loader):
                    images, labels, img_name = data
                    output = net(images.to(self.device))
                    predict = torch.max(output, dim=1)[1]
                    diff = (predict != labels.to(self.device))

                    for i, result in enumerate(diff):
                        if result:
                            wrong_id = predict[i].item()
                            true_id = labels[i].item()
                            wrong_class = class_dict[wrong_id]
                            true_class = class_dict[true_id]

                            wrong_img.append((img_name[i], wrong_class, true_class))

            return wrong_img,class_dict

def del_wrong_img(batch,total_path,wrong_img_savepath):
    
    wrong_img_file = f'./wrong_img/batch_{batch}.pkl'
    # Load the wrong_img list from the pickle file
    
    with open(wrong_img_file, 'rb') as f:
        wrong_img = pickle.load(f)

    # Create a folder for the current batch
    batch_folder = os.path.join(wrong_img_savepath, f'batch_{batch}')
    os.makedirs(batch_folder, exist_ok=True)

    # Move the images to the corresponding class folders
    num_images_removed = 0
    class_counts = {}
    for img_name, wrong_class, true_class in wrong_img:
        class_folder = os.path.join(batch_folder, true_class)
        os.makedirs(class_folder, exist_ok=True)
        img_path = os.path.join(total_path, true_class, img_name)
        shutil.move(img_path, class_folder)
        num_images_removed += 1

        # Update class counts
        if true_class in class_counts:
            class_counts[true_class] += 1
        else:
            class_counts[true_class] = 1

    # Write the log file
    log_file = os.path.join(batch_folder, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Total images removed: {num_images_removed}\n')
        for class_folder in os.listdir(batch_folder):
            if class_folder == 'log.txt':
                continue
            class_path = os.path.join(batch_folder, class_folder)
            num_images = len(os.listdir(class_path))
            f.write(f'{class_folder}: {num_images} images removed\n')

def show_wrong_img(total_path:str,wrong_class:str):

    # show error classifacation 
    rows, cols = 4, 4

    # 创建一个新的图形
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    data = random.sample(wrong_class,rows*cols)
    for i, (img_name, wrong_class,true_class) in enumerate(data):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        img_path = os.path.join(total_path,true_class,img_name)
        # 读取图像文件
        img = mpimg.imread(img_path)

        # 显示图像
        ax.imshow(img)
        true_class = true_class.split('_')[0]
        wrong_class = wrong_class.split('_')[0]
        ax.set_title(f'T :{true_class}  F :{wrong_class}')
        ax.axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.show()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def show_acc_loss(result_folder,epoch_num:None):
    data_folder = f'./result/{result_folder}/acc_loss'
    if epoch_num == None:
        epoch_num_list = [item.split('.')[0].split('_')[-1] for item in os.listdir(data_folder)]
        epoch_num = max(epoch_num_list)

    train_loss = load_pickle(os.path.join(data_folder, f'train_loss_list_epoch_{epoch_num}.pkl'))

    val_acc = load_pickle(os.path.join(data_folder, f'val_acc_list_epoch_{epoch_num}.pkl'))
    val_acc = [x*100 for x in val_acc]

    # 画出损失和准确率图
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training Loss and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/{result_folder}/acc_loss.png')
    plt.show()

def performance(
        result_folder,
        true_labels,
        predicted_labels,
        class_num,class_to_idx,
        top1_accuracy,
        top3_accuracy):
    # 计算宏平均（Micro-average）指标
    macro_precision, macro_recall, macro_f1, _ = score(
    true_labels, predicted_labels, average='macro')

    # 计算微平均（Micro-average）指标
    micro_precision, micro_recall, micro_f1, _ = score(
        true_labels, predicted_labels, average='micro')
    data = {
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Macro Average': [macro_precision, macro_recall, macro_f1],
        'Micro Average': [micro_precision, micro_recall, micro_f1],
    }

    cm = confusion_matrix(true_labels, predicted_labels, labels=range(class_num))
    # 计算每个类别的精确率、召回率和 F1 分数
    precision, recall, f1, _ = score(
        true_labels,
        predicted_labels, 
        average=None, 
        labels=range(class_num)
        )
    # 展示每个类别的性能指标
    performance_data = {
        'Label': [k for k in class_to_idx],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
}
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_latex(f'result/{result_folder}/performance.tex', index=False)
    # 创建混淆矩阵 DataFrame
    cm_df = pd.DataFrame(cm, index=[k for k in class_to_idx], columns=[k for k  in class_to_idx])
    cm_df.to_csv(f'result/{result_folder}/confusion_matrix.csv')
    # 创建 DataFrame
    df = pd.DataFrame(data)
    df.loc[len(df)]=['Top-1 Accuracy',top1_accuracy,'']
    df.loc[len(df)]=['Top-3 Accuracy',top3_accuracy,'']
    df.to_latex(f'result/{result_folder}/macro_micro.tex', index=False)
    return df,performance_df
    

def read_metrics(file_path):
    data = pd.read_csv(file_path)
    # 假设数据在第一行，且每个字段的列名是'Accuracy', 'Precision', 'Recall'
    accuracy = data.at[0, 'accuracy']
    precision = data.at[0, 'precision']
    recall = data.at[0, 'recall']
    return accuracy, precision, recall


def top_k_accuracy(true_labels, predicted_labels, k):
    top_k_correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true in pred[:k]:
            top_k_correct += 1
    return top_k_correct / len(true_labels)

def save_cm(true_labels, predicted_labels, class_num, class_to_idx,result_folder):
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(class_num))
    fig, ax = plt.subplots(figsize=(11,11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[k for k, _ in class_to_idx.items()], 
                yticklabels=[k for k, _ in class_to_idx.items()], ax=ax)
    ax.set_xlabel('Predicted Labels',labelpad=15)
    ax.set_ylabel('True Labels',labelpad=10)
    ax.set_title(f'{result_folder}')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout(pad=3.0)
    # 调整标签字体大小
    plt.yticks(fontsize=8.5)  
    plt.xticks(fontsize=8.5)
    # plt.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.8)

    plt.savefig(f'result/{result_folder}/confusion_matrix.png',dpi=320)
    plt.show()

