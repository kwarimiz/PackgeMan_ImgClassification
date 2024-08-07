import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.35, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, input, target):
#         log_prob = F.log_softmax(input, dim=1)
#         prob = torch.exp(log_prob)
#         one_hot = F.one_hot(target, num_classes=input.size(1)).float()

#         focal_weights = (1 - prob) ** self.gamma
#         focal_weights = self.alpha * one_hot * focal_weights + (1 - self.alpha) * focal_weights
#         loss = (-focal_weights * log_prob * one_hot).sum(dim=1)
        
#         return loss.mean()

#     def backward(self):
#         # Implement backward pass here if needed
#         # Compute gradients and return them
        
#         # return grad_output
#         pass

class FocalLoss(nn.Module):
    def __init__(self, device,data_path:str, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = self.get_weights(data_path).to(device)

    def get_weights(self, data_path):
        # 计算每个类别的样本数量
        class_folders = [folder for folder in os.listdir(data_path)]
        num_per_class = [len(os.listdir(os.path.join(data_path, folder))) for folder in class_folders]
        
        # 使用对数反比例公式计算权重
        total_samples = sum(num_per_class)
        weights = [1.0 / torch.log1p(torch.tensor(total_samples / num_samples)) for num_samples in num_per_class]
        
        
        # 返回转换为tensor的权重
        return torch.tensor(weights, dtype=torch.float)

    def forward(self, input, target):
        
        alpha = self.alpha[target].to(input)
   
        log_prob = F.log_softmax(input, dim=1)
        prob = torch.exp(log_prob)
        one_hot = F.one_hot(target, num_classes=input.size(1)).float()

        focal_weights = (1 - prob) ** self.gamma
        focal_weights = alpha.unsqueeze(1) * one_hot * focal_weights
        loss = (-focal_weights * log_prob * one_hot).sum(dim=1)

        return loss.mean()