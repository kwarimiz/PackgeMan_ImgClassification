import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,WeightedRandomSampler

from torchvision.datasets import ImageFolder
from PIL import Image

class DataHandler:
    def __init__(self, root_path, batch_size, nw,net):
        self.root_path = root_path
        self.batch_size = batch_size
        self.nw = nw
        self.net = net
        if self.net == 'maxvit':
            resize_num = 224
        elif self.net == 'swin':
            resize_num = 256
        else:
            resize_num = 152
        self.data_transform = {
            "train": transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(resize_num),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(resize_num),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.train_dataset = datasets.ImageFolder(os.path.join(self.root_path, 'train'),
                                                   transform=self.data_transform['train'])
        self.val_dataset = datasets.ImageFolder(os.path.join(root_path ,'val'),
                                   transform=self.data_transform['val'])
        
        # Compute weights for each class
        class_weights = self.compute_class_weights(self.train_dataset)
        self.sample_weights = [class_weights[label] for _, label in self.train_dataset.samples]

        # Create samplers
        self.train_sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True)


        self.train_loader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                # shuffle=True, 
                                num_workers=self.nw,
                                sampler=self.train_sampler
                                )
        

        self.val_loader = DataLoader(self.val_dataset,
                                    batch_size = batch_size,
                                    shuffle=False,
                                    num_workers = nw)
        
        self.test_dataset = datasets.ImageFolder(os.path.join(root_path ,'test'),
                                   transform=self.data_transform['val'])

        self.test_loader = DataLoader(self.test_dataset,
                                    batch_size = batch_size,
                                    shuffle=False,
                                    num_workers = nw)
        
    
    def compute_class_weights(self, dataset):
        # Count each class
        class_count = [0] * len(dataset.classes)
        for _, index in dataset.samples:
            class_count[index] += 1
        
        # Compute weight for each class (inverse frequency)
        total_count = sum(class_count)
        class_weights = [total_count / count for count in class_count]
        
        # # Normalize weights (optional)
        # max_weight = max(class_weights)
        # class_weights = [weight / max_weight for weight in class_weights]
        
        return class_weights
        
    def update_transform(self, transform_name, new_transform):
        self.data_transform[transform_name] = new_transform

    def get_val_dataset(self):
        return self.val_dataset
    
    # def get_test_dataset(self):
    #     return self.test_dataset
        

class GetNameDataset(ImageFolder):

    def __getitem__(self, idx):
        
        img_path,target = self.samples[idx]

        img = self.loader(img_path)
        transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(152),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        img = transform(img)

        img_name = os.path.basename(img_path)
        return img, target, img_name




