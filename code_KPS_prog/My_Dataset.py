# coding: utf-8
import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    '''
    txt_path: a file which saves paths and labels of images
    img_idx: the index list of images you want, like [1,2,3,8,9,10...]  
    '''
    def __init__(self, txt_path, img_idx, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        lines = fh.readlines()
        slices = [lines[i] for i in img_idx]
        slices = [slices[i].rstrip().split() for i in range(len(slices))]
        imgs = [(slices[i][0], int(slices[i][1])) for i in range(len(slices))]
        self.imgs = imgs  
        self.data = []
        self.targets = []
        self.generate_data()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        if os.path.isabs(fn): 
            img = Image.open(fn).convert('RGB')
        else:
            curr_dir = os.path.dirname(__file__)
            img = Image.open(os.path.join(curr_dir, fn)).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    
    def generate_data(self):
        for index in range(len(self.imgs)):
            fn, label = self.imgs[index]
            # if os.path.isabs(fn):
            #     img = Image.open(fn).convert('RGB')
            # else:
            #     curr_dir = os.path.dirname(__file__)
            #     img = Image.open(os.path.join(curr_dir, fn)).convert('RGB')

            self.data.append(fn)
            self.targets.append(label)
        
        # self.data = np.vstack(self.data).reshape((-1, 3, 32, 32))
        # self.data = self.data.transpose((0, 2, 3, 1))

    def get_cls_num_list(self):
        category_counts = {}
        for item in self.imgs:
            if item[1] in category_counts:
                category_counts[item[1]] += 1
            else:
                category_counts[item[1]] = 1
        
        return list(category_counts.values())

    def __len__(self):
        return len(self.imgs)
