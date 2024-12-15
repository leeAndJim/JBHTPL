
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class JigDataset(data.Dataset):
    def __init__(self, txt_path, img_idx, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        lines = fh.readlines()
        slices = [lines[i] for i in img_idx]
        slices = [slices[i].rstrip().split() for i in range(len(slices))]
        imgs = [(slices[i][0], int(slices[i][1])) for i in range(len(slices))]
        self.imgs = imgs
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            normalize
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize
        ])
        self.__image_transformer = transforms.Compose([
            transforms.Resize(96, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.Resize((32, 32), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename, label = self.imgs[index]
        if os.path.isabs(framename):
            img = Image.open(framename).convert('RGB')
        else:
            curr_dir = "" # os.path.dirname(__file__)
            img = Image.open(os.path.join(curr_dir, framename)).convert('RGB')
        img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        data = [tiles[t] for t in range(9)]
        data = torch.stack(data, 0)

        return data, label


    def __len__(self):
        return len(self.imgs)



def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')
