'''
    inputed image size 224x224
    jigsaw size 2x2
    model resnet 18
    datasets turntable-cropped
'''
import sys
import time

import numpy as np
import torch
from torch import cat
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
# from models.JigsawModel import Network
from models.Resnet_CCBC import resnet18, resnet50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--dataset', type=str, default='KLSG')
    parser.add_argument('--data', type=str, default="./data", help='Path to Imagenet folder')
    parser.add_argument('--p_value', type=int, default=0)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for SGD optimizer')
    parser.add_argument('--k_value', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_results', type=str, default='True')
    parser.add_argument('--save_models', type=str, default='False')
    args = parser.parse_args()
    return args

class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()

        self.conv = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(512, 64))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('BN6_s1', nn.BatchNorm1d(64))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(64, 32))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('BN7', nn.BatchNorm1d(32))
        self.fc7.add_module('drop7', nn.Dropout(p=0.3))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(32, classes))
        # self.classifier.add_module('softmax1', nn.Softmax(dim=1))

        # self.apply(weights_init)

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):

        B, C, H, W = x.size()
        # x = x.transpose(0, 1)
        x_list = []
        _,_,_,_,z = self.conv(x)
        # for i in range(4):
        #     _,_,_,_,z = self.conv(x[i,:,0:,:,:])
        #     z = self.avgpool(z)
        #     z = z.view([B, 1, -1])
        #     x_list.append(z)

        x = self.avgpool(z) # cat(x_list, 1)
        x = self.fc6(x.view(B, -1))
        x = self.fc7(x)
        x = self.classifier(x)

        return x

class MyDataset(data.Dataset):
    def __init__(self, data_path, txt_list, classes=1000, train_if=True):
        self.data_path = data_path
        self.names, self.labels = self.__dataset_info(txt_list)
        self.N = len(self.names)
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
        self.train_if = train_if
        self.__image_transformer = transforms.Compose([
            transforms.Resize(224, Image.BILINEAR),
            transforms.CenterCrop(224)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(112),
            transforms.Resize((112, 112), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        self.permutations = self.__retrive_permutations(classes)

    def __getitem__(self, index):
        framename = self.names[index]
        label = self.labels[index]
        img = Image.open(framename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')
        img = self.__image_transformer(img)
        s = float(img.size[0]) / 2
        a = s / 2
        tiles = [None] * 4
        for n in range(4):
            i = n / 2
            j = n % 2
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

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(4)]
        data = torch.stack(data, 0)
        unfolded_data = nn.functional.unfold(data, kernel_size=(112, 112), stride=(112, 112))
        reshaped_data = unfolded_data.view(3, 224, 224)

        return reshaped_data, int(order), tiles


    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        all_perm = np.load('./permutations/permutations_2x2_hamming_max_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def test(net,criterion,val_loader,steps):
    print('Evaluating network.......')
    test_loss = 0
    net.eval()
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = net(images)
            # print(outputs, labels)
            l = criterion(outputs, labels)
            test_loss += l

    net.train()
    return test_loss


class Net(nn.Module):
    def __init__(self, model, nb_classes, dim_feats):
        super(Net, self).__init__()
        # ??model??1?
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(dim_feats, nb_classes)  # ????????????????????11?

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x

def train(train_iter, val_iter, model, loss, optimizer, device, num_epochs):
    print("training on", device)
    batch_count = 0
    record_acc = 100
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y, z in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            # print(y_hat, y)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = test(net=model,criterion=criterion,val_loader=val_iter,steps=epoch)
        if test_acc < record_acc:
            record_acc = test_acc
            torch.save(model.state_dict(), "model_resnet18.pt")
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == "__main__":
    # get args
    args = parse_args()

    # get model
    dataset = args.dataset
    p_v = args.p_value
    k_v = args.k_value
    backbone = args.backbone

    # ????
    # model = torchvision.models.resnet18(pretrained=False)
    # model = Net(model, nb_classes=18, dim_feats=model.fc.in_features).to(device).train()
    nb_classes = 5
    # backbone_dac = DAC50_CC3(pretrained=False)
    # model = Classifier(backbone_dac, nb_classes).to(device)
    model = Network(classes=nb_classes).to(device)
    # print(model)
    # ????
    trainpath = args.data + '/turntable-cropped'
    train_imgs = MyDataset(trainpath, args.data+'/train.txt',
                           classes=nb_classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_imgs,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=4)
    valpath = args.data+'/turntable-cropped'
    val_data = MyDataset(valpath, args.data+'/val.txt',
                         classes=nb_classes, train_if=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-4)

    num_epochs = 100
    train(train_loader, val_loader, model, criterion, optimizer, device, num_epochs)
