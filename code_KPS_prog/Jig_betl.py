# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩
@time: 2022/04/02
@file: betl.py
@author: Jorwnpay
@contact: jwp@mail.nankai.edu.cn
"""
import sys

import torch
from torch import optim
import os
from my_utils import *
import pretrainedmodels as ptm
import pretrainedmodels.utils as utils
import argparse
# from model.Jigsaw_2x2 import Network
from model.JigRawModel import Network
from losses.loss import KPSLoss, IIFLoss, CE_ABC_Loss, KPS_ABC_Loss
# from sampler import BalancedDatasetSampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='betl')
    parser.add_argument('--dataset', type=str, default='NKSID')
    parser.add_argument('--p_value', type=int, default=0)
    parser.add_argument('--k_value', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--save_prop', type=float, default=0.6)
    parser.add_argument('--save_results', type=str, default='True')
    parser.add_argument('--save_models', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


if __name__ == '__main__':
    # get args
    args = parse_args()
    set_seed(42)

    # set params
    dataset = args.dataset
    p_v = args.p_value
    k_v = args.k_value
    backbone = args.backbone
    if dataset == 'KLSG':
        nb_classes = 2
        subset_num = 6
    elif dataset == 'LTSID':
        nb_classes = 8
        subset_num = 19
    elif dataset == 'FLSMDD':
        nb_classes = 10
        subset_num = 7
    elif dataset == 'NKSID':
        nb_classes = 8
        subset_num = 19
    elif dataset == 'NKSID_part':
        nb_classes = 7
        subset_num = 19
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')

    if backbone in ['vgg16', 'vgg19']:
        feature_dim = 4096
        lr = 0.005
    elif backbone in ['resnet18', 'resnet34']:
        feature_dim = 512
        lr = 0.01
    elif backbone in ['resnet50']:
        feature_dim = 2048
        lr = 0.01
    else:
        print(f'ERROR! BACKBONE {backbone} IS NOT EXIST!')

    # --------- phase 1: transfer learning --------
    print('--------- phase 1: transfer learning --------')
    # get backbone model
    is_train = True
    # try:
    #     model = get_pretrained_model(backbone, is_train)
    # except:
    #     print(f'THE INPUT BACKBONE {backbone} IS NOT EXIST!')
    # model = model_fc_fix(model, nb_classes)
    model = Network(classes=nb_classes)
    # 路径具体改
    pretrained_weight = torch.load("/gemini/code/model_resnet18.pt")

    model_dict = model.state_dict()
    partial_weights = {k: v for k, v in pretrained_weight.items() if k in model_dict}
    model_dict.update(partial_weights)
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)

    # get data iter
    sample_type = 'uniform'
    batch_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../data', dataset)
    kfold_train_idx, kfold_val_idx = get_kfold_img_idx(p=p_v, k=k_v, dataset=dataset, sample_type=sample_type)
    train_iter, val_iter, cls_num_list, per_cls_weights = get_kfold_img_iters(batch_size, data_dir, kfold_train_idx, kfold_val_idx, mean, std)

    # set loss and optimizer
    print(per_cls_weights)
    loss_kps = KPSLoss(cls_num_list=cls_num_list, max_m=0.2, s=1, weighted=False, weight=per_cls_weights)
    loss_ce = torch.nn.CrossEntropyLoss()
    # loss_abc = KPS_ABC_Loss(cls_num_list=cls_num_list)
    # loss = IIFLoss(dataset=cls_num_list, weight=per_cls_weights)
    output_params = list(map(id, model.classifier1.parameters()))
    output_params.extend(list(map(id, model.classifier2.parameters())))
    output_params.extend(list(map(id, model.classifier3.parameters())))
    feature_params = filter(lambda p: id(p) not in output_params, model.parameters())
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': model.classifier1.parameters(), 'lr': lr * 10},
                           {'params': model.classifier2.parameters(), 'lr': lr * 10},
                           {'params': model.classifier3.parameters(), 'lr': lr * 10}],
                          lr=lr, weight_decay=0.001)

    # finetuning model
    num_epochs = 5
    model = train(train_iter, val_iter, model, loss_kps, loss_ce, optimizer, device, num_epochs)
    # save_dir = "/gemini/code/weights/"+dataset+"/"+str(p_v)+"_"+str(k_v)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(model.state_dict(), save_dir+"/model.pt")
    
    # --------- phase 2: balanced ensemble learning --------
    print('--------- phase 2: balanced ensemble learning --------')
    # load pretrained model
    model.last_linear = ptm.utils.Identity()  # remove the linear layer of pretrained backbone
    # load datasets
    balance_train_idxes = [get_kfold_img_idx(p=p_v, k=k_v, dataset=dataset, sample_type='balance')[0] for i in
                           range(subset_num)]
    balance_train_iters, _, _, _ = get_kfold_img_iters(batch_size, data_dir, balance_train_idxes, kfold_val_idx, mean, std)

    # load fcs
    fcs = get_fcs(feature_dim, nb_classes, subset_num)

    # balanced muti-branch training
    lr2 = 0.01
    loss2 = torch.nn.CrossEntropyLoss()
    optimizer2 = []
    num_epochs2 = 2
    for i in range(len(fcs)):
        optimizer2.append(optim.SGD(fcs[i].parameters(), lr=lr2 * 10))
        fcs[i] = train_mul(balance_train_iters[i], model, fcs[i], loss2, optimizer2[i], device, num_epochs2)
        # torch.save(fcs[i].state_dict(), save_dir+"/fcs_"+str(i)+".pt")
        
    # --------- phase 3: ensemble pruning --------
    print('--------- phase 3: ensemble pruning --------')
    save_prop = args.save_prop
    saved_fcs_idx = ensemble_pruning(train_iter, model, fcs, _p=save_prop)
    print(f'Saved fully connected layer indexes after ensemble pruning: {saved_fcs_idx}')
    saved_fcs = [fcs[i].eval() for i in saved_fcs_idx]

    # save results by weight_averaging after pruning
    if args.save_results in ['True', 'true']:
        _, y_hat, y_true, logits = evaluate_gmean_optional(val_iter, model, saved_fcs, 'soft_voting',
                                                           device=device, if_get_logits=True)
        save_dir = os.path.join(curr_dir, '../output/result', dataset, 'Jig_KPS_prog_DAH_2linears', backbone)
        # save_dir = os.path.join('/gemini/output/', '/result', dataset, 'Jig_betl', backbone)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_result_to_file(save_dir, y_hat, y_true, logits, p=p_v, k=k_v)

    # save models
    if args.save_models in ['True', 'true']:
        fus_fc = get_fusion_fc(saved_fcs)
        model.last_linear = fus_fc
        model_folder = os.path.join(curr_dir, f'../output/model/{dataset}')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_dir = os.path.join(model_folder, f'p{p_v}_k{k_v}_{backbone}_baseline.pth')
        torch.save(model.state_dict(), model_dir)
