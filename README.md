## BETL

这个仓库是2024 MS文章：**Hierarchical and Progressive Learning with Key Point Sensitive Loss for Sonar Image Classification** 的实现代码。

## 运行实验

### 数据集

我们使用了NanKai Sonar Image Dataset (NKSID)和marine-debris-fls-datasets (FLSMDD)数据集来测试JBHTPL。

```
% NKSID dataset is proposed in:
@article{jiao2024open,
  title={Open-set recognition with long-tail sonar images},
  author={Jiao, Wenpei and Zhang, Jianlei and Zhang, Chunyan},
  journal={Expert Systems with Applications},
  pages={123495},
  year={2024},
  publisher={Elsevier}
}
% FLSMDD dataset is proposed in:
@inproceedings{valdenegro2021pre,
  title={Pre-trained models for sonar images},
  author={Valdenegro-Toro, Matias and Preciado-Grijalva, Alan and Wehbe, Bilal},
  booktitle={OCEANS 2021: San Diego--Porto},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

### 数据准备

首先，并把原始文件的结构调整到下面这样：

```
data
└── FLSMDD
    ├── bottle
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── can
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
└── NKSID
    ├── tire
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── floats
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
```

然后，运行下面的命令来生成 路径-标签 列表（train.txt）和十次五折交叉验证的序号列表（kfold_train.txt, kfold_val.txt）。

```shell
# generate data direction-label list, use KLSG dataset as an example 
cd ./tool/
python generate_dir_lbl_list.py --dataset NKSID
# generate 10-trail 5-fold cross-validation index list, use KLSG dataset as an example 
python generate_kfold_idx_list.py --dataset NKSID
```

现在，你应该得到了这样的文件结构：

```
data
└── FLSMDD
    ├── train.txt
    ├── kfold_train.txt
    ├── kfold_val.txt
    ├── bottle
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── can
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
└── NKSID
    ├── train.txt
    ├── kfold_train.txt
    ├── kfold_val.txt
    ├── tire
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── floats
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
```

### 源预训练

要训练模型之前，先需要进行源预训练，命令如下：

```python
python pretrain_resnet18.py
```

### 训练

要训练BETL，这里是一个快速开始的命令：

```shell
# Demo: training on KLSG
cd ./code/
python betl.py --dataset NKSID
```

通过下面的命令来对BETL进行十次五折交叉验证训练：

```shell
# Demo: training on KLSG, using resnet18 as backbone
cd ./tool/
./auto_run.sh ../code/betl.py NKSID resnet18
```

### 结果分析

在通过十次五折交叉验证训练完BETL之后，您会得到y_hat, y_true和logits结果，默认存放在`"/output/result/{dataset}/{method}/{backbone}/"`路径下，例如，`"/output/result/NKSID/Jig_KPS_prog_2linears/resnet18/y_hat.txt"`。然后，您可以使用下面的命令来得到Gmean、Macro-F1、混淆矩阵和P-R曲线结果：

```shell
# Demo: analyzing on NKSID, using resnet18 as backbone
cd ./code/
python analyse_result.py --dataset NKSID --method Jig_KPS_prog_2linears --backbone resnet18 --get_conf_matrix True --get_pr True
```

##  引用

如果您觉得这份代码对您的研究有帮助，请考虑引用我们：

```
@article{chen2024hierarchical,
  title={Hierarchical and progressive learning with key point sensitive loss for sonar image classification},
  author={Chen, X. and Tao, H. and Zhou, H. and others},
  journal={Multimedia Systems},
  year={2024},
  volume={30},
  pages={380},
  doi={10.1007/s00530-024-01590-8}
}
```

代码部分参考

```
@article{jiao2022sonar,
  title={Sonar Images Classification While Facing Long-Tail and Few-Shot},
  author={Jiao, Wenpei and Zhang, Jianlei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  volume={60},
  pages={1-20},
  publisher={IEEE}
}
```
