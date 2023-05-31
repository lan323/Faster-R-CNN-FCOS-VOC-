#神经网络与深度学习————第二次作业

1.使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。

2.在VOC数据集上训练并测试目标检测模型 Faster R-CNN 和 FCOS；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；
两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像；

第一个实验对应imc代码包
第二个实验对应faster-rcnn代码包


##第一个实验

运用imc代码包中的train.py进行模型的训练和测试
visualization.py用于可视化cutmix, cutout, mixup等结果
在imc的csv包中存储有epoch,train_loss,train_acc,test_loss,test_acc对应的数据，可以用于分析

在运用train.py训练或测试时，生成的模型会保存在imc文件下，同时也可以将训练好的模型放于imc文件中，用于预测、可视化等分析。

本实验1采取resnet-18模型，方法分为：baseline、cutout、mixup和cutmix。
对应的terminal运行命令为：

baseline：python imc/train.py --method baseline --dataset cifar100 --model resnet18 --epochs 200

cutout：python imc/train.py --method cutout --dataset cifar100 --model resnet18 --data_augmentation --epochs 200

mixup：python imc/train.py --method mixup --dataset cifar100 --model resnet18 --data_augmentation --epochs 200

cutmix：python imc/train.py --method cutmix --dataset cifar100 --model resnet18 --data_augmentation --epochs 200

train.py与visualization.py对应的路径可以自行修改

以下链接为百度网盘链接：（训练完成的模型通过网盘下载）
链接：https://pan.baidu.com/s/12CWXCuNDYPj4gXezUaCVpA?pwd=6666 
提取码：6666


##第二个实验

目前现存的代码基本要用到GPU，本代码支持没有GPU的电脑进行训练。

以下链接为百度网盘链接：（存放有训练好的模型与数据集）
链接：https://pan.baidu.com/s/1Jt7cK_vDzFDp-GKwo_CPVA?pwd=6666 
提取码：6666

数据集来自于VOC数据集。

需要在终端开启visdom (python -m visdom.server)

调用train()函数进行训练。参数来自于config.py，可以自行修改其中的路径等变量，用于保存模型、读取数据集等目的。
在train()中加入了第一阶段的proposal box，并在visdom中显示。训练完的模型会储存于config.py对应的路径中。

若要实现预测不在VOC数据集的图片时，可以把图片放在faster-rcnn/img文件夹下，main.py中可以设置该文件夹路径。
predict_data_dir = 'C:/Users/admin/Desktop/faster-rcnn/img'
load_path = 'C:/Users/admin/Desktop/fasterrcnn_2023_5_31.pth'
predict_data_dir路径下的模型从百度网盘下载
load_path下的图片数据可以自行下载添加，在这里已经加入了三张图片用于预测
所有预测的结果中会包含：类别标签、得分和boundingbox。




