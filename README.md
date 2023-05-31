# Faster-R-CNN-FCOS-VOC-

目前现存的代码基本要用到GPU，本代码支持没有GPU的电脑进行训练。

一下链接为百度网盘链接：（存放有训练好的模型与数据集）
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




