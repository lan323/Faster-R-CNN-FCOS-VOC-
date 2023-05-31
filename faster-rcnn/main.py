# -*- coding: utf-8 -*-
from __future__ import  absolute_import
import sys
sys.path.append(r"C:\Users\admin\PycharmProjects\pythonProject3\faster-rcnn")
import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


predict_data_dir = 'C:/Users/admin/Desktop/ObjectDetection/datasets/demo/images'
load_path = 'C:/Users/admin/Desktop/fasterrcnn_2023_5_31.pth'
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn)
trainer = trainer.load(load_path)

for img_file in os.listdir(predict_data_dir):
    img_file = os.path.join(predict_data_dir, img_file)
    img = read_image(img_file, color=True)
    img = t.from_numpy(img)[None]
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
    vis_bbox(at.tonumpy(img[0]),
            at.tonumpy(_bboxes[0]),
            at.tonumpy(_labels[0]).reshape(-1),
            at.tonumpy(_scores[0]).reshape(-1))