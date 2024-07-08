# -*- coding:utf-8 -*-


##数据集的类别
NUM_CLASSES = 4

#数据集的存放位置
DATASET_DIR       = r'/root/cabbage/DenseNet/VOC_NPK'
TRAIN_DATASET_DIR = r'/root/cabbage/DenseNet/VOC_NPK/train'
VAL_DATASET_DIR   = r'/root/cabbage/DenseNet/VOC_NPK/val'
TEST_DATASET_DIR  = r'/root/cabbage/DenseNet/VOC_NPK/test'

# DATASET_DIR        = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data'
# TRAIN_DATASET_DIR  = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/train'
# VAL_DATASET_DIR    = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/val'
# TEST_DATASET_DIR   = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/inference/image's


#这里需要加入自己的最终预测对应字典，例如:'0': '花'
labels_to_classes = {
    '0' : 'EW',
    '1' : 'FN',
    '2' : 'LW',
    '3' : 'N'
}
