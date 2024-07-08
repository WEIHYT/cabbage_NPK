# -*- coding:utf-8 -*-


##数据集的类别
NUM_CLASSES = 39

#数据集的存放位置
DATASET_DIR       = r'/root/cabbage/DenseNet/Plant_Village'
TRAIN_DATASET_DIR = r'/root/cabbage/DenseNet/Plant_Village/Plant_leave_diseases_dataset_with_augmentation'
VAL_DATASET_DIR   = r'/root/cabbage/DenseNet/VOC_lettuce/test'
TEST_DATASET_DIR  = r'/root/cabbage/yolov5/runs/detect/exp/crop/12_jpg.rf.3fa886da7134c2c91c4d776445790d12'

# DATASET_DIR        = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data'
# TRAIN_DATASET_DIR  = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/train'
# VAL_DATASET_DIR    = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/val'
# TEST_DATASET_DIR   = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/inference/image's


#这里需要加入自己的最终预测对应字典，例如:'0': '花'
labels_to_classes = {
    '0' : 'Apple___Apple_scab',
    '1' : 'Apple___Black_rot',
    '2' : 'Apple___Cedar_apple_rust',
    '3' : 'Apple___healthy',
    '4' : 'Background_without_leaves',
    '5' : 'Blueberry___healthy',
    '6' : 'Cherry___healthy',
    '7' : 'Cherry___Powdery_mildew',
    '8' : 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    '9' : 'Corn___Common_rust',
    '10' : 'Corn___healthy',
    '11' : 'Corn___Northern_Leaf_Blight',
    '12' : 'Grape___Black_rot',
    '13' : 'Grape___Esca_(Black_Measles)',

    '14' : 'Orange___Haunglongbing_(Citrus_greening)',
    '15' : 'Peach___Bacterial_spot',
    '16' : 'Peach___healthy',
    '17' : 'Pepper,_bell___Bacterial_spot',
    '18' : 'Pepper,_bell___healthy',
    '19' : 'Grape___healthy',
    '20' : 'Potato___Early_blight',
    '21' : 'Potato___healthy',
    '22' : 'Potato___Late_blight',
    '23' : 'Raspberry___healthy',
    '24' : 'Soybean___healthy',
    '25' : 'Squash___Powdery_mildew',
    '26' : 'Strawberry___healthy',
    '27' : 'Strawberry___Leaf_scorch',
    '28' : 'Tomato___Bacterial_spot',
    '29' : 'Tomato___Early_blight',
    '30' : 'Tomato___healthy',
    '31' : 'Tomato___Late_blight',
    '32' : 'Tomato___Leaf_Mold',
    '33' : 'Tomato___Septoria_leaf_spot',
    '34' : 'Tomato___Spider_mites Two-spotted_spider_mite',
    '35' : 'Tomato___Target_Spot',
    '36' : 'Tomato___Tomato_mosaic_virus',
    '37' : 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    '38' : 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
}
