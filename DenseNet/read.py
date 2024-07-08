import cv2
import numpy as np
import os
from pathlib import Path


def read(img_path,txt_path):
    for filename in os.listdir(img_path):
        path =img_path+ "/" + filename
        img = cv2.imread(path) 
        path_str=Path(path)
        with open(str(txt_path)+'/'+path_str.stem+".txt", "r") as f:  
            data =np.loadtxt(f)
            # data = np.genfromtxt(f,dtype=[int, float,float,float,float,float])  # 将文件中数据加载到data数组里
            print(len(data))
            # if len(data)==1:
            #     x1=int(xyxy[1])
            #     y1=int(xyxy[2])
            #     x2=int(xyxy[3])
            #     y2=int(xyxy[4])
            # elif len(data)>1:
            #     for xyxy in data:
            #         print(xyxy)
            #         x1=int(xyxy[1])
            #         y1=int(xyxy[2])
            #         x2=int(xyxy[3])
            #         y2=int(xyxy[4])
            # print(x1,y1,x2,y2)
       


if __name__ == "__main__":
   img_path="/home/xu519/HYT/cabbage/yolov5/VOC_Lettuce/images/images_test"
   txt_path="/home/xu519/HYT/cabbage/yolov5/runs/detect/exp/labels"
   read(img_path,txt_path)