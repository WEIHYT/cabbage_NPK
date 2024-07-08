import argparse
import os
import cv2
import numpy as np
import time
import datetime
import torch
import torch.backends.cudnn as cudnn

from dataloaders import cfg_cabbage as cfg
# from models.densenet import *
# from models.densenet_msff import *
from dataloaders.datasets.datasets import LoadStreams, LoadImages
from pathlib import Path
from dataloaders import make_data_loader

def main():

    parser = argparse.ArgumentParser(description="PyTorch DenseNet Detecting")
    parser.add_argument('--in_path', type=str, default=cfg.TEST_DATASET_DIR, help='image to test')
    parser.add_argument('--out_path', type=str, default='inference/cabbage', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='net201', choices=['net121', 'net161', 'net169', 'net201'],
                                                                  help='backbone name (default: net121)')
    parser.add_argument('--compression', type=int, default=0.7, help='network output stride')
    parser.add_argument('--bottleneck', type=str, default=True, help='network output stride')
    parser.add_argument('--drop_rate', type=int, default=0.5, help='dropout rate')
    parser.add_argument('--training', type=str, default=True, help='')
    parser.add_argument('--weights', type=str, default='/root/cabbage/work/cabbage_201/latest.pth', help='saved model')  #'work/densenet.base/latest.pth'
    # parser.add_argument('--weights0', type=str, default='work/densenet.base/latest.pth', help='saved model')  #'work/densenet.base/latest.pth'
    # parser.add_argument('--weights1', type=str, default='work/densenet.base/latest.pth', help='saved model')  #'work/densenet.base/latest.pth'
    # parser.add_argument('--weights2', type=str, default='work/densenet.base/latest.pth', help='saved model')  #'work/densenet.base/latest.pth'
    parser.add_argument('--num_classes', type=int, default=cfg.NUM_CLASSES, help='')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a \
                                                                 comma-separated list of integers only (default=0)')  #0,1
    parser.add_argument('--img_size', type=int, default=(84, 84), help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--save_dir', action='store_true', default='/root/cabbage/DenseNet/inference/cabbage', help='save results to *.txt')
    parser.add_argument('--batch_size', type=int, default=64)  #64
    parser.add_argument('--dataset', type=str, default='oled_data', choices=['oled_data'],
                                                                   help='dataset name (default: pascal)')
    parser.add_argument('--pretrained', action='store_true', default=False, help='True加载预训练权重')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    webcam = args.in_path.isnumeric() or args.in_path.endswith('.txt') or args.in_path.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    results = 'result' + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + '.txt'
    results_file = os.path.join(args.save_dir, results)
    print(results_file)
    torch.manual_seed(args.seed)
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
  
    model_s_time = time.time()
    # Define network
    # model = DenseNet(
    #     backbone    = args.backbone,
    #     compression = args.compression,
    #     num_classes = args.num_classes,
    #     bottleneck  = args.bottleneck,
    #     drop_rate   = args.drop_rate,
    #     sync_bn     = args.sync_bn,
    #     training    = args.training
    # )
    # model = DenseNet121(cfg.NUM_CLASSES,pretrained=args.pretrained,weights=args.weights)

    labels2classes = cfg.labels_to_classes

    model = torch.load(args.weights, map_location='cuda')
    # state_dict = state_dict['model']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    #     new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
    # model.load_state_dict(new_state_dict)

    # ckpt = torch.load(args.weights, map_location='cuda')
    # model.load_state_dict(ckpt['state_dict'])
    files = os.listdir(args.in_path)
    if not os.path.exists(args.in_path):                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(args.in_path)
    # file_path=Path(args.in_path)
    # for files in img_files:
    correct=0
    total=0
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.cuda :
            # model.half()
            model = model.cuda()
    _, _, dataset ,_= make_data_loader(args, **kwargs)
    all_s_time = time.time()
    nTrain = len(dataset)
    i=0

    # for path, image, im0s, vid_cap in dataset :
    #     image  = torch.from_numpy(image).cuda()
    #     image  = image.half() if args.cuda else image.float()  # uint8 to fp16/32
    #     # image /= 255.0              # 0 - 255 to 0.0 - 1.0

    #     if image.ndimension() == 3:
    #         image = image.unsqueeze(0)
    nTrain = len(dataset.dataset)
    print(nTrain)
    for batch_idx, (data, target) in enumerate(dataset):
        s_time = time.time()

        model.eval()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            print(target)
        with torch.no_grad():
            out = model(data)
            # prediction = torch.max(out, 1)[1]
            prediction = torch.max(out, 1)[1]
            print(prediction)

        u_time   = time.time()
        img_time = u_time-s_time
        # print(str(prediction.cpu().numpy()[0]))
        label    = labels2classes[str(prediction.cpu().numpy()[0])]
        print(label)
        print("label: {} time: {}s".format(label, img_time))
        # path_str= Path(path)
        # if label in str(path_str):
        #     i += 1

        # with open(str(results_file), 'a') as f:
        #     f.write("%s %s \n" % (path_str.stem, label))


if __name__ == "__main__":
   main()