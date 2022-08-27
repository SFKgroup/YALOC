"""
Process the test set with the .ckpt model in turn.
"""
import argparse
import os

import numpy as np
from PIL import Image
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model.src.config import data_config
from model.src.model import ResNet

set_seed(1)

device_target="CPU"   #使用显卡来加速进行模型训练#注意检查此处需改成自己ckpt文件夹下ckpt模型文件的名字

def eval(checkpoint_path='./ckpt/ResNet-50_125.ckpt'):
    cfg = data_config
    class_indexing = {name: index for index, name in enumerate(sorted(os.listdir(cfg.val_data_path)))}
    indexing_class = {index: name for index, name in enumerate(sorted(os.listdir(cfg.val_data_path)))}
    images = []
    # 存储
    for dir_name in os.listdir(cfg.val_data_path):
        for class_name in os.listdir(os.path.join(cfg.val_data_path, dir_name)):
            images.append([os.path.join(cfg.val_data_path, dir_name, class_name), dir_name])
    net = ResNet(cfg.num_classes)
    net.set_train(False)

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
   # context.set_context()#device_id=device_id)
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(net, param_dict)
    num_images = len(images)
    correct = 0
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    for image_path, label in images:
        images = Image.open(image_path)
        width, height = images.size
        left, upper = (width - 224) // 2, (height - 224) // 2
        images = images.crop((left, upper, left + 224, upper + 224))
        images = np.array(images) / 255  # (224, 224, 3)
        images = (images - mean) / std
        images = np.transpose(images, (2, 0, 1))
        images = Tensor(np.expand_dims(images, 0), dtype=mstype.float32)
        result = net(images).asnumpy()
        # print(image_path, indexing_class[int(np.argmax(result).reshape(-1))], label)
        predict = int(np.argmax(result).reshape(-1))
        #print(f"image_path: {image_path} predict: {indexing_class[predict]} ground_true: {label}")
        if predict == class_indexing[label]:
            correct += 1
    print(checkpoint_path,':',correct)
    print(f"Acc: {correct / num_images * 100}%")
    
if __name__ == '__main__':
    for i in os.listdir('./goods/'):eval('./goods/'+i)