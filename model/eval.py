# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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

from src.config import data_config
from src.model import ResNet

set_seed(1)

parser = argparse.ArgumentParser(description='resnet50_bam')
parser.add_argument('--checkpoint_path', type=str, default='../ckpt/ResNet-156_593.ckpt', help='Checkpoint dir path')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU','CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_id', type=str, default=0, help='Device id.')

args_opt = parser.parse_args()

if __name__ == '__main__':

    cfg = data_config
    cfg.val_data_path = '../val'
    class_indexing = {name: index for index, name in enumerate(sorted(os.listdir(cfg.val_data_path)))}
    indexing_class = {index: name for index, name in enumerate(sorted(os.listdir(cfg.val_data_path)))}
    images = []
    # 存储
    for dir_name in os.listdir(cfg.val_data_path):
        for class_name in os.listdir(os.path.join(cfg.val_data_path, dir_name)):
            images.append([os.path.join(cfg.val_data_path, dir_name, class_name), dir_name])
    net = ResNet(cfg.num_classes)
    net.set_train(False)

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    num_images = len(images)
    correct = 0
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    grand = open('./error.txt','w',encoding='utf-8')
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
        if predict == class_indexing[label]:
            correct += 1
        else:
            print(f"image_path: {image_path} predict: {indexing_class[predict]} ground_true: {label}")
            grand.write(f"image_path: {image_path} predict: {indexing_class[predict]} ground_true: {label}\n")
    grand.close()
    print(correct)
    print(f"Acc: {correct / num_images * 100}%")
