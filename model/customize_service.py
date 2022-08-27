import threading
import os

import logging
import numpy as np

import mindspore
import mindspore.nn as nn
from PIL import Image
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model_service.model_service import SingleNodeService
from PIL import Image

from src.config import data_config
from src.model import simple_cnn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class class_service(SingleNodeService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.predict_dict = {0:'agricultural', 1:'airplane', 2:'baseballdiamond',3:'beach',4:'buildings',
                             5:'chaparral',6:'denseresidential',7:'forest',8:'freeway',9:'golfcourse',10:'harbor',11:'intersection',12:'mediumresidential',
                             13:'mobilehomepark',14:'overpass',15:'parkinglot',16:'river',17:'runway',18:'sparseresidential',19:'storagetanks',20:'tenniscourt'}
        logger.info("self.model_name:%s self.model_path: %s", self.model_name,
                    self.model_path)
        self.cfg = data_config
        self.network = None
        # 非阻塞方式加载模型，防止阻塞超时
        self.load_model()
        # thread.start()

    def load_model(self):
        logger.info("load network ... \n")
        self.network = simple_cnn(self.cfg.num_classes)
        self.network.set_train(False)
        ckpt_file = self.model_path + "/train_simple_cnn_1-50_13.ckpt"
        logger.info("ckpt_file: %s", ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self.network, param_dict)
        # 模型预热，否则首次推理的时间会很长
        self.network_warmup()
        logger.info("load network successfully ! \n")

    def network_warmup(self):
        # 模型预热，否则首次推理的时间会很长
        logger.info("warmup network ... \n")
        images = np.array(np.random.randn(1, 3, 224, 224), dtype=np.float32)
        inputs = Tensor(images, dtype=mstype.float32)
        inference_result = self.network(inputs)
        logger.info("warmup network successfully ! \n")


    def _preprocess(self, input_data):
        preprocessed_result = {}
        images = []
        mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
        std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
        for k, v in input_data.items():
            for file_name, file_content in v.items():
                image1 = Image.open(file_content).convert("RGB")
                width, height = image1.size
                left, upper = (width - 224) // 2, (height - 224) // 2
                image1 = image1.crop((left, upper, left + 224, upper + 224))
                image1 = np.array(image1) / 255  # (224, 224, 3)
                image1 = (image1 - mean) / std
                image1 = np.transpose(image1, (2, 0, 1))
                images.append(image1)
        images = np.array(images, dtype=np.float32)
        logger.info(images.shape)
        images.resize([len(input_data), 3, 224, 224])
        logger.info("images shape: %s", images[0].shape)
        inputs = Tensor(images, dtype=mstype.float32)
        logger.info("inputs shape: %s", inputs.shape)
        preprocessed_result['images'] = inputs
        return preprocessed_result

    def _inference(self, preprocessed_result):
        logger.info("inputs shape: %s", preprocessed_result['images'].shape)
        predict = self.network(preprocessed_result['images']).asnumpy()
        inference_result = self.predict_dict.get(int(np.argmax(predict).reshape(-1)))
        return inference_result

    def _postprocess(self, inference_result):
        result = {}
        result["label"] = inference_result
        return result