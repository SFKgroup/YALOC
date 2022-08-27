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
Train.
"""
import argparse
import os

from mindspore import Tensor, context, set_seed
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.callback import EvaluateCallBack
from src.config import data_config
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth
from src.model import ResNet

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend or GPU. (Default: None)')
    parser.add_argument('--device_num', type=int, default=1, help='number of devices. (Default: 1)')

    parser.add_argument('--is_distributed', type=int, default=0,
                        help='Whether to use distributed GPU training. (Default: 0)')

    args_opt = parser.parse_args()
    cfg = data_config

    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == 'Ascend':
        context.set_context(enable_graph_kernel=True)

        device_num = int(os.getenv('DEVICE_NUM', '1'))
        device_id = int(os.getenv('DEVICE_ID', '0'))

        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=args_opt.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        device_num = 1
        device_id = 0
        if args_opt.is_distributed:
            init()
            device_num = get_group_size()
            device_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    dataset = create_dataset(cfg.data_path, 1)

    batch_num = dataset.get_dataset_size()

    net = ResNet(num_classes=cfg.num_classes)
    # Continue training if set pre_trained to be True
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net, param_dict)

    loss_scale_manager = None


    def get_param_groups(network):
        """ get param groups """
        decay_params = []
        no_decay_params = []
        for x in network.trainable_params():
            parameter_name = x.name
            if parameter_name.endswith('.bias'):
                # all bias not using weight decay
                no_decay_params.append(x)
            elif parameter_name.endswith('.gamma'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            elif parameter_name.endswith('.beta'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            else:
                decay_params.append(x)

        return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


    if cfg.is_dynamic_loss_scale:
        cfg.loss_scale = 1

    opt = Momentum(params=get_param_groups(net),
                   learning_rate=Tensor(cfg.lr_init, dtype=mstype.float32),
                   momentum=cfg.momentum,
                   loss_scale=cfg.loss_scale)

    loss = CrossEntropySmooth(sparse=True, reduction="mean", num_classes=cfg.num_classes)

    if args_opt.device_target == 'Ascend':
        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)
    else:
        loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=True,
                  loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 2, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt/"
    ckpoint_cb = ModelCheckpoint(prefix="ResNet", directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    val_dataset = create_dataset(cfg.val_data_path, training=False)
    eval_cb = EvaluateCallBack(model=model, eval_dataset=val_dataset)
    cbs = [time_cb, ckpoint_cb, loss_cb, eval_cb]
    model.train(cfg.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=cfg.use_dataset_sink)
    print("train success")
