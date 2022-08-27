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
network config setting, will be used in main.py
"""

from easydict import EasyDict as edict

data_config = edict({
    'name': 'landuse',
    'pre_trained': False,
    'use_dataset_sink': True,
    'num_classes': 21,
    'lr_init': 0.001,
    'batch_size': 32,
    'epoch_size': 300,
    'momentum': 0.0,
    'weight_decay': 0,
    'image_height': 224,
    'image_width': 224,
    'data_path': './train/',
    'val_data_path': './val/',
    'keep_checkpoint_max': 80,
    'checkpoint_path': None,

    # loss related
    'is_dynamic_loss_scale': 0,
    'loss_scale': 1024,
})
