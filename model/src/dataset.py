# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision

from .config import data_config


def create_dataset(dataset_path, repeat_num=1, training=True, num_parallel_workers=16, shuffle=None):
    device_num, rank_id = _get_rank_info()

    class_indexing = {name: index for index, name in enumerate(sorted(os.listdir(dataset_path)))}
    """
        agricultural 0
        airplane 1
        baseballdiamond 2
        beach 3
        buildings 4
        chaparral 5
        denseresidential 6
        forest 7
        freeway 8
        golfcourse 9
        harbor 10
        intersection 11
        mediumresidential 12
        mobilehomepark 13
        overpass 14
        parkinglot 15
        river 16
        runway 17
        sparseresidential 18
        storagetanks 19
        tenniscourt 20
    """
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle,
                                         class_indexing=class_indexing)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id, class_indexing=class_indexing)

    assert data_config.image_height == data_config.image_width, "image_height not equal image_width"
    image_size = data_config.image_height
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    # define map operations
    if training:
        transform_img = [
            vision.Decode(),
            vision.RandomCrop(size=int(image_size)),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        transform_img = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = [C.TypeCast(mstype.int32)]

    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=8,
                            operations=transform_img,
                            python_multiprocessing=True)
    data_set = data_set.map(input_columns="label", num_parallel_workers=4, operations=transform_label)

    # apply batch operations
    data_set = data_set.batch(data_config.batch_size, drop_remainder=training)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
