# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from core.config import cfg
from datasets.dataset_catalog import ANN_FN
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import IM_DIR
from datasets.dataset_catalog import IM_PREFIX
from utils.timer import Timer
import utils.boxes as box_utils

logger = logging.getLogger(__name__)


class ImageListDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.image_list = self.load_image_list(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        #category_ids = self.COCO.getCatIds()
        #categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        #self.category_to_id_map = dict(zip(categories, category_ids))
        #self.classes = ['__background__'] + categories
        """self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()"""
    
    def load_image_list(self,image_list_path):
        image_list=[]
        with open(image_list_path,'r') as f:
            for line in f.readlines():
                path,label=line.split(' ')
                label=int(label)
                image_list.append({'labels':label,'image':os.path.join(self.image_directory,path)})
        return image_list