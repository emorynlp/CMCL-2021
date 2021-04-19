# Copyright 2018 The Google AI Language Team Authors and DMLC.
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
"""BERT classification/regression datasets."""


__all__ = [
    'EDTask'
]

import os
import mxnet as mx
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
from gluonnlp.data import TSVDataset


class GlueTask:
    """Abstract GLUE task class.

    Parameters
    ----------
    class_labels : list of str, or None
        Classification labels of the task.
        Set to None for regression tasks with continuous real values.
    metrics : list of EValMetric
        Evaluation metrics of the task.
    is_pair : bool
        Whether the task deals with sentence pairs or single sentences.
    label_alias : dict
        label alias dict, some different labels in dataset actually means
        the same. e.g.: {'contradictory':'contradiction'} means contradictory
        and contradiction label means the same in dataset, they will get
        the same class id.
    """
    def __init__(self, class_labels, metrics, is_pair, label_alias=None):
        self.class_labels = class_labels
        self.metrics = metrics
        self.is_pair = is_pair
        self.label_alias = label_alias

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments.

        Returns
        -------
        TSVDataset : the dataset of target segment.
        """
        raise NotImplementedError()

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, TSVDataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset(segment='dev')

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'test', self.get_dataset(segment='test')

class EDTask(GlueTask):

    def __init__(self, surfix):
        is_pair = False
        class_labels = ['caring', 'devastated', 'trusting', 'annoyed', 'hopeful', 'apprehensive',
            'anticipating', 'sentimental', 'afraid', 'jealous', 'anxious', 'nostalgic', 'guilty',
            'prepared', 'disgusted', 'surprised', 'grateful', 'furious', 'sad', 'content', 'angry',
            'excited', 'disappointed', 'embarrassed', 'proud', 'confident', 'impressed', 'lonely',
            'faithful', 'ashamed', 'terrified', 'joyful']
        metric = Accuracy()
        self.surfix = surfix
        super(EDTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train', root='data'):
        if segment == 'dev':
            segment = 'valid'
        return TSVDataset(filename=os.path.join(root, '{}{}'.format(segment, self.surfix)))
