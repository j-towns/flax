# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet 32 input pipeline."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class DataSource(object):
  """CIFAR10 or ImageNet 32 data source."""

  # For ImageNet 32, use dataset_name='imagenet_resized/32x32'.
  def __init__(self, train_batch_size, eval_batch_size, dataset_name='cifar10',
      shuffle_seed=1):
    assert dataset_name in {'imagenet_resized/32x32', 'cifar10'}, \
      'only imagenet 32 and cifar10 are supported'
    if dataset_name == 'imagenet_resized/32x32':
      self.TRAIN_IMAGES = 1281149
      self.EVAL_IMAGES = 49999
    else:
      self.TRAIN_IMAGES = 50000
      self.EVAL_IMAGES = 10000

    # Training set
    train_ds = tfds.load(dataset_name, split='train').cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(16 * train_batch_size, seed=shuffle_seed)

    def process_sample(x):
      image = tf.cast(x['image'], tf.float32)
      image = image / 127.5 - 1
      batch = {'image': image, 'label': x['label']}
      return batch

    train_ds = train_ds.map(process_sample, num_parallel_calls=128)
    train_ds = train_ds.batch(train_batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(10)
    self.train_ds = train_ds

    # Test set
    eval_ds = tfds.load(dataset_name, split='test').cache()
    eval_ds = eval_ds.map(process_sample, num_parallel_calls=128)
    # Note: samples will be dropped if the number of test samples
    # (EVAL_IMAGES=10000) is not divisible by the evaluation batch
    # size
    eval_ds = eval_ds.batch(eval_batch_size, drop_remainder=True)
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.prefetch(10)
    self.eval_ds = eval_ds
