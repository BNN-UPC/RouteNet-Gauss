"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from typing import List


def seg_to_global_reshape(tensor, num_dims=3):
    assert num_dims > 1
    perms = [1, 0] + list(range(2, num_dims))
    total_flows = tf.shape(tensor)[0] * tf.shape(tensor)[1]
    if num_dims == 2:
        new_shape = (total_flows,)
    else:
        new_shape = tf.concat([[total_flows], tf.shape(tensor)[2:]], axis=0)
    return tf.reshape(tf.transpose(tensor, perms), new_shape)


def prepare_targets_and_mask(targets: List[str], mask: str) -> callable:
    """Obtains map function to prepare the targets of the dataset for the current model.

    Parameters
    ----------
    targets : List[str]
        List of features to be selected as targets, in that order.
    mask : str
        Mask feature used to determine which windows in the temporal dimension are
        valid.

    Returns
    -------
    callable
        Map function to be called by the tf.data.Dataset.map method.
    """

    def modified_target_map(x, y):
        reshaped_mask = tf.expand_dims(seg_to_global_reshape(x[mask], num_dims=2), 1)

        return x, tf.concat(
            [
                tf.reshape(
                    tf.boolean_mask(seg_to_global_reshape(x[target]), reshaped_mask),
                    (-1, 1),
                )
                for target in targets
            ],
            axis=1,
        )

    return modified_target_map


def load_dataset(name: str, data_path: str = "data") -> tf.data.Dataset:
    """Function to unshard and load a dataset from the data folder.

    Parameters
    ----------
    name : str
        Name of the dataset and partition [training/validation/test] to load, in format
        '{name}/{partition}'.
    
    data_path : str
        Path to the data folder. By default, it is 'data', which assumes the working
        directory is the root of the project.

    Returns
    -------
    tf.data.Dataset
        The dataset loaded from the shards.
    """
    path = os.path.join(data_path, name)
    shards = os.listdir(path)
    assert len(shards) > 0, f"Invalid dataset: {name}"

    ds = tf.data.Dataset.load(path + "/0", compression="GZIP")
    for ii in range(1, len(shards)):
        ds = ds.concatenate(tf.data.Dataset.load(path + f"/{ii}", compression="GZIP"))

    return ds.prefetch(tf.data.experimental.AUTOTUNE)
