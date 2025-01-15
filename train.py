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

from models import RouteNetGauss
from data import load_dataset
from utils import prepare_targets_and_mask, load_dataset

from random import seed
import numpy as np

# Set all seeds
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)
# RELOAD_WEIGHTS -> True to continue training from a checkpoint
RELOAD_WEIGHTS = False


def get_z_scores_dict(
    ds,
    params,
    include_y=None,
    flatten=False,
    summarize=-1,
    store_res_path=None,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. Later used by the
    models for the z-score normalization.

    Parameters
    ----------
    ds
        tensorflow.data.Dataset
    params
        list of strings indicating the parameters to extract the features from
    include_y, optional
        Indicates if to also extract the features of the output variable. Inputs
        indicate the string key used on the return dict. If None, it is not included.
    flatten, optional
        If true, mean and std are computed globally for all dimensions in each feature.
        Otherwise, the values are computed for each dimension separately.
    summarize, optional
        If > 0, only uses the first n samples to compute the mean and std.
    store_res_path, optional
        If not None, the results are stored in the path indicated by the string.
        The dictionary is stored using the pickle library.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if store_res_path is not None and check_existing:
        if os.path.exists(store_res_path):
            with open(store_res_path, "rb") as ff:
                return pickle.load(ff)

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    next_sample = next(iter_ds)
    sample, label = next_sample[0], next_sample[1]
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    if summarize > 0:
        max_samples = summarize - 1

    # Include the rest of the samples
    for ii, (sample, label) in enumerate(map(lambda x: (x[0], x[1]), iter_ds)):
        if summarize > 0 and ii > max_samples:
            break
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    axis = None if flatten else 0
    for param, param_list in params_lists.items():
        scores[param] = [np.mean(param_list, axis=axis), np.std(param_list, axis=axis)]
        # Check if std is 0
        if scores[param][1].size == 1 and scores[param][1] == 0:
            print(f"Z-score normalization Warning: {param} has a std of 0.")
            scores[param][1] = 1
        elif scores[param][1].size > 1 and np.any(scores[param][1] == 0):
            print(
                f"Z-score normalization Warning: Several values of {param} has a std of 0."
            )
            scores[param][1][scores[param][1] == 0] = 1

    if store_res_path is not None:
        store_res_dir, _ = os.path.split(store_res_path)
        os.makedirs(store_res_dir, exist_ok=True)
        with open(store_res_path, "wb") as ff:
            pickle.dump(scores, ff)

    return scores


def get_positional_mape(pos: int, name: str) -> callable:
    """Returns the MAPE metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the MAPE metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def mape_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

    mape_metric.__name__ = f"mape_{name}_metric"
    return mape_metric


def r2_score(y_true: tf.TensorSpec, y_pred: tf.TensorSpec) -> float:
    """Returns the R2 score between the true and predicted values.

    Parameters
    ----------
    y_true : tf.TensorSpec
        True values
    y_pred : tf.TensorSpec
        Predicted values

    Returns
    -------
    float
        R2 score
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def get_positional_r2(pos: int, name: str) -> callable:
    """Returns the R2 metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the R2 metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def r2_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        return r2_score(y_true, y_pred)

    r2_metric.__name__ = f"r2_{name}_metric"
    return r2_metric


# Name of the dataset
ds_name = "data_mawi_pcaps"
# Experiment identifier
experiment_name = "paper_weights"
# Target to be predicted
target = "delay"
# List of subtargets (avg or percentiles)
targets = [
    f"flow_avg_{target}",
    f"flow_p50_{target}",
    f"flow_p90_{target}",
    f"flow_p95_{target}",
    f"flow_p99_{target}",
]
mask = f"flow_has_{target}"
experiment_path = f"{experiment_name}/{ds_name}/{RouteNetGauss.__name__}/{target}"

ds_train = (
    load_dataset(f"{ds_name}/training")
    .map(prepare_targets_and_mask(targets, mask))
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
)
ds_val = load_dataset(f"{ds_name}/validation").map(
    prepare_targets_and_mask(targets, mask)
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
loss = tf.keras.losses.MeanAbsolutePercentageError()
model = RouteNetGauss(
    output_dim=len(targets),
    mask_field=mask,
    use_true_trans_pkts=target == "delay",
    z_scores=get_z_scores_dict(
        ds_train,
        RouteNetGauss.z_scores_fields,
        summarize=500,
        flatten=True,
        store_res_path=os.path.join("normalization", experiment_path, "z_scores.pkl"),
        check_existing=True,
    ),
)

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=[
        get_positional_mape(0, "avg"),
        get_positional_mape(1, "p50"),
        get_positional_mape(2, "p90"),
        get_positional_mape(3, "p95"),
        get_positional_mape(4, "p99"),
        get_positional_r2(0, "avg"),
        get_positional_r2(1, "p50"),
        get_positional_r2(2, "p90"),
        get_positional_r2(3, "p95"),
        get_positional_r2(4, "p99"),
    ],
)

ckpt_dir = f"ckpt/{experiment_path}"
latest = tf.train.latest_checkpoint(ckpt_dir)
if RELOAD_WEIGHTS and latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    save_best_only=False,
    save_weights_only=True,
    save_freq="epoch",
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"tensorboard/{experiment_path}", histogram_freq=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    start_from_epoch=4,
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
)

model.fit(
    ds_train,
    epochs=300,
    steps_per_epoch=500,
    validation_data=ds_val,
    callbacks=[
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    use_multiprocessing=True,
)
