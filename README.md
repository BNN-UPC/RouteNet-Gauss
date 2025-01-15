# RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning

**Carlos Güemes Palau, Miquel Ferrior Galmés, Jordi Paillisse Vilanova, Albert López Brescó, Pere Barlet Ros, Albert Cabellos Aparicio**

This repository is the code of the paper *RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning* (publication pending)

Contact us: *[carlos.guemes@upc.edu](mailto:carlos.guemes@upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract

Network simulation is pivotal in network modeling, assisting with tasks ranging from capacity planning to performance estimation. Traditional approaches such as Discrete Event Simulation (DES) face limitations in terms of computational cost and accuracy. This paper introduces RouteNet-Gauss, a novel integration of a testbed network with a Machine Learning (ML) model to address these challenges. By using the testbed as a hardware accelerator, RouteNet-Gauss generates training datasets rapidly and simulates network scenarios with high fidelity to real-world conditions. Experimental results show that RouteNet-Gauss significantly reduces prediction errors by up to 95\% and achieves a 488x speedup in inference time compared to state-of-the-art DES-based methods. RouteNet-Gauss's modular architecture is dynamically constructed based on the specific characteristics of the network scenario, such as topology and routing. 
This enables it to understand and generalize to different network configurations beyond those seen during training, including networks up to 10x larger. Additionally, it supports Temporal Aggregated Performance Estimation (TAPE), providing configurable temporal granularity and maintaining high accuracy in flow performance metrics. This approach shows promise in improving both simulation efficiency and accuracy, offering a valuable tool for network operators.

## Quick start

1. Please ensure that your OS has installed Python 3 (ideally 3.9)
2. Create the virtual environment and activate the environment:
```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```
3. Then we install the required packages (to avoid issues, make sure to install the specific package versions, especially for TensorFlow):
```bash
pip install tensorflow==2.11.1 numpy==1.24.2 notebook==7.0.7
```
- The following configuration with TensorFlow 2.15 was also used successfully:

```bash
pip install tensorflow==2.15.0 numpy==1.26.3 notebook==7.0.7
```

Once done you can either
- Modify and run [`train.py`](train.py) to train the model
- Evaluate the trained models [`evaluation.ipynb`](evaluation.ipynb).

## Repository structure

The repository contains the following structure:
- `ckpt`: Folder containing the checkpoints used in the paper evaluation.
- `data`: Folder containing the datasets used in the paper. For information on these, read [Datasets information](#datasets-information).
- `normalization`: Folder containing the z-score normalizations used by the trained checkpoints (internal path should match the `ckpt` directory).
- [`evaluation.ipynb`](evaluation.ipynb): an interactive Python notebook file used to evaluate the trained models.
- [`models.py`](models.py) contains the implementation of RouteNet-Gauss.
- [`train.py`](train.py): script to train a RouteNet-Fermi model normally, without fine-tuning.
- [`utils.py`](utils.py) contains auxiliary functions common in the previous files.
- [LICENSE](LICENSE): see the file for the full license.

## Datasets information

In the `data` folder, we can find all the variants of the three datasets used in the paper. Inside each directory, data is split according to the training, validation, and test splits. Then each partition is subdivided into shards, to keep the repository's file size under git and GitHub's limits. *NOTE*: please use the `load_dataset` function from `utils.py` to load these shards correctly. The datasets go as follows:

- `mawi_pcaps`: referred to as Real-World Packet Traces in the paper. Includes training, validation, and test partitions.
- `mawi_pcaps_simulated`: version of the `mawi_pcaps` but run with OMNeT++ simulator. Only includes test partition.
- `trex_multiburst`: referred to as TREX-MULTIBURST in the paper. Includes training, validation, and test partitions.
- `trex_multiburst_filtered`: a subset of samples from `trex_multiburst`. Experiments showed that delay models trained from the subset were more accurate later during the evaluation. This includes only training and validation (the test partition is the same as `trex_multiburst`).
- `trex_multiburst_simulated`: version of the `trex_multiburst` but run with OMNeT++ simulator. Only includes test partition.
- `trex_synthetic`: referred to as TREX-SYNTHETIC in the paper. Includes training, validation, and test partitions.
- `trex_synthetic_filtered`: a subset of samples from `trex_synthetic`. Experiments showed that delay models trained from the subset were more accurate later during the evaluation. This includes only training and validation (the test partition is the same as `trex_synthetic`).
- `trex_synthetic_simulated`: version of the `trex_synthetic` but run with OMNeT++ simulator. Only includes test partition.

## Modifying the `train.py` script

The script contains the default hyperparameters and configurations used in the paper. Follow the comments in the code to perform your modifications. In a summary:

- Use the `RUN_EAGERLY` variable (line 36) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 39) to resume training from the latest recorded checkpoint.
- Modify the experiment configuration to change aspects such as the dataset used (lines 204-219)
- Change the optimizer (and its hyperparameters) and the loss function on lines 231 and 232, respectively.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 233-245) and the call to fit the model (lines 295-307)

## License

See the [file](LICENSE) for the full license:


```
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
```
