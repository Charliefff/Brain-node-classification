# Brain-node-classification

This repository contains deep learning models and related configurations, organized into two architectures: **Convolution-based** and **Graph-based**.

## Project Structure

```
./src
├── conv_based/                # Convolution-based models (CNN)
│   ├── model.py               # CNN model definition
│   ├── config.yaml            # Training hyperparameters and model settings
│   ├── checkpoint/            # Saved model weights
│   ├── dataloader.py          # Data loading functions
│   ├── trainer.py             # Training process
│   ├── main.py                # Main script for execution
│   └── runs/                  # Training logs and TensorBoard records
│
└── graph_based/               # Graph-based models (GNN)
    ├── model.py               # GNN model definition
    ├── config.yaml            # Training hyperparameters and model settings
    ├── dataloader.py          # Data loading functions
    ├── trainer.py             # Training process
    ├── main.py                # Main script for execution
    └── runs/                  # Training logs and TensorBoard records
```

## Models in `conv_based/model.py`

The following models are implemented in `conv_based/model.py`:

- **CNN**
- **CNN_3D**
- **EEGNetV1**
- **EEGNetV4**
- **waveletEEGNet**
- **waveletEEGNetV4**
- **DeepCONV**
- **wavDeepCONV**
- **ShallowConvNet**
- **wavShallowConvNet**

## Models in `graph_based/model.py`

The following models are implemented in `graph_based/model.py`:

- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GAT_GCN** (Hybrid GAT and GCN Model)

## Dataset Formatting

- **Non-wavelet Data:** Must be formatted as `(Channel, Time)`.
- **Wavelet Data:** Must be formatted as `(W, Channel, Time)`, where `W` represents the wavelet dimension.

## Key Files and Directories

- `conv_based/model.py` & `graph_based/model.py`  
  - Defines the architecture of the neural network models.
- `conv_based/config.yaml` & `graph_based/config.yaml`  
  - Stores hyperparameter settings such as learning rate, batch size, etc., which can be modified as needed.
- `dataloader.py`  
  - Handles data loading and preprocessing.
- `trainer.py`  
  - Implements the training and validation process.
- `checkpoint/`  
  - Stores trained model weights, supporting loading and fine-tuning.

## Usage

### 1. Set Up the Environment
```bash
pip install -r requirements.txt
```

### 2. Modify Configuration
Before training, update `config.yaml` with appropriate hyperparameters.

### 3. Train the Model
#### (1) Train the CNN Model
```bash
cd conv_based
python main.py
```
#### (2) Train the GNN Model
```bash
cd graph_based
python main.py
```

### 3. Monitor Training Progress (TensorBoard)
```bash
tensorboard --logdir=conv_based/runs
```
or
```bash
tensorboard --logdir=graph_based/runs
```

## Version Control Guidelines
- **Model changes** should be made in `model.py`
- **Configuration updates** should be made in `config.yaml`

## Contact
