# Neural Network Distillation

In this project, we will investigate another way to build light weight deep networks commonly known as distillation.

*Note: Unfortunately due to 100mb compression limit model weights could not be uploaded using basic public license.*

## Project Structure

```
project/
│
├── data/
│   ├── __init__.py
│   ├── cifar10.py               # Dataset loading and preprocessing
│   ├── mnist.py                 # Dataset loading and preprocessing
│   └── voc.py                   # Dataset loading and preprocessing
│
├── models/
│   ├── __init__.py
│   ├── baseline.py              # MLP
│   ├── unet.py                  # ResNet backbone UNet
│   ├── resnet.py                # ResNet
│   └── resnet_sd.py             # ResNet w/ self distill hook
│
├── utils/
│   ├── __init__.py
│   ├── distill.py               # Reusing distillation loops
│   └── train.py                 # Resuing training/evaluation loops
│
├── notebooks/
│   ├── figures.ipynb            # Dataset analysis
│   ├── simplemlp.ipynb          # MNIST models (Note current version may be experimental)
│   ├── resnet.ipynb             # CIFAR models (Note current version may be experimental)
│   ├── unet.ipynb               # VOC models   (Note current version may be experimental)
│   ├── selfdistillation.ipynb   # CIFAR models (Note current version may be experimental)
│   └── self_distillation.ipynb  # CIFAR models (Note current version may be experimental)
│
├── requirements.txt             # Project dependencies
└── .gitignore                   # Git ignore file
```

## Installation

```bash
git clone https://github.com/Dodogama/project-compresion.git
cd project-compression
pip install -r requirements.txt
```

## Usage

## Results

## License

[MIT](LICENSE)
