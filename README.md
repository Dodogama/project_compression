# Neural Network Distillation

In this project, we will investigate another way to build light weight deep networks commonly known as distillation.

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
│   ├── simplemlp.ipynb          # MNIST models (Note current version may not be final)
│   ├── resnet.ipynb             # CIFAR models (Note current version may not be final)
│   ├── unet.ipynb               # VOC models   (Note current version may not be final)
│   └── self_distillation.ipynb  # CIFAR models (Note current version may not be final)
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
