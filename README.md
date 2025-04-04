# Neural Network Distillation

In this project, we will investigate another way to build light weight deep networks commonly known as distillation.

## Project Structure

```
project/
│
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── data_utils.py            # Data handling utilities
│   └── augmentations.py         # Image augmentation functions
│
├── models/
│   ├── __init__.py
│   └── ResNet50.py         # Image augmentation functions
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop and optimization
│   ├── loss_functions.py        # Custom loss implementations
│   └── config.py                # Training hyperparameters
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # BLEU, CIDEr, SPICE implementations
│   ├── evaluator.py             # Evaluation pipeline
│   └── visualizations.py        # Result visualization tools
│
├── inference/
│   ├── __init__.py
│   └── infer.py           # Probably don't need this
│
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py         # Logging functionality
│   └── checkpoint_utils.py      # Model saving and loading
│
├── notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis
│   ├── model_training.ipynb     # Interactive training notebook
│   └── results_analysis.ipynb   # Performance analysis
│
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
│
├── configs/
│   ├── base_config.yaml         # Base configuration
│   ├── prompt_tuning.yaml       # Prompt tuning configuration
│   └── cross_attention.yaml     # Cross-attention configuration
│
├── requirements.txt             # Project dependencies
├── setup.py                     # Package installation
└── .gitignore                   # Git ignore file
```

## Installation

```bash
git clone https://github.com/Dodogama/project-compresion.git
cd image-captioning-project
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config configs/base_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model-path checkpoints/model.pt --data-dir data/coco
```

### Generating Captions

```bash
python scripts/generate_captions.py --image-path path/to/image.jpg --model-path checkpoints/model.pt
```

## Prompt Engineering Approaches

This project implements neural network distillation:

1. **Learnable Prompts**: Trainable embedding vectors that are prepended to the input sequence

## Results

## License

[MIT](LICENSE)
