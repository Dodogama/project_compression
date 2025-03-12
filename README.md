# Image Captioning with CLIP and DistilGPT2

This project implements an image captioning system using CLIP for image encoding and DistilGPT2 for generating captions. It includes various prompt engineering techniques to improve caption quality.

## Project Structure

```
image_captioning_project/
│
├── data/
│   ├── __init__.py
│   ├── coco_dataset.py          # COCO dataset loading and preprocessing
│   ├── data_utils.py            # Data handling utilities
│   └── augmentations.py         # Image augmentation functions
│
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py          # CLIP image encoder component
│   ├── gpt2_decoder.py          # DistilGPT2 text generation component
│   ├── caption_model.py         # Combined model architecture
│   └── prompt_engineering.py    # Learnable prompt implementations
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
│   ├── caption_generator.py     # Inference pipeline
│   └── beam_search.py           # Advanced decoding strategies
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
│   ├── evaluate.py              # Evaluation script
│   └── generate_captions.py     # Demo script for inference
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
git clone https://github.com/yourusername/image-captioning-project.git
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

This project implements three different prompt engineering techniques:

1. **Learnable Prompts**: Trainable embedding vectors that are prepended to the input sequence
2. **Cross-Attention**: A mechanism that allows the text generator to attend to image features
3. **Prefix Tuning**: Layer-specific prompt vectors that guide the generation process

## Results

| Model Variant | BLEU-4 | CIDEr | SPICE |
|---------------|--------|-------|-------|
| Baseline      | 0.XX   | 0.XX  | 0.XX  |
| Learnable Prompts | 0.XX | 0.XX | 0.XX |
| Cross-Attention | 0.XX | 0.XX | 0.XX |

## License

[MIT](LICENSE)
